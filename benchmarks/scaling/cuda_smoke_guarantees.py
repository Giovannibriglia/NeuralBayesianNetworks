"""CUDA smoke test of the guarantee code paths (round 5): every new or fixed
path runs with the model and data on the accelerator and is compared with
the same call on CPU.  Usage: ``python cuda_smoke_guarantees.py [device]``."""
from __future__ import annotations

import copy
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
DEV = sys.argv[1] if len(sys.argv) > 1 else ("cuda" if torch.cuda.is_available() else "cpu")
results = []


def check(name, fn):
    t0 = time.perf_counter()
    try:
        msg = fn()
        results.append((name, "ok", msg, time.perf_counter() - t0))
    except Exception as exc:  # noqa: BLE001
        results.append((name, "FAIL", f"{type(exc).__name__}: {exc}"[:300], time.perf_counter() - t0))


def model_on(dev, seed=0, **kw):
    from tests.unit.test_conditioning import _dense_bn
    m, names = _dense_bn(seed, **kw)
    if dev != "cpu":
        m = copy.deepcopy(m)
        m.to(dev)
    return m, names


def t_certified():
    from nbn.inference.certified import CertifiedEngine, certified_query
    from nbn.inference.tensor_ve import TensorVariableElimination
    m, names = model_on(DEV, n=12, k=3, window=3)
    tgt, ev = names[2], {names[-1]: 1, names[7]: 0}
    exact = TensorVariableElimination().query(m, [tgt], ev).cpu().double()
    out = []
    for kw in ({"n_samples": 50_000}, {"collapse": "auto"}, {"target_width": 0.2}):
        a = certified_query(m, tgt, ev, **kw)
        inside = bool(((exact >= a.lower - 1e-12) & (exact <= a.upper + 1e-12)).all())
        assert inside, (kw, exact, a.lower, a.upper)
        out.append(f"{list(kw)[0]} w={a.max_width:.3f}")
    try:
        certified_query(m, tgt, {names[-1]: torch.tensor([0, 1], device=DEV)})
        raise AssertionError("batched evidence accepted")
    except ValueError:
        pass
    assert CertifiedEngine.applicable(m, [tgt], ev)
    return "; ".join(out)


def t_router_batched():
    from nbn.inference.hybrid import HybridRouter
    m, names = model_on(DEV, n=10, k=3, window=3)
    r = HybridRouter(fallback="certified", n_samples=4000)
    r._select = lambda *a, **k: r._lw
    q = r.query(m, [names[2]], {names[-1]: torch.tensor([0, 1, 2], device=DEV)})
    assert r.last_certificate is None
    r.query(m, [names[2]], {names[-1]: 1})
    assert r.last_certificate is not None
    return f"batched shape {tuple(q.shape)} on {q.device}"


def t_credal_pac():
    from nbn.uncertainty.credal import pac_query, pac_sets, credal_query
    from nbn.inference.tensor_ve import TensorVariableElimination
    m, names = model_on(DEV, n=8, k=3, window=2, p=0.8)
    data = m.sample(20_000)
    tgt, ev = names[1], {names[-1]: 2}
    exact = TensorVariableElimination().query(m, [tgt], ev).cpu().double()
    sets = pac_sets(m, data, delta=0.1)
    q = credal_query(m, sets, tgt, ev)
    q_cpu = credal_query(copy.deepcopy(m).to("cpu") if DEV != "cpu" else m,
                         pac_sets(copy.deepcopy(m).to("cpu") if DEV != "cpu" else m,
                                  {k: v.cpu() for k, v in data.items()}, delta=0.1), tgt, ev)
    assert torch.allclose(q.lower, q_cpu.lower, atol=1e-5) and torch.allclose(q.upper, q_cpu.upper, atol=1e-5)
    pq = pac_query(m, data, tgt, ev, delta=0.1, kind="l1")
    cover = bool(((exact >= q.lower) & (exact <= q.upper)).all())
    return f"interval w={q.max_width:.3f} (cpu same), l1 w={pq.max_width:.3f}, covers truth: {cover}"


def t_query_many():
    from nbn.inference import query_many
    from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
    m, names = model_on(DEV, n=12, k=3, window=3)
    qs = [([names[i]], {names[-1]: i % 3}) for i in range(6)]
    out = query_many(m, qs, engine=LikelihoodWeightingEngine(n_samples=4000), workers=3)
    assert len(out) == 6 and all(torch.isfinite(o).all() for o in out)
    return f"{len(out)} answers on {out[0].device}"


def t_portfolio():
    from nbn.learning import select_mechanisms
    from nbn.mechanisms import CategoricalTableMechanism, NeuralCategoricalMechanism
    m, names = model_on(DEV, n=4, k=3, window=2, p=1.0)
    data = m.sample(3000)
    rep = select_mechanisms(m, data, {"table": CategoricalTableMechanism,
                                      "neural": NeuralCategoricalMechanism},
                            nodes=names[1:3], max_epochs=20, min_epochs=5)
    assert rep.train_index is not None and not bool(torch.isin(rep.train_index, rep.val_index).any())
    return str(rep.chosen)


def t_export_and_unseen(tmp="/tmp/alfredo_cuda_export"):
    import shutil
    from nbn.structure import HypothesisSet, export_factored_sets, joint_row_bounds, load_factored_sets
    from tests.unit.structure.test_agreement import N, VARIABLES, ring_data
    shutil.rmtree(tmp, ignore_errors=True)
    data = ring_data(n_episodes=50, device=DEV)
    cands = {f"m{i}'": [(f"m{i}", f"m{(i - 1) % N}", f"rep{i}")] for i in range(N)}
    hs = HypothesisSet(VARIABLES, cands)
    fs = hs.factor_sets(data, delta=0.1, sampling="trajectory")
    out = export_factored_sets(tmp, fs, hs, [f"m{i}" for i in range(N)], [f"rep{i}" for i in range(N)],
                               extra_events={"structure": 0.01})
    ex = load_factored_sets(out)
    hyp = next(ex.hypotheses())
    lo, up, _ = joint_row_bounds(ex, hyp, 0)
    assert (lo <= up).all()
    return f"level {ex.manifest['composition']['level']:.3f}, joint row 0 width {float((up - lo).max()):.3f}"


def t_calibration():
    from nbn.scm import AdditiveNoiseMechanism, StructuralCausalModel, check_held_out
    from nbn.scm.sets import cdf_band_set
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(2000, 1, generator=gen)
    data = {"x": x, "y": 2 * x + 0.5 * torch.randn(2000, 1, generator=gen)}
    model = StructuralCausalModel([("x", "y")], dict.fromkeys(data, ("continuous", 1)), device=DEV)
    model.set_mechanism("y", AdditiveNoiseMechanism("linear"))
    model.auto_mechanisms(default_continuous="lg")
    model.fit({k: v[:1000] for k, v in data.items()})
    held = {k: v[1000:] for k, v in data.items()}
    band = cdf_band_set(model, held, "y", fit_ids=torch.arange(1000), ids=torch.arange(1000, 2000))
    lo, up = band.noise_cdf_bounds(torch.tensor([0.0], device=DEV), mechanism=model.mechanisms["y"])
    try:
        check_held_out(torch.arange(10), torch.arange(5, 15))
        raise AssertionError("overlap accepted")
    except ValueError:
        pass
    return f"F(0) in [{float(lo):.3f}, {float(up):.3f}]"


def t_noise_polytope_tightened():
    from tests.unit.structure.test_export import _health_model
    from nbn.scm import noise_parameter_set
    from nbn.uncertainty.sets import NoiseParameterSet
    model, data = _health_model()
    nps = noise_parameter_set(model, data, ["h'", "g'"], delta=0.05)
    dt = torch.float32 if DEV == "mps" else nps.lower.dtype      # the set's tensors on the device
    nps = NoiseParameterSet(expressions=nps.expressions.to(DEV, dt), lower=nps.lower.to(DEV, dt),
                            upper=nps.upper.to(DEV, dt), index=nps.index, level=nps.level,
                            guarantee=nps.guarantee, unseen=nps.unseen)
    t = nps.tightened()
    assert (t.lower <= t.upper).all() and (t.lower >= nps.lower).all()
    return f"dtype {t.lower.dtype} on {t.lower.device}, mean width {float((t.upper - t.lower).mean()):.3f}"


def t_safe_lp():
    from nbn.uncertainty.confounded import IVProgram, RegimeBox
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(6)).reshape(1, 2, 3)          # one regime: never refuted
    prog = IVProgram(RegimeBox.point(p))
    lo, hi = prog.rows()
    return f"IV rows ok, width {float((hi - lo).max()):.3f}"


for name, fn in [("certified", t_certified), ("router_batched", t_router_batched),
                 ("credal_pac", t_credal_pac), ("query_many", t_query_many),
                 ("portfolio", t_portfolio), ("export_unseen", t_export_and_unseen),
                 ("calibration", t_calibration), ("noise_tightened", t_noise_polytope_tightened),
                 ("safe_lp", t_safe_lp)]:
    check(name, fn)

print(f"device={DEV} torch={torch.__version__}")
for name, status, msg, secs in results:
    print(f"{status:4s} {name:16s} {secs:6.2f}s  {msg}")
print("ALL_OK" if all(s == "ok" for _, s, _, _ in results) else "SOME_FAILED")
