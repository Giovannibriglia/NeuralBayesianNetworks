"""Regressions for the open risks left by the guarantee review (October 2026)."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from nbn.inference.certified import CertifiedEngine, certified_query
from nbn.inference.hybrid import HybridRouter
from nbn.uncertainty.sets import IntervalSet, SampleSet
from tests.unit.test_conditioning import _dense_bn


# ---------------------------------------------------------------- certificates

def test_certified_query_refuses_batched_evidence_instead_of_crashing():
    m, names = _dense_bn(0, n=8, k=3, window=3, p=0.7)
    for kw in ({}, {"collapse": "auto"}, {"target_width": 0.5}):
        with pytest.raises(ValueError, match="query_batch"):
            certified_query(m, names[2], {names[-1]: torch.tensor([0, 1])}, **kw)


def test_router_answers_a_batched_query_with_lw_not_the_certificate():
    m, names = _dense_bn(0, n=8, k=3, window=3, p=0.7)
    r = HybridRouter(fallback="certified", n_samples=2000)
    r._select = lambda *a, **k: r._lw                     # force the sampling route
    out = r.query(m, [names[2]], {names[-1]: torch.tensor([0, 1, 2])})
    assert out.shape[-1] == 3 and r.last_certificate is None
    r.query(m, [names[2]], {names[-1]: 1})                # one row: certified
    assert r.last_certificate is not None




def test_evidence_mechanism_whose_log_prob_differs_from_its_table_is_refused():
    m, names = _dense_bn(0, n=6, k=3, window=2, p=1.0)
    ev_node = names[-1]
    mech = m.mechanisms[ev_node]
    orig = mech.log_prob

    def skewed(x, parents):                               # log_prob != tabulate
        return orig(x, parents) + 0.3
    mech.log_prob = skewed
    with pytest.raises(ValueError, match="log_prob"):
        certified_query(m, names[1], {ev_node: 1}, n_samples=1000)
    assert not CertifiedEngine.applicable(m, [names[1]], {ev_node: 1})
    mech.log_prob = orig
    assert CertifiedEngine.applicable(m, [names[1]], {ev_node: 1})


# ---------------------------------------------------------------- export

def _two_node_export(tmp_path, set_b, **kw):
    from nbn import NeuralBayesianNetwork
    from nbn.structure import export_factored_sets, load_factored_sets
    m = NeuralBayesianNetwork([("s", "s'")], variables={"s": ("discrete", 2), "s'": ("discrete", 2)},
                              device="cpu")
    out = export_factored_sets(tmp_path / "e", {"s'": set_b}, m, ["s"], [], **kw)
    return load_factored_sets(out)


def test_joint_row_bounds_use_that_rows_sum_to_one(tmp_path):
    from nbn.structure import joint_row_bounds
    box = IntervalSet(level=0.95, guarantee="pac", unseen=torch.zeros(2, dtype=torch.bool),
                      lower=torch.tensor([[0.6, 0.0], [0.0, 0.0]], dtype=torch.float64),
                      upper=torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64))
    ex = _two_node_export(tmp_path, box)
    lo, up, _ = joint_row_bounds(ex, {"s'": ex.manifest["factors"][0]["id"]}, 0)
    assert up[1] <= 0.4 + 1e-12 and lo[0] >= 0.6 * (1 - 1e-12)   # 1 - 0.6, not 1


def test_extra_events_enter_the_union_bound(tmp_path):
    box = IntervalSet(level=0.95, guarantee="pac", unseen=torch.zeros(2, dtype=torch.bool),
                      lower=torch.full((2, 2), 0.2, dtype=torch.float64),
                      upper=torch.full((2, 2), 0.8, dtype=torch.float64))
    ex = _two_node_export(tmp_path, box, extra_events={"structure_test": 0.02})
    comp = ex.manifest["composition"]
    assert comp["delta_total"] == pytest.approx(0.07) and comp["level"] == pytest.approx(0.93)
    assert "structure_test" in comp["events"] and comp["extra_events"] == {"structure_test": 0.02}
    jsonschema = pytest.importorskip("jsonschema")
    from nbn.structure.export import SCHEMA_FILE
    jsonschema.validate(ex.manifest, json.loads(SCHEMA_FILE.read_text()))
    with pytest.raises(ValueError, match="clashes"):
        _two_node_export(tmp_path / "x", box, extra_events={"f0": 0.01})
    with pytest.raises(ValueError, match="probability"):
        _two_node_export(tmp_path / "y", box, extra_events={"s": 1.5})


def test_rows_without_data_are_unseen_for_a_set_without_a_mask(tmp_path):
    from nbn.structure import HypothesisSet, export_factored_sets, load_factored_sets
    from tests.unit.structure.test_agreement import N, VARIABLES, ring_data
    data = ring_data(n_episodes=3)
    cands = {f"m{i}'": [(f"m{i}", f"m{(i - 1) % N}", f"rep{i}")] for i in range(N)}
    hs = HypothesisSet(VARIABLES, cands)
    fs = hs.factor_sets(data, delta=0.1, sampling="trajectory")
    for f, s in list(fs.sets.items()):                      # a bootstrap-like set: no mask
        lo, _ = s.bounds()
        fs.sets[f] = SampleSet(level=s.level, guarantee="asymptotic",
                               unseen=torch.zeros(0, dtype=torch.bool),
                               samples=torch.full((4, *lo.shape), 1.0 / lo.shape[-1]))
    out = export_factored_sets(tmp_path / "b", fs, hs, [f"m{i}" for i in range(N)],
                               [f"rep{i}" for i in range(N)])
    ex = load_factored_sets(out)
    f = ex.manifest["factors"][0]
    unseen = ex.array(f["id"], "unseen")
    assert unseen.any() and (unseen == (ex.array(f["id"], "counts") == 0)).all()
    assert f["graph_preserving"] is False


def test_bootstrap_set_keeps_a_given_unseen_mask():
    from nbn.uncertainty.builders import bootstrap_set
    m, names = _dense_bn(0, n=3, k=2, window=1, p=1.0)
    data = m.sample(50)
    mask = torch.tensor([True, False])
    s = bootstrap_set(m, data, lambda r: torch.softmax(r.mechanisms[names[1]]._logits, -1),
                      torch.arange(50), n_boot=3, unseen=mask)
    assert torch.equal(s.unseen, mask)


# ---------------------------------------------------------------- credal

def test_credal_query_treats_unseen_rows_as_the_simplex():
    from nbn.uncertainty.credal import credal_query
    m, names = _dense_bn(0, n=2, k=2, window=1, p=1.0)
    a, b = names
    pa = torch.softmax(m.mechanisms[a]._logits.detach().double(), -1)
    pb = torch.softmax(m.mechanisms[b]._logits.detach().double(), -1)
    sets = {a: IntervalSet(level=1.0, guarantee="pac", unseen=torch.zeros(1, dtype=torch.bool),
                           lower=pa, upper=pa),
            b: IntervalSet(level=1.0, guarantee="pac", unseen=torch.tensor([False, True]),
                           lower=pb, upper=pb)}          # row 1 claims a point, but has no data
    q = credal_query(m, sets, b)
    p_a1 = float(pa[0, 1])
    assert float(q.upper[1] - q.lower[1]) >= p_a1 - 1e-9   # row 1 free: width P(a=1)


# ---------------------------------------------------------------- calibration

def test_check_held_out():
    from nbn.scm import check_held_out
    check_held_out(None, None)                                   # off by default
    check_held_out(torch.arange(10), torch.arange(10, 20))
    with pytest.raises(ValueError, match="also used to fit"):
        check_held_out(torch.arange(10), torch.arange(5, 15))
    with pytest.raises(ValueError, match="ids"):
        check_held_out(torch.arange(10))
    with pytest.raises(ValueError, match="entries"):
        check_held_out(torch.arange(10), torch.arange(10, 13), n_rows=4)


def _additive_model(n=600, seed=0):
    from nbn.scm import AdditiveNoiseMechanism, StructuralCausalModel
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, generator=gen)
    data = {"x": x, "y": 2 * x + 0.5 * torch.randn(n, 1, generator=gen)}
    model = StructuralCausalModel([("x", "y")], dict.fromkeys(data, ("continuous", 1)), device="cpu")
    model.set_mechanism("y", AdditiveNoiseMechanism("linear"))
    model.auto_mechanisms(default_continuous="lg")
    model.fit({k: v[: n // 2] for k, v in data.items()})
    return model, {k: v[n // 2:] for k, v in data.items()}


def test_cdf_band_is_bound_to_its_mechanism_and_checks_held_out_rows():
    from nbn.scm.sets import cdf_band_set
    model, held = _additive_model()
    n = held["y"].shape[0]
    with pytest.raises(ValueError, match="also used to fit"):
        cdf_band_set(model, held, "y", fit_ids=torch.arange(300), ids=torch.arange(n))
    band = cdf_band_set(model, held, "y", fit_ids=torch.arange(300), ids=torch.arange(300, 300 + n))
    mech = model.mechanisms["y"]
    band.check(mech)
    lo, up = band.noise_cdf_bounds(torch.tensor([0.0]), mechanism=mech)
    assert float(lo) <= float(up)
    other = copy.deepcopy(mech)
    with torch.no_grad():
        for p in other.parameters():
            p.add_(0.1)
    with pytest.raises(ValueError, match="another mechanism"):
        band.noise_cdf_bounds(torch.tensor([0.0]), mechanism=other)


# ---------------------------------------------------------------- LP

def test_safe_lp_bound_is_below_every_optimum_even_with_a_bad_dual():
    scipy_opt = pytest.importorskip("scipy.optimize")
    from nbn.uncertainty._safe_lp import safe_lower_bound, safe_min
    rng = np.random.default_rng(1)
    for _ in range(50):
        n, mrows = int(rng.integers(3, 20)), int(rng.integers(2, 12))
        A = rng.normal(size=(mrows, n))
        b = A @ rng.dirichlet(np.ones(n)) + rng.uniform(0, 0.1, mrows)
        c = rng.normal(size=n)
        res = scipy_opt.linprog(c, A_ub=A, b_ub=b, A_eq=np.ones((1, n)), b_eq=[1.0],
                                bounds=(0, None), method="highs")
        s = safe_min(res, c, A, b, np.ones((1, n)), [1.0], 0.0, 1.0)
        assert s <= res.fun and res.fun - s < 1e-9
        rough = safe_lower_bound(c, A, b, np.ones((1, n)), [1.0], rng.normal(size=mrows),
                                 rng.normal(size=1), 0.0, 1.0)
        assert rough <= res.fun


def test_iv_program_ends_are_unchanged_when_the_slack_suffices():
    pytest.importorskip("scipy")
    from nbn.uncertainty.confounded import SOLVER_SLACK, IVProgram, RegimeBox
    p = np.array([[[0.2, 0.1], [0.3, 0.4]]])
    prog = IVProgram(RegimeBox.point(p))
    lo, hi = prog.bounds(prog.indicator(0, 1))
    # Manski for a point law: P(y=1 | do(a=0)) in [0.1, 0.1 + P(a=1)]
    # (the solver's optimum moves by the RHS_SLACK relaxation, 1e-7; the
    # dual bound is within ~1e-12 of it, so the fixed slack decides the ends)
    assert lo == pytest.approx(0.1 - prog.slack, abs=3e-7)
    assert hi == pytest.approx(0.8 + prog.slack, abs=3e-7)
    assert prog.slack == pytest.approx(2 * SOLVER_SLACK)


# ---------------------------------------------------------------- portfolio

def test_portfolio_reports_the_rows_it_used():
    from nbn.learning import select_mechanisms
    from nbn.mechanisms import CategoricalTableMechanism
    m, names = _dense_bn(0, n=3, k=2, window=1, p=1.0)
    data = m.sample(200)
    rep = select_mechanisms(m, data, {"table": CategoricalTableMechanism}, nodes=[names[1]])
    tr, va = rep.train_index, rep.val_index
    assert tr.numel() + va.numel() == 200 and not bool(torch.isin(tr, va).any())


# ---------------------------------------------------------------- review follow-ups

def test_evidence_child_of_a_knn_parent_is_certified_not_crashed():
    from nbn import NeuralBayesianNetwork
    from nbn.mechanisms import CategoricalTableMechanism, KNNConditionalMechanism
    torch.manual_seed(0)
    k, n = 3, 400
    a = torch.randint(0, k, (n, 1)).float()
    b = ((a + torch.randint(0, 2, (n, 1))) % k).float()
    c = ((b + torch.randint(0, 2, (n, 1))) % k).float()
    m = NeuralBayesianNetwork([("A", "B"), ("B", "C")], variables=dict.fromkeys("ABC", ("discrete", k)),
                              device="cpu")
    m.set_mechanism("A", CategoricalTableMechanism())
    m.set_mechanism("C", CategoricalTableMechanism())
    m.set_mechanism("B", KNNConditionalMechanism(k=15, discrete_child=True))
    m.fit({"A": a, "B": b, "C": c})
    assert CertifiedEngine.applicable(m, ["A"], {"C": 1})
    r = HybridRouter(fallback="certified", n_samples=2000)
    r._select = lambda *a_, **k_: r._lw
    r.query(m, ["A"], {"C": 1})
    assert r.last_certificate is not None


def test_joint_row_bounds_refuses_a_row_off_the_simplex(tmp_path):
    from nbn.structure import joint_row_bounds
    box = IntervalSet(level=0.95, guarantee="pac", unseen=torch.zeros(2, dtype=torch.bool),
                      lower=torch.tensor([[0.6, 0.6], [0.0, 0.0]], dtype=torch.float64),
                      upper=torch.tensor([[0.7, 0.7], [1.0, 1.0]], dtype=torch.float64))
    ex = _two_node_export(tmp_path, box)
    with pytest.raises(ValueError, match="simplex"):
        joint_row_bounds(ex, {"s'": ex.manifest["factors"][0]["id"]}, 0)


def test_extra_event_keeps_its_declared_guarantee(tmp_path):
    box = IntervalSet(level=0.95, guarantee="pac", unseen=torch.zeros(2, dtype=torch.bool),
                      lower=torch.full((2, 2), 0.2, dtype=torch.float64),
                      upper=torch.full((2, 2), 0.8, dtype=torch.float64))
    ex = _two_node_export(tmp_path, box, extra_events={"test": (0.01, "asymptotic")})
    assert ex.manifest["composition"]["guarantees"] == ["asymptotic", "pac"]


def test_credal_query_checks_the_unseen_mask():
    from nbn.uncertainty.credal import credal_query
    m, names = _dense_bn(0, n=2, k=2, window=1, p=1.0)
    a, b = names
    pa = torch.softmax(m.mechanisms[a]._logits.detach().double(), -1)
    pb = torch.softmax(m.mechanisms[b]._logits.detach().double(), -1)
    sets = {a: IntervalSet(level=1.0, guarantee="pac", unseen=torch.zeros(1, dtype=torch.bool),
                           lower=pa, upper=pa),
            b: IntervalSet(level=1.0, guarantee="pac", unseen=torch.zeros(3, dtype=torch.bool),
                           lower=pb, upper=pb)}
    with pytest.raises(ValueError, match="unseen mask"):
        credal_query(m, sets, b)
    sets[b] = SampleSet(level=0.9, guarantee="asymptotic", unseen=torch.zeros(0, dtype=torch.bool),
                        samples=pb.unsqueeze(0).repeat(3, 1, 1))
    with pytest.warns(UserWarning, match="no unseen mask"):
        credal_query(m, sets, b)


def test_safe_lp_bound_survives_underflow():
    from fractions import Fraction
    from nbn.uncertainty._safe_lp import safe_lower_bound
    t = 2.0 ** -1074
    s = safe_lower_bound(np.array([-t]), np.array([[2 * t]]), np.array([t]), None, None,
                         np.array([-0.5]), None, 0.0, 1.0)
    assert Fraction(s) <= -Fraction(t) / 2         # true optimum -t/2


# ---------------------------------------------------------------- device smoke follow-ups

def test_portfolio_gives_the_class_count_to_neural_candidates():
    from nbn.learning import select_mechanisms
    from nbn.mechanisms import NeuralCategoricalMechanism
    m, names = _dense_bn(0, n=3, k=3, window=1, p=1.0)
    data = m.sample(600)
    rep = select_mechanisms(m, data, {"neural": NeuralCategoricalMechanism}, nodes=[names[1]],
                            max_epochs=5, min_epochs=2)
    assert m.mechanisms[names[1]].n_classes == 3 and rep.chosen[names[1]] == "neural"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_query_many_runs_sequentially_on_mps():
    from nbn.inference import query_many
    from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
    m, names = _dense_bn(0, n=8, k=3, window=2, p=0.8)
    m = copy.deepcopy(m)
    m.to("mps")
    out = query_many(m, [([names[i]], {names[-1]: 1}) for i in range(4)],
                     engine=LikelihoodWeightingEngine(n_samples=2000), workers=4)
    assert len(out) == 4


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_credal_query_accepts_sets_on_mps():
    from nbn.uncertainty.credal import credal_query
    m, names = _dense_bn(0, n=3, k=2, window=1, p=1.0)
    sets = {}
    for n in names:
        p = torch.softmax(m.mechanisms[n]._logits.detach(), -1).float()
        sets[n] = IntervalSet(level=1.0, guarantee="pac", unseen=torch.zeros(p.shape[0], dtype=torch.bool,
                              device="mps"), lower=(p - 0.01).clamp(0, 1).to("mps"),
                              upper=(p + 0.01).clamp(0, 1).to("mps"))
    q = credal_query(m, sets, names[-1])
    assert (q.lower <= q.upper).all()
