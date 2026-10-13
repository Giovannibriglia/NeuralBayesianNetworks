"""Export of factored probability sets with sharing (#314)."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pytest
import torch

from nbn import NeuralBayesianNetwork
from nbn.mechanisms import CategoricalTableMechanism
from nbn.structure import (
    HypothesisSet,
    agreement_set,
    export_factored_sets,
    joint_row_bounds,
    load_factored_sets,
)
from nbn.structure.export import SCHEMA_FILE
from nbn.uncertainty import IntervalSet, interval_set, parent_row_index
from tests.conftest import available_devices
from tests.unit.structure.test_agreement import INPUTS, N, VARIABLES, p_up, ring_data

STATE = [f"m{i}" for i in range(N)]


def _agreement_sets(data, delta=0.1):
    return {f"m{i}'": agreement_set(VARIABLES, data, f"m{i}'", INPUTS, max_parents=3,
                                    delta=delta / N, sampling="trajectory") for i in range(N)}


def _export_ring(tmp_path, device="cpu", **kw):
    data = ring_data(n_episodes=200, device=device)
    sets = _agreement_sets(data)
    out = export_factored_sets(tmp_path / "ring", sets, VARIABLES, STATE, [f"rep{i}" for i in range(N)], **kw)
    return out, sets


@pytest.mark.parametrize("device", available_devices())
def test_round_trip_is_exact(tmp_path, device):
    out, sets = _export_ring(tmp_path, device)
    ex = load_factored_sets(out)
    assert ex.manifest["schema"] == "nbn.factored_sets/1"
    assert json.loads((out / "manifest.json").read_text()) == ex.manifest
    for f in ex.manifest["factors"]:
        res = sets[f["node"]]
        lo, hi = res.set.bounds()
        np.testing.assert_array_equal(ex.array(f["id"], "lower"), lo.detach().cpu().double().numpy())
        np.testing.assert_array_equal(ex.array(f["id"], "upper"), hi.detach().cpu().double().numpy())
        np.testing.assert_array_equal(ex.array(f["id"], "radius"), res.set.radius.detach().cpu().double().numpy())
        np.testing.assert_array_equal(ex.array(f["id"], "unseen"), res.set.unseen.cpu().numpy())
        assert f["parents"] == list(res.parents) and f["set_type"] == "l1"
        assert f["guarantee"] == "pac_under_model" and f["sampling"] == "trajectory"
        assert f["agreement"]["max_parents"] == 3 and len(f["assumptions"]) == 4
        assert ex.array(f["id"], "lower").dtype == np.float64


def test_manifest_matches_the_json_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    out, _ = _export_ring(tmp_path, exact=True)
    schema = json.loads(SCHEMA_FILE.read_text())
    jsonschema.validate(json.loads((out / "manifest.json").read_text()), schema)


def test_joint_rows_rebuilt_from_the_export_contain_the_true_kernel(tmp_path):
    out, sets = _export_ring(tmp_path)
    ex = load_factored_sets(out)
    (h,) = ex.hypotheses()
    plan_vars = STATE + [f"rep{i}" for i in range(N)]
    grid = list(itertools.product(range(2), repeat=len(plan_vars)))
    checked = 0
    for row, values in enumerate(grid):
        col = dict(zip(plan_vars, values))
        if sum(col[f"rep{i}"] for i in range(N)) > 1:
            continue                                   # at most one repair per step
        if any(bool(ex.array(h[f"m{i}'"], "unseen")[ex.array(h[f"m{i}'"], "row_index")[row]])
               for i in range(N)):
            continue
        lower, upper, nodes = joint_row_bounds(ex, h, row)
        marg = [p_up(torch.tensor(col[f"m{i}"]), torch.tensor(col[f"m{(i - 1) % N}"]),
                     torch.tensor(col[f"rep{i}"])).item() for i in range(N)]
        truth = np.ones(())
        for p in marg:
            truth = np.multiply.outer(truth, np.array([1 - p, p]))
        assert nodes == [f"m{i}'" for i in range(N)]
        assert np.all(truth >= lower - 1e-9) and np.all(truth <= upper + 1e-9)
        checked += 1
    assert checked > 50


def test_row_index_matches_parent_row_index_and_is_omitted_when_too_large(tmp_path):
    out, sets = _export_ring(tmp_path)
    ex = load_factored_sets(out)
    plan_vars = STATE + [f"rep{i}" for i in range(N)]
    grid = torch.tensor(list(itertools.product(range(2), repeat=len(plan_vars))))
    f = ex.manifest["factors"][0]
    cols = [plan_vars.index(p) for p in f["parents"]]
    np.testing.assert_array_equal(ex.array(f["id"], "row_index"),
                                  parent_row_index(grid[:, cols], f["parent_cards"]).numpy())
    assert ex.manifest["planning_space"]["n_rows"] == 2 ** len(plan_vars)
    small, _ = _export_ring(tmp_path / "small", max_rows=10)
    ex2 = load_factored_sets(small)
    assert ex2.manifest["planning_space"]["n_rows"] is None
    assert not any(k.startswith("row_index/") for k in ex2.arrays)
    (h,) = ex.hypotheses()
    (h2,) = ex2.hypotheses()
    for row in (0, 17, 200, 255):                       # the layout gives the same rows and bounds
        for fid in h.values():
            assert ex2.factor_row(fid, row) == ex.factor_row(fid, row)
        a, b = joint_row_bounds(ex, h, row), joint_row_bounds(ex2, h2, row)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


def test_exact_bounds_are_rounded_outwards(tmp_path):
    from fractions import Fraction

    out, _ = _export_ring(tmp_path, exact=True, denominator=1000)
    ex = load_factored_sets(out)
    assert ex.manifest["exact"] == {"denominator": 1000}
    for f in ex.manifest["factors"]:
        lo, hi = ex.array(f["id"], "lower"), ex.array(f["id"], "upper")
        num_lo, num_hi = ex.array(f["id"], "lower_num"), ex.array(f["id"], "upper_num")
        assert num_lo.dtype == np.int64
        for x, n in zip(lo.ravel(), num_lo.ravel()):
            assert Fraction(int(n), 1000) <= Fraction(float(x))          # exact, no tolerance
        for x, n in zip(hi.ravel(), num_hi.ravel()):
            assert Fraction(int(n), 1000) >= Fraction(float(x))


def test_outward_rounding_on_values_that_float_multiplication_rounds_inward():
    from fractions import Fraction

    from nbn.structure.export import _outward

    lo = np.array([0.3, 0.7, 0.1, 0.0, 0.5])
    hi = np.array([0.3, 0.7, 0.1, 1.0, 0.5])
    num_lo, num_hi = _outward(lo, hi, 10)
    for x, a, b in zip(lo, num_lo, num_hi):
        assert Fraction(int(a), 10) <= Fraction(float(x)) <= Fraction(int(b), 10)
    assert num_lo[3] == 0 and num_hi[3] == 10          # exact end points stay exact


def test_hypothesis_set_export_lists_candidates_and_the_composition(tmp_path):
    data = ring_data(n_episodes=100)
    cands = {f"m{i}'": [(f"m{i}", f"m{(i - 1) % N}", f"rep{i}"), (f"m{i}", f"rep{i}")] for i in range(N)}
    hs = HypothesisSet(VARIABLES, cands)
    fs = hs.factor_sets(data, delta=0.1, sampling="trajectory")
    out = export_factored_sets(tmp_path / "hs", fs, hs, STATE, [f"rep{i}" for i in range(N)])
    ex = load_factored_sets(out)
    assert len(ex.manifest["factors"]) == hs.n_factors == 8
    assert {n: len(v) for n, v in ex.manifest["hypotheses"]["candidates"].items()} == dict.fromkeys(cands, 2)
    assert ex.n_hypotheses == hs.n_hypotheses == 16
    assert sum(1 for _ in ex.hypotheses()) == 16                  # lazy, but complete
    keys = {f["row_index"] for f in ex.manifest["factors"]}
    assert len(keys) == len({tuple(f["parents"]) for f in ex.manifest["factors"]})   # one map per parent tuple
    assert ex.manifest["composition"]["level"] == pytest.approx(0.9)
    assert ex.manifest["composition"]["guarantees"] == ["pac"]
    assert all(f["sampling"] == "trajectory" for f in ex.manifest["factors"])
    counts = ex.array("f0", "counts")
    assert counts.sum() == data["m0"].numel()


def _health_model():
    from nbn.scm import DiscreteAdditiveNoiseMechanism, StructuralCausalModel

    gen = torch.Generator().manual_seed(0)
    n = 2000
    h = torch.randint(0, 4, (n,), generator=gen)
    e = torch.randint(0, 2, (n,), generator=gen)
    u = torch.multinomial(torch.tensor([0.15, 0.7, 0.15]), n, replacement=True, generator=gen) - 1
    data = {"h": h.float(), "e": e.float(), "h'": (h + e + u).clamp(0, 3).float(),
            "g'": (h + u).clamp(0, 3).float()}
    model = StructuralCausalModel([("h", "h'"), ("e", "h'"), ("h", "g'")],
                                  {"h": ("discrete", 4), "e": ("discrete", 2), "h'": ("discrete", 4),
                                   "g'": ("discrete", 4)}, device="cpu")
    model.set_mechanism("h'", DiscreteAdditiveNoiseMechanism(max_offset=1))
    model.set_mechanism("g'", DiscreteAdditiveNoiseMechanism(max_offset=1))
    model.tie_noise(["h'", "g'"])                       # one shared noise law U
    model.auto_mechanisms()
    model.fit(data)
    return model, data


def test_noise_polytope_is_one_shared_parameter_block(tmp_path):
    from nbn.scm import noise_parameter_set

    model, data = _health_model()
    poly = noise_parameter_set(model, data, ["h'", "g'"], delta=0.1)
    out = export_factored_sets(tmp_path / "noise", {"h'": poly, "g'": poly}, model, ["h"], ["e"])
    ex = load_factored_sets(out)
    assert len(ex.manifest["parameters"]) == 1
    pid = ex.manifest["parameters"][0]["id"]
    assert all(f["set_type"] == "noise_polytope" and f["parameter"] == pid for f in ex.manifest["factors"])
    assert ex.manifest["composition"]["events"] == [pid]            # one event, counted once
    np.testing.assert_array_equal(ex.array(pid, "expressions"), poly.expressions.cpu().numpy().astype(np.int8))
    np.testing.assert_array_equal(ex.array("f0", "index"), poly.index["h'"].cpu().numpy())


@pytest.mark.parametrize("unseen_row, expected", [(False, True), (True, False)])
def test_graph_preserving_flag_follows_zero_pattern_and_unseen_rows(tmp_path, unseen_row, expected):
    v = {"x": ("discrete", 2), "a": ("discrete", 2), "x'": ("discrete", 2)}
    model = NeuralBayesianNetwork([("x", "x'"), ("a", "x'")], v, device="cpu")
    for node in model.dag.nodes():
        model.set_mechanism(node, CategoricalTableMechanism())
    lower = torch.full((4, 2), 0.2, dtype=torch.float64)
    upper = torch.full((4, 2), 0.8, dtype=torch.float64)
    unseen = torch.zeros(4, dtype=torch.bool)
    if unseen_row:
        lower[3], upper[3], unseen[3] = 0.0, 1.0, True
    box = IntervalSet(level=0.9, guarantee="pac", unseen=unseen, lower=lower, upper=upper)
    out = export_factored_sets(tmp_path / f"g{unseen_row}", {"x'": box}, model, ["x"], ["a"])
    assert load_factored_sets(out).manifest["factors"][0]["graph_preserving"] is expected


def test_single_model_interval_sets(tmp_path):
    model, data = _health_model()
    sets = {n: interval_set(model, data, n, delta=0.05) for n in ("h'", "g'")}
    out = export_factored_sets(tmp_path / "m", sets, model, ["h"], ["e"])
    ex = load_factored_sets(out)
    assert [f["parents"] for f in ex.manifest["factors"]] == [["h", "e"], ["h"]]
    assert ex.manifest["composition"]["level"] == pytest.approx(0.9)
    roles = {v["name"]: v["role"] for v in ex.manifest["variables"]}
    assert roles == {"h": "state", "e": "action", "h'": "next_state", "g'": "other"}


def test_errors(tmp_path):
    out, sets = _export_ring(tmp_path)
    with pytest.raises(FileExistsError):
        export_factored_sets(out, sets, VARIABLES, STATE, [f"rep{i}" for i in range(N)])
    export_factored_sets(out, sets, VARIABLES, STATE, [f"rep{i}" for i in range(N)], overwrite=True)
    with pytest.raises(ValueError, match="not state or action"):
        export_factored_sets(tmp_path / "e1", sets, VARIABLES, STATE[:3], [f"rep{i}" for i in range(N)])
    with pytest.raises(ValueError, match="not state or action"):              # also without row maps
        export_factored_sets(tmp_path / "e1b", sets, VARIABLES, STATE[:3], [f"rep{i}" for i in range(N)],
                             max_rows=2)
    (out / "notes.txt").write_text("mine")
    with pytest.raises(FileExistsError, match="other files"):
        export_factored_sets(out, sets, VARIABLES, STATE, [f"rep{i}" for i in range(N)], overwrite=True)
    target = tmp_path / "kernel.npz"
    target.write_text("x")
    with pytest.raises(FileExistsError, match="not a directory"):
        export_factored_sets(target, sets, VARIABLES, STATE, [f"rep{i}" for i in range(N)])
    with pytest.raises(TypeError):
        export_factored_sets(tmp_path / "e2", {"m0'": object()}, VARIABLES, STATE, ["rep0"])
    with pytest.raises(ValueError, match="model's DAG"):
        box = IntervalSet(level=0.9, guarantee="pac", unseen=torch.zeros(2, dtype=torch.bool),
                          lower=torch.zeros(2, 2), upper=torch.ones(2, 2))
        export_factored_sets(tmp_path / "e3", {"m0'": box}, VARIABLES, STATE, ["rep0"])
    with pytest.raises(TypeError):
        export_factored_sets(tmp_path / "e4", sets, VARIABLES, "m0", ["rep0"])
    with pytest.raises(ValueError, match="schema"):
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "manifest.json").write_text(json.dumps({"schema": "other/9"}))
        load_factored_sets(bad)



def test_noise_polytope_exact_bounds(tmp_path):
    from fractions import Fraction

    from nbn.scm import noise_parameter_set

    model, data = _health_model()
    poly = noise_parameter_set(model, data, ["h'", "g'"], delta=0.1)
    out = export_factored_sets(tmp_path / "nq", {"h'": poly, "g'": poly}, model, ["h"], ["e"],
                               exact=True, denominator=10**6)
    ex = load_factored_sets(out)
    pid = ex.manifest["parameters"][0]["id"]
    for x, n in zip(ex.array(pid, "lower"), ex.array(pid, "lower_num")):
        assert Fraction(int(n), 10**6) <= Fraction(float(x))
    for x, n in zip(ex.array(pid, "upper"), ex.array(pid, "upper_num")):
        assert Fraction(int(n), 10**6) >= Fraction(float(x))


def test_sample_set_bounds_are_computed_once(tmp_path):
    import warnings

    data = ring_data(n_episodes=50)
    hs = HypothesisSet(VARIABLES, {"m0'": [("m0", "m3", "rep0")]})
    fs = hs.factor_sets(data, builder="dirichlet", delta=0.5, n_samples=20)   # too few draws: warns
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export_factored_sets(tmp_path / "d", fs, hs, STATE, [f"rep{i}" for i in range(N)])
    assert sum("does not support the declared level" in str(w.message) for w in caught) == 1


# --- review round 2 regressions (#321) ------------------------------------------------------
def test_a_failed_overwrite_keeps_the_previous_export(tmp_path):
    out, sets = _export_ring(tmp_path)
    before = sorted(p.name for p in out.iterdir())
    with pytest.raises(ValueError, match="not state or action"):
        export_factored_sets(out, sets, VARIABLES, STATE[:3], [f"rep{i}" for i in range(N)], overwrite=True)
    assert sorted(p.name for p in out.iterdir()) == before
    assert load_factored_sets(out).manifest["schema"] == "nbn.factored_sets/1"
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]      # no temp dirs left


def test_wrong_row_count_is_rejected(tmp_path):
    v = {"x": ("discrete", 2), "a": ("discrete", 2), "x'": ("discrete", 2)}
    model = NeuralBayesianNetwork([("x", "x'"), ("a", "x'")], v, device="cpu")
    box = IntervalSet(level=0.9, guarantee="pac", unseen=torch.zeros(2, dtype=torch.bool),
                      lower=torch.zeros(2, 2), upper=torch.ones(2, 2))          # 2 rows, parents give 4
    with pytest.raises(ValueError, match="give"):
        export_factored_sets(tmp_path / "w", {"x'": box}, model, ["x"], ["a"])


def test_sample_sets_without_unseen_info_and_unseen_rows_in_joint_bounds(tmp_path):
    from nbn.uncertainty import SampleSet

    v = {"x": ("discrete", 2), "a": ("discrete", 2), "x'": ("discrete", 2)}
    model = NeuralBayesianNetwork([("x", "x'"), ("a", "x'")], v, device="cpu")
    draws = torch.distributions.Dirichlet(torch.ones(4, 2) * 5).sample((400,))
    boot = SampleSet(level=0.9, guarantee="asymptotic", unseen=torch.zeros(0, dtype=torch.bool),
                     samples=draws, simultaneous=False)
    ex = load_factored_sets(export_factored_sets(tmp_path / "b", {"x'": boot}, model, ["x"], ["a"]))
    assert ex.array("f0", "unseen").shape == (4,) and ex.array("f0", "samples").dtype == np.float32
    # A Dirichlet factor with an unseen row: the joint bounds give [0, 1] there.
    hs = HypothesisSet(v, {"x'": [("x", "a")]})
    data = {"x": torch.tensor([0, 0, 1]), "a": torch.tensor([0, 1, 0]), "x'": torch.tensor([1, 0, 1])}
    fs = hs.factor_sets(data, builder="dirichlet", delta=0.1, n_samples=4000)
    ex = load_factored_sets(export_factored_sets(tmp_path / "d", fs, hs, ["x"], ["a"]))
    (h,) = ex.hypotheses()
    unseen_row = int(np.flatnonzero(ex.array("f0", "unseen"))[0])          # (x=1, a=1)
    lo, hi, _ = joint_row_bounds(ex, h, unseen_row)
    assert np.all(lo == 0) and np.all(hi == 1)


def test_parameter_block_lists_only_exported_nodes_and_row_maps_are_compact(tmp_path):
    from nbn.scm import noise_parameter_set

    model, data = _health_model()
    poly = noise_parameter_set(model, data, ["h'", "g'"], delta=0.1)
    ex = load_factored_sets(export_factored_sets(tmp_path / "p", {"h'": poly}, model, ["h"], ["e"]))
    assert ex.manifest["parameters"][0]["nodes"] == ["h'"]
    assert ex.array("f0", "row_index").dtype == np.int32
