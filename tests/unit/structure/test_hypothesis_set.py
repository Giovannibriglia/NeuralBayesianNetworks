"""HypothesisSet and the factor cache (#312).

Every factor quantity must equal what one model per hypothesis gives: counts
(exactly), probability sets (built at the per-factor level), and the fitted
tables and query answers of ``model_for``.
"""
from __future__ import annotations

import itertools
import math
import random

import pytest
import torch

from nbn import NeuralBayesianNetwork
from nbn.mechanisms import CategoricalTableMechanism
from nbn.structure import FactorCounts, HypothesisSet
from nbn.uncertainty import count_table, dirichlet_set, interval_set, l1_ball_set
from tests.conftest import available_devices


def _random_two_slice(seed: int, n_state: int = 3):
    """A random 2-slice spec: state s0..s{d-1}, action a, next states s0'..; 1–3 candidates each."""
    rng = random.Random(seed)
    cards = {f"s{i}": rng.choice([2, 3]) for i in range(n_state)}
    cards["a"] = rng.choice([2, 3])
    inputs = list(cards)
    candidates = {}
    for i in range(n_state):
        nxt = f"s{i}'"
        cards[nxt] = cards[f"s{i}"]
        pool = [c for r in range(0, 4) for c in itertools.combinations(inputs, r)]
        chosen = rng.sample(pool, rng.choice([1, 2, 3]))
        candidates[nxt] = [tuple(rng.sample(c, len(c))) for c in chosen]  # random parent order
    variables = {v: ("discrete", c) for v, c in cards.items()}
    return variables, candidates, cards


def _data(cards, n, device, seed):
    gen = torch.Generator().manual_seed(seed)
    data = {v: torch.randint(0, c, (n,), generator=gen) for v, c in cards.items()}
    # Make some parent values unseen: s0 never takes its last value when a == 0.
    data["s0"][data["a"] == 0] = data["s0"][data["a"] == 0].clamp_max(cards["s0"] - 2)
    return {v: t.to(device) for v, t in data.items()}


def _model(hs: HypothesisSet, hypothesis, device: str) -> NeuralBayesianNetwork:
    model = NeuralBayesianNetwork(hs.edges(hypothesis), dict(hs.variables), device=device)
    for node in model.dag.nodes():
        model.set_mechanism(node, CategoricalTableMechanism())
    return model


# ---------------------------------------------------------------------------
# Sizes, enumeration, validation
# ---------------------------------------------------------------------------


def test_sizes_and_lazy_hypotheses():
    variables, candidates, _ = _random_two_slice(0)
    hs = HypothesisSet(variables, candidates)
    assert hs.n_factors == sum(len(c) for c in candidates.values())
    assert hs.n_hypotheses == math.prod(len(c) for c in candidates.values())
    hyps = list(hs.hypotheses())
    assert len(hyps) == hs.n_hypotheses
    assert len({tuple(sorted(h.items())) for h in hyps}) == hs.n_hypotheses


def test_fixed_edges_join_every_candidate():
    v = {"x": ("discrete", 2), "a": ("discrete", 2), "x'": ("discrete", 2), "a'": ("discrete", 2)}
    hs = HypothesisSet(v, {"x'": [("x",), ()]}, fixed_edges=[("a", "x'"), ("a", "a'")])
    assert hs.candidates["x'"] == [("x", "a"), ("a",)]
    assert hs.candidates["a'"] == [("a",)]


@pytest.mark.parametrize("bad, err", [
    ({"y": [("x",)]}, KeyError),                       # unknown node
    ({"x'": []}, ValueError),                          # no candidate
    ({"x'": [("x", "x")]}, ValueError),                # repeated parent
    ({"x'": [("x'",)]}, ValueError),                   # self parent
    ({"x'": [("z",)]}, KeyError),                      # unknown parent
    ({"x'": [("x", "a"), ("a", "x")]}, ValueError),    # same family twice
    ({"x'": [("x",)], "x": [("x'",)]}, ValueError),    # cycle in the union
])
def test_validation(bad, err):
    v = {"x": ("discrete", 2), "a": ("discrete", 2), "x'": ("discrete", 2)}
    with pytest.raises(err):
        HypothesisSet(v, bad)


def test_continuous_variables_point_to_the_continuous_issue():
    v = {"x": ("continuous", 1), "x'": ("discrete", 2)}
    hs = HypothesisSet(v, {"x'": [("x",)]})
    with pytest.raises(ValueError, match="#318"):
        hs.counts({"x": torch.randn(5), "x'": torch.zeros(5)})


def test_hypothesis_must_pick_a_listed_candidate():
    variables, candidates, cards = _random_two_slice(1)
    hs = HypothesisSet(variables, candidates)
    fc = hs.counts(_data(cards, 50, "cpu", 0))
    h = next(hs.hypotheses())
    with pytest.raises(ValueError):
        fc.for_hypothesis(dict(list(h.items())[:-1]))
    bad = dict(h)
    bad[hs.targets[0]] = ("not", "a", "candidate")
    with pytest.raises(ValueError):
        hs.model_for(bad, fc)


# ---------------------------------------------------------------------------
# Factor cache: counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("seed", range(4))
def test_counts_equal_count_table_of_each_hypothesis_model(device, seed):
    variables, candidates, cards = _random_two_slice(seed)
    hs = HypothesisSet(variables, candidates)
    data = _data(cards, 400, device, seed)
    fc = hs.counts(data)
    assert isinstance(fc, FactorCounts)
    for h in hs.hypotheses():
        model = _model(hs, h, device)
        for node, table in fc.for_hypothesis(h).items():
            expected, _ = count_table(model, data, node)
            assert table.device.type == torch.device(device).type
            assert tuple(model.dag.parents(node)) == tuple(h[node])  # DAG keeps the edge order
            assert torch.equal(table, expected)


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("weighted", [False, True])
def test_union_path_equals_direct_counting(device, seed, weighted):
    variables, candidates, cards = _random_two_slice(seed)
    hs = HypothesisSet(variables, candidates)
    data = _data(cards, 400, device, seed)
    w = (torch.rand(400, generator=torch.Generator().manual_seed(seed)) * 3).to(device) if weighted else None
    union = hs.counts(data, weights=w)
    direct = hs.counts(data, weights=w, max_union_cells=0)
    assert not any(direct.union_used.values())
    assert any(union.union_used.values()) or all(len(c) == 1 for c in candidates.values())
    for f in hs.factors:
        if weighted:
            torch.testing.assert_close(union[f], direct[f], rtol=1e-5, atol=1e-4)
        else:
            assert torch.equal(union[f], direct[f])


# ---------------------------------------------------------------------------
# Probability sets per factor
# ---------------------------------------------------------------------------


def _same_set(a, b):
    assert type(a) is type(b) and a.guarantee == b.guarantee
    for f in ("lower", "upper", "center", "radius", "unseen", "samples"):
        if hasattr(a, f):
            torch.testing.assert_close(getattr(a, f), getattr(b, f), rtol=0, atol=1e-12)


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("builder", ["interval", "l1", "dirichlet"])
@pytest.mark.parametrize("sampling", ["iid", "trajectory"])
def test_factor_sets_equal_one_model_per_hypothesis(device, builder, sampling):
    variables, candidates, cards = _random_two_slice(2)
    hs = HypothesisSet(variables, candidates)
    data = _data(cards, 400, device, 2)
    delta = 0.1
    torch.manual_seed(7)
    fs = hs.factor_sets(data, builder=builder, delta=delta, sampling=sampling, n_samples=200)
    assert fs.n_factors == hs.n_factors
    assert fs.level == pytest.approx(1 - delta)
    assert fs.delta_per_factor == pytest.approx(delta / hs.n_factors)
    torch.manual_seed(7)
    for h in hs.hypotheses():
        model = _model(hs, h, device)
        got = fs.sets_for(h)
        for node in hs.targets:
            assert tuple(model.dag.parents(node)) == tuple(h[node])
            if builder == "interval":
                ref = interval_set(model, data, node, delta=fs.delta_per_factor, sampling=sampling)
            elif builder == "l1":
                ref = l1_ball_set(model, data, node, delta=fs.delta_per_factor, sampling=sampling)
            else:
                assert got[node].level == pytest.approx(1 - fs.delta_per_factor)
                continue  # random draws: equality of the posterior is checked via the counts
            _same_set(got[node], ref)
        break  # one hypothesis covers every node; the cache makes the rest identical views


def test_sets_for_returns_views_shared_across_hypotheses():
    variables, candidates, cards = _random_two_slice(3)
    hs = HypothesisSet(variables, candidates)
    fs = hs.factor_sets(_data(cards, 200, "cpu", 3), delta=0.1)
    seen = {}
    for h in hs.hypotheses():
        for node, s in fs.sets_for(h).items():
            key = (node, h[node])
            assert seen.setdefault(key, s) is s


def test_factor_sets_reject_foreign_counts_and_bad_delta():
    variables, candidates, cards = _random_two_slice(0)
    a, b = HypothesisSet(variables, candidates), HypothesisSet(variables, candidates)
    data = _data(cards, 100, "cpu", 0)
    with pytest.raises(ValueError, match="another"):
        a.factor_sets(b.counts(data))
    with pytest.raises(ValueError):
        a.factor_sets(data, delta=0.0)
    with pytest.raises(ValueError, match="builder"):
        a.factor_sets(data, builder="nope")


# ---------------------------------------------------------------------------
# model_for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_model_for_equals_fitting_the_hypothesis(device, alpha):
    variables, candidates, cards = _random_two_slice(4)
    hs = HypothesisSet(variables, candidates)
    data = _data(cards, 400, device, 4)
    fc = hs.counts(data)
    for h in itertools.islice(hs.hypotheses(), 4):
        cached = hs.model_for(h, fc, alpha=alpha)
        used = {v: hs.variables[v] for v in cached.variables}       # the hypothesis' variables
        direct = NeuralBayesianNetwork(hs.edges(h), used, device=device)
        for node in direct.dag.nodes():
            direct.set_mechanism(node, CategoricalTableMechanism(alpha=alpha))
        direct.fit({v: data[v] for v in used})
        for node in direct.dag.nodes():
            torch.testing.assert_close(
                cached.mechanisms[node]._logits, direct.mechanisms[node]._logits, rtol=0, atol=1e-6,
            )
            assert torch.equal(cached.mechanisms[node]._counts, direct.mechanisms[node]._counts)
        node = hs.targets[0]
        ev = {p: torch.tensor([0], device=device) for p in h[node]}
        torch.testing.assert_close(cached.query([node], evidence=ev), direct.query([node], evidence=ev),
                                   rtol=0, atol=1e-6)


def test_model_for_tables_follow_the_dag_parent_order():
    v = {"x": ("discrete", 2), "a": ("discrete", 3), "x'": ("discrete", 2)}
    hs = HypothesisSet(v, {"x'": [("a", "x")]})
    gen = torch.Generator().manual_seed(0)
    data = {k: torch.randint(0, c[1], (300,), generator=gen) for k, c in v.items()}
    fc = hs.counts(data)
    h = {"x'": ("a", "x")}
    m = hs.model_for(h, fc)
    dag_parents = tuple(m.dag.parents("x'"))
    direct, _ = count_table(m, data, "x'")
    assert torch.equal(m.mechanisms["x'"]._counts, direct.float())
    assert set(dag_parents) == {"a", "x"}


# ---------------------------------------------------------------------------
# Coverage: the sets of every hypothesis hold jointly
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_superset_hypothesis_covers_the_true_tables_jointly():
    """True model: x' <- (x, a), y' <- (y,); candidates include supersets and a wrong subset.

    On the joint event (all factor sets hold, prob >= 1 - delta), the sets of
    every superset candidate contain the true conditional on every seen row.
    """
    torch.manual_seed(0)
    px = torch.tensor([[[0.8, 0.2], [0.3, 0.7]], [[0.1, 0.9], [0.6, 0.4]]])  # [x, a, x']
    py = torch.tensor([[0.7, 0.3], [0.2, 0.8]])                             # [y, y']
    v = dict.fromkeys(["x", "y", "a", "x'", "y'"], ("discrete", 2))
    hs = HypothesisSet(v, {"x'": [("x", "a"), ("x", "a", "y"), ("x",)], "y'": [("y",), ("y", "a")]})
    supersets = {("x'", ("x", "a")), ("x'", ("x", "a", "y")), ("y'", ("y",)), ("y'", ("y", "a"))}
    delta, reps, hits = 0.1, 100, 0
    for r in range(reps):
        gen = torch.Generator().manual_seed(100 + r)
        n = 300
        x, y, a = (torch.randint(0, 2, (n,), generator=gen) for _ in range(3))
        x1 = torch.bernoulli(px[x, a, 1], generator=gen).long()
        y1 = torch.bernoulli(py[y, 1], generator=gen).long()
        fs = hs.factor_sets({"x": x, "y": y, "a": a, "x'": x1, "y'": y1}, delta=delta)
        ok = True
        for node, parents in supersets:
            idx = torch.tensor(list(itertools.product(*[range(2)] * len(parents))))
            cols = dict(zip(parents, idx.T)) if parents else {}
            if node == "x'":
                truth = px[cols["x"], cols["a"]]
            else:
                truth = py[cols["y"]]
            s = fs[(node, parents)]
            ok &= bool(s.row_contains(truth.double())[~s.unseen].all())
        hits += int(ok)
    assert hits / reps >= 1 - delta - 3 * math.sqrt(delta * (1 - delta) / reps)


@pytest.mark.parametrize("device", available_devices())
def test_reorder_parents_matches_counting_in_the_other_order(device):
    from nbn.structure.hypotheses import _reorder_parents
    from nbn.uncertainty import count_table_for

    v = {"x": ("discrete", 2), "a": ("discrete", 3), "y": ("discrete", 4), "x'": ("discrete", 3)}
    hs = HypothesisSet(v, {"x'": [("x", "a", "y")]})
    gen = torch.Generator().manual_seed(0)
    data = {k: torch.randint(0, c[1], (500,), generator=gen).to(device) for k, c in v.items()}
    t, cards = count_table_for(hs.variables, data, "x'", ("x", "a", "y"))
    for target in itertools.permutations(("x", "a", "y")):
        expected, _ = count_table_for(hs.variables, data, "x'", target)
        assert torch.equal(_reorder_parents(t, cards, ("x", "a", "y"), target), expected)
    with pytest.raises(ValueError):
        _reorder_parents(t, cards, ("x", "a", "y"), ("x", "a"))



# ---------------------------------------------------------------------------
# Review regressions (#319)
# ---------------------------------------------------------------------------


def test_fit_from_counts_refuses_subclasses_with_their_own_fit_local():
    from nbn.mechanisms.parametric.binning_categorical import BinningCategoricalTable
    from nbn.mechanisms.non_parametric.smoothed_empirical_categorical import (
        SmoothedEmpiricalCategoricalMechanism,
    )
    from nbn.scm.categorical import StructuralCategoricalTable

    counts = torch.tensor([[3.0, 1.0], [0.0, 2.0]])
    binning = BinningCategoricalTable.__new__(BinningCategoricalTable)  # the guard runs first
    for mech in (binning, StructuralCategoricalTable()):
        with pytest.raises(NotImplementedError, match="fit_local"):
            mech.fit_from_counts(counts, [2])
    SmoothedEmpiricalCategoricalMechanism().fit_from_counts(counts, [2])  # no fit_local override


def _small():
    v = {"x": ("discrete", 2), "a": ("discrete", 2), "x'": ("discrete", 2)}
    gen = torch.Generator().manual_seed(0)
    data = {k: torch.randint(0, 2, (200,), generator=gen) for k in v}
    return v, data


def test_counts_validate_weights_like_fit():
    v, data = _small()
    hs = HypothesisSet(v, {"x'": [("x",), ("x", "a")]})
    col = torch.ones(200, 1)                                # [N, 1] is accepted, as in fit
    torch.testing.assert_close(hs.counts(data, weights=col)[("x'", ("x",))],
                               hs.counts(data)[("x'", ("x",))])
    with pytest.raises(ValueError):
        hs.counts(data, weights=-torch.ones(200))
    with pytest.raises(ValueError):
        hs.counts(data, weights=torch.ones(10))


def test_weighted_counts_give_no_pac_sets():
    v, data = _small()
    hs = HypothesisSet(v, {"x'": [("x",), ("x", "a")]})
    w = torch.full((200,), 2.0)
    with pytest.raises(ValueError, match="independent draws"):
        hs.factor_sets(data, builder="interval", weights=w)
    with pytest.raises(ValueError, match="independent draws"):
        hs.factor_sets(hs.counts(data, weights=w), builder="l1")
    assert hs.factor_sets(data, builder="dirichlet", weights=w, n_samples=20).builder == "dirichlet"


def test_fixed_edge_that_merges_two_candidates_keeps_one():
    v, _ = _small()
    hs = HypothesisSet(v, {"x'": [("a",), ()]}, fixed_edges=[("a", "x'")])
    assert hs.candidates["x'"] == [("a",)]


def test_same_family_in_another_order_gets_a_clear_message():
    v, data = _small()
    hs = HypothesisSet(v, {"x'": [("x", "a")]})
    fs = hs.factor_sets(data)
    with pytest.raises(ValueError, match="another order"):
        fs.sets_for({"x'": ("a", "x")})


def test_empty_hypothesis_set_is_rejected():
    v, _ = _small()
    with pytest.raises(ValueError, match="at least one node"):
        HypothesisSet(v, {})


def test_union_path_is_skipped_when_its_table_dwarfs_the_data():
    v = {f"p{i}": ("discrete", 10) for i in range(6)}
    v["y"] = ("discrete", 2)
    gen = torch.Generator().manual_seed(0)
    data = {k: torch.randint(0, c[1], (50,), generator=gen) for k, c in v.items()}
    hs = HypothesisSet(v, {"y": [("p0", "p1", "p2"), ("p3", "p4", "p5")]})   # union 10^6 x 2 cells, 50 rows
    assert hs.counts(data).union_used == {"y": False}


def test_model_for_ignores_inputs_no_target_uses():
    v, data = _small()
    v = {**v, "reward": ("continuous", 1)}
    hs = HypothesisSet(v, {"x'": [("x",), ("x", "a")]})
    fc = hs.counts({**data, "reward": torch.randn(200)})
    m = hs.model_for({"x'": ("x",)}, fc)
    assert set(m.variables) == {"x", "x'"}


def test_large_table_warns_before_allocating():
    from nbn.mechanisms.parametric import categorical_table as ct

    mech = CategoricalTableMechanism()
    with pytest.warns(UserWarning, match="CPT has"):
        old = ct._CARDINALITY_WARN_THRESHOLD
        ct._CARDINALITY_WARN_THRESHOLD = 4
        try:
            mech.fit_local(torch.tensor([0, 1, 1]), torch.tensor([[0], [1], [2]]), parent_cards=[3],
                           n_classes=2)
        finally:
            ct._CARDINALITY_WARN_THRESHOLD = old
    assert mech._counts.shape == (3, 2)



def test_strings_are_not_split_into_candidates():
    v, _ = _small()
    with pytest.raises(TypeError, match="list of parent tuples"):
        HypothesisSet(v, {"x'": ("x", "a")})
    hs = HypothesisSet(v, {"x'": [("x",)]})
    with pytest.raises(TypeError, match="tuple of names"):
        hs.edges({"x'": "x"})


def test_results_compare_by_identity():
    v, data = _small()
    hs = HypothesisSet(v, {"x'": [("x",)]})
    a, b = hs.counts(data), hs.counts(data)
    assert a != b and a == a                       # no tensor comparison, no crash


def test_dirichlet_draws_scale_with_the_split_level():
    import warnings

    v, data = _small()
    hs = HypothesisSet(v, {"x'": [("x",), ("x", "a"), ("a",)]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fs = hs.factor_sets(data, builder="dirichlet", delta=0.1)
        for s in fs.sets.values():
            s.bounds()                               # would warn with too few draws
    assert all(s.samples.shape[0] >= 4000 for s in fs.sets.values())


def test_model_for_works_from_factor_set_counts_without_marginals():
    v, data = _small()
    hs = HypothesisSet(v, {"x'": [("x",), ("x", "a")]})
    fs = hs.factor_sets(data)
    assert fs.counts.input_marginals == {}                # not counted again
    full = hs.counts(data)
    assert set(full.input_marginals) == {"x", "a"}
    for h in hs.hypotheses():
        a, b = hs.model_for(h, fs.counts), hs.model_for(h, full)
        for node in a.dag.nodes():                         # marginals summed from factor tables = direct
            assert torch.equal(a.mechanisms[node]._counts, b.mechanisms[node]._counts)


def test_unknown_builder_fails_before_counting():
    v, data = _small()
    hs = HypothesisSet(v, {"x'": [("x",)]})
    with pytest.raises(ValueError, match="Unknown builder"):
        hs.factor_sets({}, builder="Interval")      # no data needed to fail
