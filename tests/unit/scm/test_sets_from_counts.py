"""Builders from count tables (#316): same results as the model-based builders.

``interval_set`` / ``l1_ball_set`` / ``dirichlet_set`` are thin wrappers over
their ``*_from_counts`` twins; ``count_table_for`` counts a node against any
parent tuple.  An exact old-vs-new comparison against ``master`` was run once
for the pull request; these tests pin the wrapper equivalence and the counts.
"""
from __future__ import annotations

import itertools

import pytest
import torch

from nbn import NeuralBayesianNetwork
from nbn.uncertainty import (
    count_table,
    count_table_for,
    dirichlet_set,
    dirichlet_set_from_counts,
    interval_set,
    interval_set_from_counts,
    l1_ball_set,
    l1_ball_set_from_counts,
)
from tests.conftest import available_devices

CARDS = {"a": 2, "b": 3, "x": 3, "y": 4}
PARENT_SETS = [(), ("a",), ("a", "b"), ("b", "a")]


def _data(device: str, n: int = 300, seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    data = {v: torch.randint(0, c, (n,), generator=gen) for v, c in CARDS.items()}
    data["b"][data["a"] == 1] = data["b"][data["a"] == 1].clamp_max(1)  # rows (a=1, b=2) unseen
    return {v: t.to(device) for v, t in data.items()}


def _model(parents: tuple[str, ...], device: str) -> NeuralBayesianNetwork:
    edges = [(p, "y") for p in parents]
    variables = {v: ("discrete", c) for v, c in CARDS.items()}
    model = NeuralBayesianNetwork(edges, variables, device=device)
    model.auto_mechanisms()
    return model


def _brute_force(data, node, parents, weights=None) -> torch.Tensor:
    cards = [CARDS[p] for p in parents]
    rows = list(itertools.product(*[range(c) for c in cards]))
    out = torch.zeros(len(rows), CARDS[node], dtype=torch.float64)
    cols = {v: t.cpu() for v, t in data.items()}
    w = torch.ones(cols[node].numel(), dtype=torch.float64) if weights is None else weights.cpu().double()
    for i in range(cols[node].numel()):
        r = rows.index(tuple(int(cols[p][i]) for p in parents))
        out[r, int(cols[node][i])] += w[i]
    return out


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("parents", PARENT_SETS)
@pytest.mark.parametrize("weighted", [False, True])
def test_count_table_for_matches_brute_force(device, parents, weighted):
    data = _data(device)
    weights = (torch.rand(300, generator=torch.Generator().manual_seed(1)) * 2).to(device) if weighted else None
    model = _model(("a",), device)
    counts, cards = count_table_for(model.variables, data, "y", parents, weights)
    assert cards == [CARDS[p] for p in parents]
    assert counts.device.type == torch.device(device).type
    torch.testing.assert_close(counts.cpu().double(), _brute_force(data, "y", parents, weights),
                               rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("parents", PARENT_SETS)
def test_count_table_is_count_table_for_on_the_dag_parents(device, parents):
    data = _data(device)
    model = _model(parents, device)
    counts, cards = count_table(model, data, "y")
    expected, expected_cards = count_table_for(model.variables, data, "y", model.dag.parents("y"))
    assert cards == expected_cards
    assert torch.equal(counts, expected)


def _equal_sets(a, b) -> None:
    assert type(a) is type(b)
    assert a.level == b.level and a.guarantee == b.guarantee
    for f in ("lower", "upper", "center", "radius", "unseen", "samples"):
        if hasattr(a, f):
            assert torch.equal(getattr(a, f), getattr(b, f)), f


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("parents", [(), ("a",), ("a", "b")])
@pytest.mark.parametrize("sampling", ["iid", "trajectory"])
def test_wrappers_equal_from_counts(device, parents, sampling):
    data = _data(device)
    model = _model(parents, device)
    counts, _ = count_table(model, data, "y")
    for method in ("kl", "hoeffding", "clopper_pearson"):
        _equal_sets(interval_set(model, data, "y", delta=0.1, method=method, sampling=sampling),
                    interval_set_from_counts(counts, delta=0.1, method=method, sampling=sampling))
    _equal_sets(l1_ball_set(model, data, "y", delta=0.1, sampling=sampling),
                l1_ball_set_from_counts(counts, delta=0.1, sampling=sampling))
    torch.manual_seed(3)
    a = dirichlet_set(model, data, "y", delta=0.1, n_samples=200)
    torch.manual_seed(3)
    b = dirichlet_set_from_counts(counts, delta=0.1, n_samples=200)
    _equal_sets(a, b)


@pytest.mark.parametrize("device", available_devices())
def test_unseen_rows_are_the_whole_simplex(device):
    data = _data(device)
    model = _model(("a", "b"), device)
    counts, _ = count_table_for(model.variables, data, "y", ("a", "b"))
    unseen = counts.sum(-1) == 0
    assert bool(unseen[1 * 3 + 2])  # (a=1, b=2) never occurs
    box = interval_set_from_counts(counts, delta=0.1)
    ball = l1_ball_set_from_counts(counts, delta=0.1)
    assert torch.equal(box.unseen, unseen) and torch.equal(ball.unseen, unseen)
    assert bool((box.lower[unseen] == 0).all()) and bool((box.upper[unseen] == 1).all())
    assert bool((ball.radius[unseen] == 2).all())


def test_from_counts_accepts_integer_tables():
    counts = torch.tensor([[3, 1], [0, 0]])
    box = interval_set_from_counts(counts, delta=0.1)
    assert torch.is_floating_point(box.lower)
    _equal_sets(box, interval_set_from_counts(counts.double(), delta=0.1))


@pytest.mark.parametrize("bad", [torch.ones(3), torch.ones(2, 0), torch.tensor([[1.0, -1.0]]),
                                 torch.tensor([[1.0, float("nan")]])])
def test_from_counts_rejects_bad_tables(bad):
    with pytest.raises(ValueError):
        interval_set_from_counts(bad)


def test_count_table_for_validates_parents():
    model = _model(("a",), "cpu")
    data = _data("cpu")
    with pytest.raises(ValueError, match="own parent"):
        count_table_for(model.variables, data, "y", ("y",))
    with pytest.raises(ValueError, match="Repeated"):
        count_table_for(model.variables, data, "y", ("a", "a"))
    with pytest.raises(KeyError):
        count_table_for(model.variables, data, "y", ("z",))


@pytest.mark.parametrize("device", available_devices())
def test_pac_builders_reject_weighted_counts_dirichlet_accepts_them(device):
    data = _data(device)
    model = _model(("a",), device)
    w = torch.full((300,), 0.5, device=device)
    weighted, _ = count_table_for(model.variables, data, "y", ("a",), weights=w)
    with pytest.raises(ValueError, match="whole numbers"):
        interval_set_from_counts(weighted)
    with pytest.raises(ValueError, match="whole numbers"):
        l1_ball_set_from_counts(weighted)
    posterior = dirichlet_set_from_counts(weighted, n_samples=50)
    assert posterior.guarantee == "bayesian"


@pytest.mark.parametrize("device", available_devices())
def test_low_precision_tables_are_upcast(device):
    counts = torch.tensor([[40000.0, 40000.0], [3.0, 1.0]], device=device).half()  # row sum overflows fp16
    ball = l1_ball_set_from_counts(counts, delta=0.1)
    assert ball.center.dtype != torch.float16
    assert torch.allclose(ball.center[0].float().cpu(), torch.tensor([0.5, 0.5]))
    assert bool(torch.isfinite(ball.radius).all()) and float(ball.radius[0]) > 0


def test_count_table_for_rejects_a_bare_string():
    model = _model(("a",), "cpu")
    with pytest.raises(TypeError, match="sequence of names"):
        count_table_for(model.variables, _data("cpu"), "y", "ab")
