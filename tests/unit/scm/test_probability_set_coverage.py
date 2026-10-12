"""Coverage of the probability sets: the truth must be inside at the declared level.

A set of level ``1 - delta`` that is valid for all entries at once must contain
the whole true table in at least ``1 - delta`` of independent replicates.  We
accept a Monte-Carlo shortfall of three standard errors.
"""
from __future__ import annotations

import math

import pytest
import torch

from nbn.scm import noise_parameter_set
from nbn.uncertainty import dirichlet_set, interval_set, l1_ball_set
from tests.unit.scm.simulators import (
    TRUE_Q,
    random_chain_episodes,
    rehab_iid,
    rehab_model,
    rehab_true_table,
)

DELTA = 0.1


def _min_coverage(replicates: int, delta: float = DELTA) -> float:
    return 1 - delta - 3 * math.sqrt(delta * (1 - delta) / replicates)


def _chain_model():
    from nbn import NeuralBayesianNetwork

    model = NeuralBayesianNetwork(
        [("s", "s'"), ("a", "s'")],
        {"s": ("discrete", 3), "a": ("discrete", 2), "s'": ("discrete", 3)}, device="cpu",
    )
    model.auto_mechanisms()
    return model


def _coverage(replicates: int, make_set, truth_check) -> float:
    hits = 0
    for r in range(replicates):
        gen = torch.Generator().manual_seed(1000 + r)
        hits += int(truth_check(make_set(gen)))
    return hits / replicates


@pytest.mark.parametrize("builder", ["interval", "l1"])
def test_iid_sets_cover_the_true_table(builder):
    replicates = 60
    truth = rehab_true_table()
    model = rehab_model()

    def make(gen):
        data = rehab_iid(400, generator=gen).data
        if builder == "interval":
            return interval_set(model, data, "h'", delta=DELTA)
        return l1_ball_set(model, data, "h'", delta=DELTA)

    assert _coverage(replicates, make, lambda s: s.contains(truth)) >= _min_coverage(replicates)


@pytest.mark.parametrize("builder", ["interval", "l1"])
def test_trajectory_sets_cover_the_true_kernel(builder):
    replicates = 60
    model = _chain_model()

    def make(gen):
        data, kernel = random_chain_episodes(20, 15, generator=gen)
        make.kernel = kernel
        if builder == "interval":
            return interval_set(model, data, "s'", delta=DELTA, sampling="trajectory")
        return l1_ball_set(model, data, "s'", delta=DELTA, sampling="trajectory")

    assert _coverage(replicates, make, lambda s: s.contains(make.kernel)) >= _min_coverage(replicates)


def test_noise_polytope_covers_the_true_noise_law():
    replicates = 40

    def make(gen):
        data = rehab_iid(400, generator=gen).data
        model = rehab_model()
        model.fit(data)
        return noise_parameter_set(model, data, ["h'"], delta=DELTA)

    assert _coverage(replicates, make, lambda s: s.contains(TRUE_Q.double())) >= _min_coverage(replicates)


@pytest.mark.slow
@pytest.mark.parametrize("builder", ["interval", "l1", "noise"])
def test_coverage_large_monte_carlo(builder):
    replicates = 400
    truth = rehab_true_table()

    def make(gen):
        data = rehab_iid(300, generator=gen).data
        model = rehab_model()
        if builder == "interval":
            return interval_set(model, data, "h'", delta=DELTA)
        if builder == "l1":
            return l1_ball_set(model, data, "h'", delta=DELTA)
        model.fit(data)
        return noise_parameter_set(model, data, ["h'"], delta=DELTA)

    check = (lambda s: s.contains(TRUE_Q.double())) if builder == "noise" else (lambda s: s.contains(truth))
    assert _coverage(replicates, make, check) >= _min_coverage(replicates)


# ---------------------------------------------------------------------------
# Tightness and soundness
# ---------------------------------------------------------------------------


def test_shared_noise_gives_a_much_tighter_table():
    data = rehab_iid(5000, generator=torch.Generator().manual_seed(0)).data
    model = rehab_model()
    model.fit(data)
    per_entry = interval_set(model, data, "h'", delta=DELTA)
    pooled = noise_parameter_set(model, data, ["h'"], delta=DELTA).table_set("h'")
    assert float(pooled.width().mean()) < 0.5 * float(per_entry.width().mean())
    assert pooled.contains(rehab_true_table())


def test_tightened_polytope_is_inside_the_raw_one():
    pytest.importorskip("scipy")
    data = rehab_iid(2000, generator=torch.Generator().manual_seed(0)).data
    model = rehab_model()
    model.fit(data)
    raw = noise_parameter_set(model, data, ["h'"], delta=DELTA)
    tight = raw.tightened()
    assert bool((tight.lower >= raw.lower - 1e-12).all()) and bool((tight.upper <= raw.upper + 1e-12).all())
    assert tight.contains(TRUE_Q.double())


def test_width_shrinks_like_inverse_square_root_of_n():
    model = rehab_model()
    sizes = [500, 2000, 8000]
    widths = []
    for n in sizes:
        data = rehab_iid(n, generator=torch.Generator().manual_seed(n)).data
        widths.append(float(interval_set(model, data, "h'", delta=DELTA).width().mean()))
    slope = (math.log(widths[-1]) - math.log(widths[0])) / (math.log(sizes[-1]) - math.log(sizes[0]))
    assert -0.65 < slope < -0.35


def test_unseen_rows_get_the_whole_simplex_and_a_flag():
    data = rehab_iid(2000, generator=torch.Generator().manual_seed(0)).data
    keep = data["a"] == 0  # never train: half the rows are unseen
    data = {k: v[keep] for k, v in data.items()}
    model = rehab_model()
    box = interval_set(model, data, "h'")
    ball = l1_ball_set(model, data, "h'")
    for s in (box, ball):
        lower, upper = s.bounds()
        assert int(s.unseen.sum()) == 8
        assert bool((lower[s.unseen] == 0).all()) and bool((upper[s.unseen] == 1).all())


def test_dirichlet_posterior_mean_is_the_smoothed_frequency():
    from nbn.uncertainty import count_table

    torch.manual_seed(0)
    data = rehab_iid(5000, generator=torch.Generator().manual_seed(0)).data
    model = rehab_model()
    post = dirichlet_set(model, data, "h'", prior=1.0, n_samples=4000)
    counts, _ = count_table(model, data, "h'")
    expected = (counts + 1.0) / (counts.sum(-1, keepdim=True) + 4.0)
    assert post.samples.shape == (4000, 16, 4)
    assert torch.allclose(post.samples.mean(0).double(), expected, atol=5e-3)
