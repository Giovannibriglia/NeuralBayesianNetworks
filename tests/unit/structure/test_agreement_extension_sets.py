"""Extension sets in the agreement rule (#333).

Ground truth: ``y = x0 XOR x1``, flipped with probability 0.1; x2..x4 are
irrelevant.  A candidate without x0 and x1 agrees with every one-variable
extension, so only extension sets of size 2 refute it.
"""
from __future__ import annotations

import pytest
import torch

from nbn.structure import agreement_set

INPUTS = [f"x{i}" for i in range(5)]
VARIABLES = dict.fromkeys(INPUTS + ["y"], ("discrete", 2))
FLIP = 0.1


def xor_data(n=20000, seed=0, drop=None):
    """Independent uniform inputs; ``drop`` removes one joint value of (x0, x1)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 2, (n, 5), generator=gen)
    if drop is not None:
        x = x[~((x[:, 0] == drop[0]) & (x[:, 1] == drop[1]))]
    flip = (torch.rand(x.shape[0], generator=gen) < FLIP).long()
    data = {f"x{i}": x[:, i] for i in range(5)}
    data["y"] = (x[:, 0] ^ x[:, 1]) ^ flip
    return data


def truth(x0, x1):
    p1 = (1 - FLIP) if (x0 ^ x1) else FLIP
    return torch.tensor([1 - p1, p1])


def run(data, **kw):
    return agreement_set(VARIABLES, data, "y", INPUTS, max_parents=2, delta=0.05, **kw)


def test_default_is_the_one_variable_rule():
    data = xor_data()
    a, b = run(data), run(data, extension_size=1)
    assert a.extension_size == 1
    assert a.survivors == b.survivors and a.parents == b.parents
    assert a.n_families == b.n_families
    assert torch.equal(a.set.center, b.set.center) and torch.equal(a.set.radius, b.set.radius)


def test_one_variable_extensions_do_not_refute_a_candidate_missing_both_xor_parents():
    res = run(xor_data())
    assert ("x2", "x3") in res.survivors
    assert ("x0", "x1") in res.survivors            # a superset candidate is never discarded
    assert set(res.common_parents) != {"x0", "x1"}


def test_extension_sets_refute_it_and_recover_the_parents():
    res = run(xor_data(), extension_size=2)
    assert res.extension_size == 2
    assert res.survivors == [("x0", "x1")]
    assert res.common_parents == ("x0", "x1")
    assert res.parents == ("x0", "x1")
    assert not res.assumption_refuted
    evidence = res.discarded[("x2", "x3")]
    assert {"x0", "x1"} <= set(evidence["against"]) and evidence["rows"] > 0 and evidence["gap"] > 0
    # more families than the one-variable rule, all sharing the budget
    assert res.n_families > run(xor_data()).n_families
    assert res.delta_per_family == pytest.approx(0.05 / (res.n_families + 1))


@pytest.mark.parametrize("sampling", ["iid", "trajectory"])
def test_truth_is_inside_every_row(sampling):
    res = run(xor_data(seed=1), extension_size=2, sampling=sampling)
    assert res.parents == ("x0", "x1")
    for x0 in range(2):
        for x1 in range(2):
            row = 2 * x0 + x1
            dist = (res.set.center[row] - truth(x0, x1)).abs().sum()
            assert dist <= res.set.radius[row] + 1e-6
            assert not res.set.unseen[row]


def test_a_row_without_data_is_the_whole_simplex():
    res = run(xor_data(drop=(1, 1)), extension_size=2)
    assert res.survivors == [("x0", "x1")]
    row = 2 * 1 + 1
    assert bool(res.set.unseen[row]) and float(res.set.radius[row]) == 2.0
    assert not bool(res.known[row])
    assert int(res.set.unseen.sum()) == 1
    for x0, x1 in ((0, 0), (0, 1), (1, 0)):
        r = 2 * x0 + x1
        assert (res.set.center[r] - truth(x0, x1)).abs().sum() <= res.set.radius[r] + 1e-6


def test_small_data_keeps_everything():
    """With little data nothing is refuted: every candidate survives and nothing is known."""
    res = run(xor_data(n=40), extension_size=2)
    assert len(res.survivors) == len(res.candidates)
    assert res.common_parents == ()
    assert not bool(res.known.any())


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "2"])
def test_extension_size_is_validated(bad):
    with pytest.raises(ValueError, match="extension_size"):
        run(xor_data(n=100), extension_size=bad)


def test_max_families_guards_the_extension_sets():
    with pytest.raises(ValueError, match="max_families"):
        run(xor_data(n=100), extension_size=3, max_families=20)


def test_union_over_a_given_family_ignores_extension_size():
    res = agreement_set(VARIABLES, xor_data(), "y", INPUTS, candidates=[("x0", "x1"), ("x2", "x3")],
                        refute=False, extension_size=2)
    assert res.extension_size == 0 and res.n_families == 2
    assert res.survivors == [("x0", "x1"), ("x2", "x3")]
