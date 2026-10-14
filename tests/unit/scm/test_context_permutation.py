"""Permutation reference of ``context_invariance_test`` (NBN #328).

The chi-square reference of the stratified G statistic over-rejects a true
null when many parent configurations hold few rows; the permutation reference
(context labels exchanged within each configuration, margins fixed) keeps the
level at any sample size.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from nbn.scm import StructuralCausalModel, context_invariance_test
from nbn.scm.diagnostics import _g_numpy, _stratified_g, _stratified_g_permutation

N_PARENTS = 6  # 64 parent configurations


def _sparse_model(device="cpu"):
    parents = [f"x{i}" for i in range(N_PARENTS)]
    variables = dict.fromkeys(parents, ("discrete", 2))
    variables.update({"y": ("discrete", 2), "c": ("discrete", 2)})
    return StructuralCausalModel([(p, "y") for p in parents], variables, context="c", device=device)


def _sparse_data(n, seed, drift=0.0):
    """``y`` depends on the parity of the parents; ``drift`` shifts it in context 1."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 2, (n, N_PARENTS), generator=gen)
    c = torch.randint(0, 2, (n,), generator=gen)
    p_y = 0.2 + 0.6 * (x.sum(1) % 2).float() + drift * c.float()
    y = (torch.rand(n, generator=gen) < p_y.clamp(0, 1)).long()
    data = {f"x{i}": x[:, i].float() for i in range(N_PARENTS)}
    data.update(y=y.float(), c=c.float())
    return data


def test_permutation_keeps_the_level_where_chi_square_over_rejects():
    model = _sparse_model()
    reps, alpha = 200, 0.05
    perm = chi = 0
    for seed in range(reps):
        data = _sparse_data(200, seed)  # about 3 rows per configuration and context
        gen = torch.Generator().manual_seed(seed)
        perm += context_invariance_test(model, data, "y", n_perm=199, generator=gen).p_value <= alpha
        chi += context_invariance_test(model, data, "y", reference="chi2").p_value <= alpha
    # binomial(200, 0.05): mean 10, sd 3.1; 20 is about 3 sd above the nominal level
    assert perm <= 20
    assert chi > perm and chi >= 25


def test_permutation_has_power_against_a_changed_mechanism():
    model = _sparse_model()
    data = _sparse_data(2000, 0, drift=0.15)
    res = context_invariance_test(model, data, "y", generator=torch.Generator().manual_seed(0))
    assert res.p_value <= 1 / 1000
    assert res.details["method"] == "G-test, permutation reference" and res.details["n_perm"] == 999


def test_statistic_matches_the_chi_square_branch_and_is_reproducible():
    model = _sparse_model()
    data = _sparse_data(500, 3, drift=0.1)
    a = context_invariance_test(model, data, "y", n_perm=99, generator=torch.Generator().manual_seed(7))
    b = context_invariance_test(model, data, "y", n_perm=99, generator=torch.Generator().manual_seed(7))
    chi = context_invariance_test(model, data, "y", reference="chi2")
    assert a.p_value == b.p_value
    assert a.statistic == pytest.approx(chi.statistic, rel=1e-9, abs=1e-9)
    assert chi.details["method"] == "G-test, chi-square reference"


def test_p_value_grid_and_floor():
    model = _sparse_model()
    data = _sparse_data(300, 1)
    for n_perm in (1, 19, 99):
        p = context_invariance_test(model, data, "y", n_perm=n_perm,
                                    generator=torch.Generator().manual_seed(0)).p_value
        assert 1 / (n_perm + 1) <= p <= 1.0
        assert abs(p * (n_perm + 1) - round(p * (n_perm + 1))) < 1e-9  # (1 + count) / (n_perm + 1)


def test_exact_permutation_law_on_a_2x2_table():
    """The sequential hypergeometric draws are exact: brute-force the law of a tiny table."""
    t = torch.tensor([[[2.0, 1.0], [0.0, 2.0]]])  # one stratum, 2 contexts x 2 values
    # Under the null, the count n_00 is hypergeometric(N=5, K=2 zeros, n=3 rows in context 0).
    labels = [0, 0, 0, 1, 1]
    values = [0, 0, 1, 1, 1]
    g_all = []
    for perm in itertools.permutations(range(5)):
        tab = np.zeros((2, 2))
        for i, j in enumerate(perm):
            tab[labels[i], values[j]] += 1
        g_all.append(_g_numpy(tab))
    g_obs = _g_numpy(t[0].double().numpy())  # float64, as in the helper: float32 rounding breaks ties
    exact = float(np.mean(np.array(g_all) >= g_obs - 1e-9 * max(1.0, g_obs)))
    assert exact == pytest.approx(0.4)        # hypergeometric: P(n00 = 2) + P(n00 = 0) = 3/10 + 1/10
    p, g = _stratified_g_permutation(t, 20000, torch.Generator().manual_seed(0))
    assert g == pytest.approx(g_obs)
    assert p == pytest.approx(exact, abs=0.02)


def test_strata_with_one_context_or_one_value_contribute_nothing():
    t = torch.zeros(3, 2, 2)
    t[0] = torch.tensor([[4.0, 0.0], [0.0, 0.0]])  # one context
    t[1] = torch.tensor([[3.0, 0.0], [2.0, 0.0]])  # one value
    t[2] = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    p, g = _stratified_g_permutation(t, 99, torch.Generator().manual_seed(0))
    assert g == pytest.approx(_stratified_g(t[2:])[0])
    assert 0.0 < p <= 1.0


def test_invalid_arguments():
    model = _sparse_model()
    data = _sparse_data(100, 0)
    with pytest.raises(ValueError, match="reference"):
        context_invariance_test(model, data, "y", reference="exact")  # type: ignore[arg-type]
    for bad in (0, 10.5, True):
        with pytest.raises(ValueError, match="n_perm"):
            context_invariance_test(model, data, "y", n_perm=bad)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
                         + (["mps"] if torch.backends.mps.is_available() else []))
def test_permutation_reference_on_every_device(device):
    model = _sparse_model(device)
    data = _sparse_data(400, 2, drift=0.3)
    cpu = context_invariance_test(_sparse_model(), data, "y", n_perm=99,
                                  generator=torch.Generator().manual_seed(1))
    res = context_invariance_test(model, data, "y", n_perm=99, generator=torch.Generator().manual_seed(1))
    assert res.p_value == cpu.p_value
    assert res.statistic == pytest.approx(cpu.statistic, rel=1e-5)


def _exact_p(labels, values, n_ctx, k):
    """Exact permutation p-value of one stratum by enumerating every relabelling."""
    def g_of(vals):
        tab = np.zeros((n_ctx, k))
        for a, b in zip(labels, vals):
            tab[a, b] += 1
        return _g_numpy(tab)

    g_obs = g_of(values)
    g_all = np.array([g_of([values[j] for j in perm]) for perm in itertools.permutations(range(len(values)))])
    return float(np.mean(g_all >= g_obs - 1e-9 * max(1.0, g_obs)))


def test_three_contexts_three_values_match_the_exact_permutation_law():
    labels = [0, 0, 1, 1, 2, 2, 2]
    values = [0, 1, 1, 2, 2, 2, 0]
    t = torch.zeros(1, 3, 3)
    for a, b in zip(labels, values):
        t[0, a, b] += 1
    p, _ = _stratified_g_permutation(t, 20000, torch.Generator().manual_seed(0))
    assert p == pytest.approx(_exact_p(labels, values, 3, 3), abs=0.015)


def test_chunking_over_replicates_and_strata_keeps_the_exact_law():
    """Tiny blocks (one replicate, one stratum at a time) give the same statistic and the exact law."""
    labels = [0, 0, 1, 1, 2, 2, 2]
    values = [0, 1, 1, 2, 2, 2, 0]
    t = torch.zeros(3, 3, 3)
    for a, b in zip(labels, values):
        t[0, a, b] += 1
    t[1] = t[0]                                   # two identical strata, one uninformative
    t[2, 0, 0] = 4.0
    p_big, g_big = _stratified_g_permutation(t, 20000, torch.Generator().manual_seed(0))
    p_small, g_small = _stratified_g_permutation(t, 20000, torch.Generator().manual_seed(0), max_elements=7)
    assert g_small == pytest.approx(g_big, rel=1e-12) == pytest.approx(float(_stratified_g(t)[0]), rel=1e-12)
    # exact law of the sum over the two identical strata, by convolving the one-stratum law
    one = []
    for perm in itertools.permutations(range(len(values))):
        tab = np.zeros((3, 3))
        for i, j in enumerate(perm):
            tab[labels[i], values[j]] += 1
        one.append(_g_numpy(tab))
    one = np.array(one)
    total = (one[:, None] + one[None, :]).ravel()
    exact = float(np.mean(total >= g_big - 1e-9 * max(1.0, g_big)))
    for p in (p_big, p_small):
        assert p == pytest.approx(exact, abs=0.015)


def test_sparse_context_ids_cost_nothing():
    """Contexts {0, 999} give the same tables, hence the same draws, as contexts {0, 1}."""
    t = torch.zeros(8, 1000, 3)
    dense = torch.zeros(8, 2, 3)
    gen = torch.Generator().manual_seed(0)
    dense[:] = torch.randint(0, 6, (8, 2, 3), generator=gen).float()
    t[:, 0], t[:, 999] = dense[:, 0], dense[:, 1]
    a = _stratified_g_permutation(t, 999, torch.Generator().manual_seed(3))
    b = _stratified_g_permutation(dense, 999, torch.Generator().manual_seed(3))
    assert a == b


def test_rejection_includes_the_level_and_the_gate_warns_at_an_impossible_alpha():
    from nbn.scm import DiagnosticResult
    from nbn.uncertainty import confounding_gate

    res = DiagnosticResult("context_invariance", 9.0, 0.001, null="", details={"n_perm": 999})
    assert res.rejected(0.001) and not res.rejected(0.0009)
    assert confounding_gate(True, res, alpha=0.001).use_confounded_rows
    with pytest.warns(UserWarning, match="can never reject"):
        confounding_gate(True, res, alpha=0.0005)


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else [])
                         + (["mps"] if torch.backends.mps.is_available() else []))
def test_cpu_generator_works_under_another_default_device(device):
    model = _sparse_model()
    data = _sparse_data(300, 5)
    ref = context_invariance_test(model, data, "y", n_perm=49, generator=torch.Generator().manual_seed(0))
    old = torch.get_default_device()
    torch.set_default_device(device)
    try:
        res = context_invariance_test(model, data, "y", n_perm=49, generator=torch.Generator(device="cpu").manual_seed(0))
    finally:  # None removes the device mode; setting "cpu" back would leave it installed
        torch.set_default_device(None if old.type == "cpu" else old)
    assert res.p_value == ref.p_value
