"""Unit tests for ``nbn.bench.synthetic.make_synthetic_bn`` (v0.4)."""
from __future__ import annotations

import math
import warnings

import networkx as nx
import pytest
import torch

from nbn.bench import synthetic as synthetic_mod
from nbn.bench.synthetic import (
    SyntheticBN,
    _FAMILIES,
    make_synthetic_bn,
)


# ---------------------------------------------------------------------- #
# Family × size matrix
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("n_nodes", [10, 50, 100])
def test_family_size_matrix_smoke(family: str, n_nodes: int) -> None:
    bn = make_synthetic_bn(
        family=family, n_nodes=n_nodes,
        n_train=200, n_test=50, n_reference=200, seed=0, device="cpu",
    )
    assert isinstance(bn, SyntheticBN)
    assert bn.n_nodes == n_nodes
    assert bn.family == family
    # DAG well-formed
    assert nx.is_directed_acyclic_graph(bn.dag)
    assert len(bn.dag.nodes()) == n_nodes
    # Variable specs cover every node
    assert set(bn.variable_specs.keys()) == set(bn.dag.nodes())
    # Train/test data is keyed by node
    assert set(bn.train_data.keys()) == set(bn.dag.nodes())
    assert set(bn.test_data.keys()) == set(bn.dag.nodes())
    for v in bn.train_data.values():
        assert v.shape[0] == 200
    for v in bn.test_data.values():
        assert v.shape[0] == 50
    # Ground-truth samples present iff continuous (or hybrid)
    if family == "discrete":
        assert bn.ground_truth_samples is None
    else:
        assert bn.ground_truth_samples is not None
        assert bn.ground_truth_samples.shape == (200, n_nodes)


# ---------------------------------------------------------------------- #
# Acyclicity + max_in_degree
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("family", _FAMILIES)
def test_max_in_degree_respected(family: str) -> None:
    bn = make_synthetic_bn(
        family=family, n_nodes=80, max_in_degree=3,
        n_train=100, n_test=50, n_reference=100, seed=7, device="cpu",
    )
    for node in bn.dag.nodes():
        assert bn.dag.in_degree(node) <= 3, (
            f"node {node} has in-degree {bn.dag.in_degree(node)} > 3"
        )


# ---------------------------------------------------------------------- #
# Determinism + independence
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("family", _FAMILIES)
def test_seed_determinism(family: str) -> None:
    kw = dict(
        family=family, n_nodes=20,
        n_train=300, n_test=100, n_reference=300, seed=11, device="cpu",
    )
    a = make_synthetic_bn(**kw)
    b = make_synthetic_bn(**kw)
    for k in a.train_data:
        assert torch.allclose(a.train_data[k].float(), b.train_data[k].float()), (
            f"non-deterministic train_data[{k}]"
        )
    for k in a.test_data:
        assert torch.allclose(a.test_data[k].float(), b.test_data[k].float())
    if a.ground_truth_samples is not None:
        assert torch.allclose(a.ground_truth_samples, b.ground_truth_samples)


@pytest.mark.parametrize("family", _FAMILIES)
def test_seed_independence(family: str) -> None:
    a = make_synthetic_bn(
        family=family, n_nodes=20,
        n_train=200, n_test=50, n_reference=200, seed=0, device="cpu",
    )
    b = make_synthetic_bn(
        family=family, n_nodes=20,
        n_train=200, n_test=50, n_reference=200, seed=1, device="cpu",
    )
    different = sum(
        (not torch.allclose(a.train_data[k].float(), b.train_data[k].float()))
        for k in a.train_data
    )
    assert different > 0, f"family={family}: two seeds produced identical data"


def test_global_rng_not_perturbed() -> None:
    """The generator must restore the caller's torch RNG state on exit."""
    torch.manual_seed(99)
    pre = torch.rand(1).item()
    torch.manual_seed(99)
    _ = make_synthetic_bn(
        family="hybrid", n_nodes=15,
        n_train=100, n_test=50, n_reference=100, seed=42, device="cpu",
    )
    post = torch.rand(1).item()
    assert math.isclose(pre, post, abs_tol=1e-9), (
        f"caller RNG perturbed: pre={pre} post={post}"
    )


# ---------------------------------------------------------------------- #
# Sanity check: data → fitter → close to true parameters
# ---------------------------------------------------------------------- #


def test_lg_fitter_recovers_true_weights_within_tolerance() -> None:
    """Sanity: a fresh LG NBN fitted on train_data should land near the truth.

    We pick a single non-root continuous_lg node, refit a brand-new
    ``LinearGaussianMechanism`` on that node's training data, and check
    that the fitted weight is within Frobenius distance < 0.5 of the
    true mechanism's weight at ``n_train=10_000``.
    """
    bn = make_synthetic_bn(
        family="continuous_lg", n_nodes=20, edge_density=0.30,
        n_train=10_000, n_test=200, n_reference=2000,
        seed=3, device="cpu",
    )
    target = next(
        n for n in bn.dag.nodes() if list(bn.dag.predecessors(n))
    )
    parents = list(bn.dag.predecessors(target))
    pa = torch.cat(
        [bn.train_data[p].reshape(-1, 1).float() for p in parents], dim=-1,
    )
    x = bn.train_data[target].reshape(-1, 1).float()

    from nbn.mechanisms.parametric.linear_gaussian import LinearGaussianMechanism
    fitted = LinearGaussianMechanism(ridge=1e-4)
    fitted.fit_local(x, pa)

    true_mech = bn.true_model.mechanisms[target]
    true_w = true_mech._weight.detach().reshape(-1)
    fit_w = fitted._weight.detach().reshape(-1)
    err = (true_w - fit_w).norm().item()
    assert err < 0.5, (
        f"LG fitter weights diverged from truth: ||Δw|| = {err:.3f}"
    )


# ---------------------------------------------------------------------- #
# Memory-bound warning
# ---------------------------------------------------------------------- #


def test_large_network_halves_n_reference_with_warning(monkeypatch) -> None:
    """The assertion is that the guard fires and halves n_reference.

    That needs a network *over the threshold*, not a large one — so lower the
    threshold rather than building the 1000-node network the production
    default implies.  Same code path, same warning, same halving, ~70s less
    wall clock in every CI run.
    """
    monkeypatch.setattr(synthetic_mod, "_REFERENCE_LARGE_NETWORK_THRESHOLD", 20)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bn = make_synthetic_bn(
            family="discrete", n_nodes=25,
            n_train=200, n_test=50, n_reference=50_000,
            seed=0, device="cpu",
        )
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("halving n_reference" in m for m in msgs), (
        f"expected halving warning, got: {msgs}"
    )
    assert isinstance(bn, SyntheticBN)


# ---------------------------------------------------------------------- #
# Validation
# ---------------------------------------------------------------------- #


def test_invalid_family_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown family"):
        make_synthetic_bn(
            family="bogus",  # type: ignore[arg-type]
            n_nodes=10, n_train=10, n_test=5, n_reference=10, device="cpu",
        )


def test_invalid_density_rejected() -> None:
    with pytest.raises(ValueError, match="edge_density"):
        make_synthetic_bn(
            family="discrete", n_nodes=10, edge_density=2.0,
            n_train=10, n_test=5, n_reference=10, device="cpu",
        )


def test_hybrid_requires_strict_continuous_fraction() -> None:
    with pytest.raises(ValueError, match="fraction_continuous"):
        make_synthetic_bn(
            family="hybrid", n_nodes=10, fraction_continuous=0.0,
            n_train=10, n_test=5, n_reference=10, device="cpu",
        )
