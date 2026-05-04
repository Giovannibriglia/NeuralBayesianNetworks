"""v0.4 contract: continuous baseline adapters return posterior **samples**.

These tests pin the v0.4 ``query_batch_samples`` shape and dispersion
contract that the inference crash test depends on.  They run on small
synthetic ``continuous_lg`` BNs so Pyro's ``Importance`` sampler stays
under a few seconds.
"""
from __future__ import annotations

import pytest
import torch

from benchmarking.domains.base import BenchmarkProblem, Query


# ---------------------------------------------------------------------- #
# Fixture: a tiny continuous_lg BenchmarkProblem
# ---------------------------------------------------------------------- #


def _make_lg_problem(b: int = 8) -> tuple[BenchmarkProblem, Query]:
    """Build a 4-node Linear-Gaussian benchmark problem for adapter tests."""
    from benchmarking.synthetic import make_synthetic_bn
    bn = make_synthetic_bn(
        family="continuous_lg", n_nodes=4, edge_density=0.5,
        n_train=400, n_test=100, n_reference=200, seed=0, device="cpu",
    )
    problem = BenchmarkProblem(
        name="lg4", dag=list(bn.dag.edges()), variables=bn.variable_specs,
        train_data={k: v.float() for k, v in bn.train_data.items()},
        test_data={k: v.float() for k, v in bn.test_data.items()},
        queries=[], ground_truth=None,
    )
    target = "X3"
    parents = [p for p in problem.variables if p != target][:2]
    evidence = {p: bn.test_data[p][:b].float().reshape(b) for p in parents}
    q = Query(targets=[target], evidence=evidence, kind="marginal")
    return problem, q


# ---------------------------------------------------------------------- #
# pgmpy
# ---------------------------------------------------------------------- #


def test_pgmpy_lg_returns_samples_with_dispersion() -> None:
    pytest.importorskip("pgmpy")
    from benchmarking.baselines.pgmpy_adapter import PgmpyAdapter
    problem, q = _make_lg_problem(b=8)
    adapter = PgmpyAdapter()
    adapter.fit(problem)
    if adapter.kind != "continuous_lg":
        pytest.skip(
            f"pgmpy LinearGaussianBayesianNetwork not fully supported "
            f"in this pgmpy version (kind={adapter.kind!r})",
        )
    samples = adapter.query_batch_samples(q, n_samples=200)
    assert samples.shape == (8, 200, 1), f"got {samples.shape}"
    assert not torch.isnan(samples).any(), "pgmpy LG must not produce NaNs"
    assert samples.std(dim=1).min().item() > 1e-3, (
        "pgmpy LG samples have zero per-row variance — these look like "
        "stacked means rather than posterior samples"
    )


def test_pgmpy_returns_nan_sentinel_on_unsupported_family() -> None:
    pytest.importorskip("pgmpy")
    from benchmarking.baselines.pgmpy_adapter import PgmpyAdapter
    from benchmarking.synthetic import make_synthetic_bn
    bn = make_synthetic_bn(
        family="continuous_nongauss", n_nodes=4, edge_density=0.5,
        n_train=200, n_test=50, n_reference=100, seed=0, device="cpu",
    )
    problem = BenchmarkProblem(
        name=bn.name, dag=list(bn.dag.edges()), variables=bn.variable_specs,
        train_data={k: v.float() for k, v in bn.train_data.items()},
        test_data={k: v.float() for k, v in bn.test_data.items()},
        queries=[], ground_truth=None,
    )
    adapter = PgmpyAdapter()
    adapter.fit(problem)
    target = "X3"
    parent = next(p for p in problem.variables if p != target)
    q = Query(
        targets=[target],
        evidence={parent: bn.test_data[parent][:4].float().reshape(4)},
        kind="marginal",
    )
    samples = adapter.query_batch_samples(q, n_samples=64)
    assert samples.shape == (4, 64, 1)
    assert torch.isnan(samples).all(), (
        "pgmpy must return a NaN sentinel for non-Gaussian continuous"
    )


# ---------------------------------------------------------------------- #
# gpytorch
# ---------------------------------------------------------------------- #


def test_gpytorch_returns_samples_with_dispersion() -> None:
    pytest.importorskip("gpytorch")
    from benchmarking.baselines.gpytorch_adapter import GPyTorchAdapter
    problem, q = _make_lg_problem(b=4)
    adapter = GPyTorchAdapter(device="cpu")
    adapter.fit(problem)
    samples = adapter.query_batch_samples(q, n_samples=128)
    assert samples.shape == (4, 128, 1), f"got {samples.shape}"
    assert not torch.isnan(samples).any()
    assert samples.std(dim=1).min().item() > 1e-3


def test_gpytorch_skips_discrete_evidence_with_empty_sentinel() -> None:
    pytest.importorskip("gpytorch")
    from benchmarking.baselines.gpytorch_adapter import GPyTorchAdapter
    from benchmarking.synthetic import make_synthetic_bn
    bn = make_synthetic_bn(
        family="hybrid", n_nodes=6, edge_density=0.4,
        n_train=200, n_test=50, n_reference=100, seed=1, device="cpu",
    )
    problem = BenchmarkProblem(
        name="hyb6", dag=list(bn.dag.edges()), variables=bn.variable_specs,
        train_data={k: v.float() for k, v in bn.train_data.items()},
        test_data={k: v.float() for k, v in bn.test_data.items()},
        queries=[], ground_truth=None,
    )
    adapter = GPyTorchAdapter(device="cpu")
    adapter.fit(problem)
    cont_target = next(
        n for n, (k, _) in problem.variables.items() if k == "continuous"
    )
    disc_evidence = next(
        n for n, (k, _) in problem.variables.items() if k == "discrete"
    )
    q = Query(
        targets=[cont_target],
        evidence={disc_evidence: torch.tensor([0, 1, 0, 1])},
        kind="marginal",
    )
    samples = adapter.query_batch_samples(q, n_samples=32)
    assert samples.shape == (0, 32, 1), (
        f"GPyTorch must signal not-applicable on discrete evidence with a "
        f"[0, n_samples, |T|] tensor; got {samples.shape}"
    )


# ---------------------------------------------------------------------- #
# pyro
# ---------------------------------------------------------------------- #


def test_pyro_returns_samples_with_dispersion() -> None:
    pytest.importorskip("pyro")
    from benchmarking.baselines.pyro_adapter import PyroAdapter
    problem, q = _make_lg_problem(b=2)   # B=2 → tractable importance loop
    adapter = PyroAdapter(n_samples=64)
    adapter.fit(problem)
    samples = adapter.query_batch_samples(q, n_samples=64)
    assert samples.shape == (2, 64, 1), f"got {samples.shape}"
    assert not torch.isnan(samples).any()
    assert samples.std(dim=1).min().item() > 1e-3
