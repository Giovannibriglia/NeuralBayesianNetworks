"""Port of tests/integration/test_inference_accuracy_oracle.py to v0.13.

Tests the v0.13 oracle helpers in benchmarking/core/oracle.py.  The
oracle functions were adapted from crash_test_runner v0.12 private helpers;
these tests verify the same behavioral contracts against the v0.13 API.

Original tests:
  test_forward_clamp_matches_analytic_lg_posterior         → PORT
  test_forward_clamp_matches_tight_rejection_at_marginal_mean → PORT
  test_forward_clamp_handles_degenerate_evidence_dim       → PORT
  test_filter_ground_truth_handles_mixed_device_evidence   → PORT (gpu)
"""
from __future__ import annotations

import math

import pytest
import torch

from benchmarking.core.oracle import (
    filter_ground_truth,
    forward_with_clamp_posterior_samples,
)
from benchmarking.problems import SyntheticConfig, SyntheticProblemSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_problem(family: str, n_nodes: int = 5, seed: int = 0, device: str = "cpu"):
    src = SyntheticProblemSource()
    cfg = SyntheticConfig(
        families=[family],
        n_nodes_list=[n_nodes],
        seeds=[seed],
        n_train=500,
        n_test=100,
        n_reference=200_000 if family == "continuous_nongauss" else 2000,
        edge_density=0.40,
        max_in_degree=2,
        device=device,
    )
    return next(iter(src.iter_problems(cfg)))


def _scm_matrices(problem):
    """Extract A, b, d from a true_model whose every node is LinearGaussian."""
    import networkx as nx

    dag_nx = nx.DiGraph()
    dag_nx.add_nodes_from(problem.variables)
    dag_nx.add_edges_from(problem.dag)
    nodes = list(nx.topological_sort(dag_nx))
    n = len(nodes)

    A = torch.zeros(n, n)
    b = torch.zeros(n)
    d = torch.zeros(n)
    for node in nodes:
        j = nodes.index(node)
        parents = list(dag_nx.predecessors(node))
        mech = problem.true_model.mechanisms[node]
        b[j] = mech._bias.item()
        d[j] = torch.exp(mech._log_scale).item() ** 2
        for k, p in enumerate(parents):
            A[j, nodes.index(p)] = mech._weight[k].item()
    return A, b, d, nodes


# ---------------------------------------------------------------------------
# Test 1 — forward-with-clamp matches analytic LG posterior
# ---------------------------------------------------------------------------

def test_forward_clamp_matches_analytic_lg_posterior() -> None:
    """forward_with_clamp_posterior_samples on continuous_lg must match
    the closed-form Gaussian posterior within 5% of analytic std."""
    torch.manual_seed(0)
    problem = _make_problem("continuous_lg", n_nodes=5, seed=0)

    nodes = list(problem.variables.keys())
    target = nodes[-1]
    ev_nodes = nodes[:2]
    evidence = {
        ev_nodes[0]: torch.tensor([1.5]),
        ev_nodes[1]: torch.tensor([-1.2]),
    }

    # Closed-form posterior
    A, b, d, sorted_nodes = _scm_matrices(problem)
    M = torch.linalg.inv(torch.eye(len(A)) - A)
    mu = M @ b
    Sigma = M @ torch.diag(d) @ M.T
    t_idx = sorted_nodes.index(target)
    e_idx = [sorted_nodes.index(n) for n in ev_nodes]
    e_vals = torch.tensor([float(evidence[n].item()) for n in ev_nodes])
    post_mean = mu[t_idx] + Sigma[t_idx, e_idx] @ torch.linalg.solve(
        Sigma[e_idx][:, e_idx], e_vals - mu[e_idx],
    )
    post_var = Sigma[t_idx, t_idx] - Sigma[t_idx, e_idx] @ torch.linalg.solve(
        Sigma[e_idx][:, e_idx], Sigma[e_idx, t_idx],
    )
    post_std = post_var.clamp_min(1e-12).sqrt()

    # v0.13 oracle (scalar evidence → reshape to [1])
    ev_row = {k: v.reshape(-1)[0] for k, v in evidence.items()}
    samples = forward_with_clamp_posterior_samples(
        problem, [target], ev_row, n_samples=10_000,
    )
    assert samples is not None and samples.shape[0] == 10_000
    fwc_mean = float(samples.mean().item())
    fwc_std = float(samples.std().item())

    assert abs(fwc_mean - float(post_mean)) < 0.05 * float(post_std), (
        f"oracle mean {fwc_mean:.4f} differs from analytic {float(post_mean):.4f} "
        f"by > 5% of analytic std {float(post_std):.4f}"
    )
    assert abs(fwc_std - float(post_std)) < 0.10 * float(post_std), (
        f"oracle std {fwc_std:.4f} differs from analytic {float(post_std):.4f} by > 10%"
    )


# ---------------------------------------------------------------------------
# Test 2 — forward-with-clamp matches tight rejection at marginal mean
# ---------------------------------------------------------------------------

def test_forward_clamp_matches_tight_rejection_at_marginal_mean() -> None:
    """On continuous_nongauss with evidence at the marginal mean, the two
    oracle estimates agree within 3-SE."""
    torch.manual_seed(0)
    problem = _make_problem("continuous_nongauss", n_nodes=5, seed=0)

    import networkx as nx
    dag_nx = nx.DiGraph()
    dag_nx.add_nodes_from(problem.variables)
    dag_nx.add_edges_from(problem.dag)
    topo_nodes = list(nx.topological_sort(dag_nx))

    target = topo_nodes[-1]
    ev_node = topo_nodes[0]
    ev_idx = topo_nodes.index(ev_node)
    target_idx = topo_nodes.index(target)

    samples_pool = problem.ground_truth.samples
    ev_marginal = samples_pool[:, ev_idx]
    ev_mean = float(ev_marginal.mean().item())
    ev_std = float(ev_marginal.std().item())
    evidence = {ev_node: torch.tensor(ev_mean)}

    # Tight rejection
    eps = 0.05 * ev_std
    mask = (ev_marginal - ev_mean).abs() < eps
    n_eff = int(mask.sum().item())
    if n_eff < 100:
        eps = 0.20 * ev_std
        mask = (ev_marginal - ev_mean).abs() < eps
        n_eff = int(mask.sum().item())
    rejection_target = samples_pool[mask, target_idx]
    rejection_mean = float(rejection_target.mean().item())
    rejection_std = float(rejection_target.std().item())
    se = rejection_std / math.sqrt(max(1, n_eff))

    fwc_samples = forward_with_clamp_posterior_samples(
        problem, [target], evidence, n_samples=10_000,
    )
    assert fwc_samples is not None
    fwc_mean = float(fwc_samples.mean().item())

    fwc_se = rejection_std / math.sqrt(10_000)
    tol = 3 * (se + fwc_se)
    assert abs(fwc_mean - rejection_mean) < tol, (
        f"forward-clamp {fwc_mean:.4f} differs from tight-rejection "
        f"{rejection_mean:.4f} (n_eff={n_eff}) by > 3·SE={tol:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3 — forward-with-clamp handles degenerate (scalar) evidence
# ---------------------------------------------------------------------------

def test_forward_clamp_handles_degenerate_evidence_dim() -> None:
    """Scalar evidence values are accepted; output shape is (n_samples, 1)."""
    torch.manual_seed(0)
    problem = _make_problem("continuous_lg", n_nodes=4, seed=0)

    nodes = list(problem.variables.keys())
    samples = forward_with_clamp_posterior_samples(
        problem, [nodes[-1]], {nodes[0]: torch.tensor(0.5)}, n_samples=200,
    )
    assert samples is not None and samples.shape == (200, 1)


# ---------------------------------------------------------------------------
# Test 4 — filter_ground_truth handles mixed-device evidence (gpu)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires cuda to surface the device-mismatch path",
)
def test_filter_ground_truth_handles_mixed_device_evidence() -> None:
    """filter_ground_truth must not raise RuntimeError when ground_truth.samples
    are on CUDA but evidence values are on CPU (v0.5c bug 1 regression)."""
    problem = _make_problem("hybrid", n_nodes=10, seed=0, device="cuda")

    continuous_node = next(
        n for n, (kind, _) in problem.variables.items() if kind == "continuous"
    )
    target_node = next(
        n for n in problem.variables if n != continuous_node
    )
    ev_row = {continuous_node: torch.tensor(0.0)}  # explicitly cpu

    out = filter_ground_truth(
        problem, ev_row, target_node, eps_factor=0.5, n_eff_min=10,
    )
    assert out is None or isinstance(out, torch.Tensor)
