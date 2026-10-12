"""Port of tests/integration/test_inference_accuracy_oracle.py to v0.13.

Tests the v0.13 oracle helpers in nbn/bench/core/oracle.py.  The
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

from nbn.bench.core.oracle import (
    conditional_posterior_samples,
    filter_ground_truth,
)
from nbn.bench.problems import SyntheticConfig, SyntheticProblemSource


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
# Test 1 — conditional oracle matches analytic LG posterior (upstream evidence)
# ---------------------------------------------------------------------------

def test_forward_clamp_matches_analytic_lg_posterior() -> None:
    """conditional_posterior_samples on continuous_lg must match
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
    samples = conditional_posterior_samples(
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


def _analytic_posterior(problem, target, evidence):
    A, b, d, nodes = _scm_matrices(problem)
    M = torch.linalg.inv(torch.eye(len(A)) - A)
    mu = M @ b
    Sigma = M @ torch.diag(d) @ M.T
    t = nodes.index(target)
    e = [nodes.index(n) for n in evidence]
    ev = torch.tensor([float(torch.as_tensor(v).reshape(-1)[0]) for v in evidence.values()])
    gain = torch.linalg.solve(Sigma[e][:, e], Sigma[e, t])
    mean = mu[t] + gain @ (ev - mu[e])
    var = Sigma[t, t] - Sigma[t, e] @ gain
    return float(mean), float(var.clamp_min(1e-12).sqrt()), float(mu[t]), float(Sigma[t, t].sqrt())


def _chain_ends(problem):
    """(root, descendant) pair with the strongest ancestor→descendant link."""
    import networkx as nx

    g = nx.DiGraph()
    g.add_nodes_from(problem.variables)
    g.add_edges_from(problem.dag)
    A, b, d, nodes = _scm_matrices(problem)
    M = torch.linalg.inv(torch.eye(len(A)) - A)
    Sigma = M @ torch.diag(d) @ M.T
    sd = Sigma.diag().sqrt()
    corr = Sigma / sd[:, None] / sd[None, :]
    best = max(
        ((a, c) for a in nodes for c in nx.descendants(g, a)),
        key=lambda ac: abs(float(corr[nodes.index(ac[0]), nodes.index(ac[1])])),
    )
    return best


def test_conditional_oracle_matches_analytic_on_downstream_evidence() -> None:
    """#288: evidence on a DESCENDANT of the target.  Clamped ancestral
    sampling returns the target's prior here (p(T | do(E)) = p(T)); the
    oracle must return the conditional p(T | E=e)."""
    torch.manual_seed(0)
    problem = _make_problem("continuous_lg", n_nodes=5, seed=0)
    target, child = _chain_ends(problem)
    A, b, d, nodes = _scm_matrices(problem)
    M = torch.linalg.inv(torch.eye(len(A)) - A)
    mu, Sigma = M @ b, M @ torch.diag(d) @ M.T
    c = nodes.index(child)
    # Evidence 2 sd above the child's mean, so the conditional is clearly
    # separated from the clamped (prior) answer.
    evidence = {child: torch.tensor(float(mu[c] + 2.0 * Sigma[c, c].sqrt()))}
    post_mean, post_std, prior_mean, prior_std = _analytic_posterior(problem, target, evidence)
    assert abs(post_mean - prior_mean) > 0.5 * prior_std, "test needs a real shift"

    samples = conditional_posterior_samples(problem, [target], evidence, n_samples=10_000)
    assert samples is not None and samples.shape == (10_000, 1)
    m, s = float(samples.mean()), float(samples.std())
    assert abs(m - post_mean) < 0.05 * post_std + 0.02, (m, post_mean, prior_mean)
    assert abs(s - post_std) < 0.10 * post_std, (s, post_std, prior_std)


def test_conditional_oracle_returns_none_when_ess_collapses() -> None:
    """Evidence far in the tail of a strong descendant: too few effective
    particles within the cap → None (cell absent), never a biased answer."""
    torch.manual_seed(0)
    problem = _make_problem("continuous_lg", n_nodes=5, seed=0)
    target, child = _chain_ends(problem)
    out = conditional_posterior_samples(
        problem, [target], {child: torch.tensor(1e4)}, n_samples=500,
        max_particles=1000, min_ess=100,
    )
    assert out is None


def test_conditional_oracle_constant_weights_skip_resampling() -> None:
    """Evidence only on a root: weights are constant, so the particles are
    returned as drawn (no multinomial duplicates) — the old exact case."""
    torch.manual_seed(0)
    problem = _make_problem("continuous_lg", n_nodes=5, seed=0)
    parents = {n: [a for a, b in problem.dag if b == n] for n in problem.variables}
    root = next(n for n in problem.variables
                if not parents[n] and any(a == n for a, _ in problem.dag))
    child = next(b for a, b in problem.dag if a == root)
    out = conditional_posterior_samples(
        problem, [child], {root: torch.tensor(0.3)}, n_samples=1000)
    assert out is not None and out.shape == (1000, 1)
    assert torch.unique(out).numel() > 990


def test_bnlearn_weighted_sample_matches_analytic_diagnosis() -> None:
    """bnlearn Gaussian true model: X -> Y, Y observed, target X."""
    from nbn.bench.domains.base import BenchmarkProblem, GroundTruth
    from nbn.bench.problems.bnlearn import _BnlearnContinuousModel

    data = {
        "kind": "gaussian", "nodes": ["X", "Y"], "edges": [["X", "Y"]],
        "cpds": [
            {"name": "X", "type": "gaussian", "intercept": 1.0, "coefficients": {}, "sd": 2.0},
            {"name": "Y", "type": "gaussian", "intercept": 0.5,
             "coefficients": {"X": 1.5}, "sd": 1.0},
        ],
    }
    variables = {"X": ("continuous", None), "Y": ("continuous", None)}
    tm = _BnlearnContinuousModel(data, variables)
    problem = BenchmarkProblem(
        name="xy", dag=[("X", "Y")], variables=variables, train_data=None,
        test_data=None, queries=[], ground_truth=GroundTruth(samples=None),
        true_model=tm, family="continuous_gauss", problem_id="xy", seed=0,
    )
    # Var X = 4, Var Y = 1.5^2*4 + 1 = 10, Cov = 6; E[Y] = 0.5 + 1.5 = 2.
    y = 8.0
    post_mean = 1.0 + 6.0 / 10.0 * (y - 2.0)
    post_std = math.sqrt(4.0 - 36.0 / 10.0)
    samples = conditional_posterior_samples(problem, ["X"], {"Y": torch.tensor(y)},
                                            n_samples=10_000)
    assert samples is not None
    assert abs(float(samples.mean()) - post_mean) < 0.05
    assert abs(float(samples.std()) - post_std) < 0.05 * post_std + 0.02


# ---------------------------------------------------------------------------
# Test 2 — conditional oracle matches tight rejection at marginal mean
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

    fwc_samples = conditional_posterior_samples(
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
# Test 3 — conditional oracle handles degenerate (scalar) evidence
# ---------------------------------------------------------------------------

def test_forward_clamp_handles_degenerate_evidence_dim() -> None:
    """Scalar evidence values are accepted; output shape is (n_samples, 1)."""
    torch.manual_seed(0)
    problem = _make_problem("continuous_lg", n_nodes=4, seed=0)

    nodes = list(problem.variables.keys())
    samples = conditional_posterior_samples(
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
