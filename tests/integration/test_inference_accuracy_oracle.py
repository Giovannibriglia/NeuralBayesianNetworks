"""PR-B round-4 regression: inference accuracy oracle.

The v0.5b round-3 finding showed that ε-ball rejection of
``bn.ground_truth_samples`` was a biased oracle on multi-evidence
non-zero-mean continuous queries.  Forward-with-clamp ancestral
sampling from the true SCM is exact for the synthetic generator's
mechanism families, so it replaced the ε-ball oracle in round-4.

The same investigation also surfaced a column-ordering schema bug:
``benchmarking/synthetic.make_synthetic_bn`` writes
``bn.ground_truth_samples`` with columns in *topological-sort* order,
but the runner was indexing it via ``list(bn.dag.nodes()).index(target)``
(insertion order).  These two orders differ for non-trivial DAGs;
the round-2/3 W₁≈3 mystery was almost entirely the resulting column
shuffle, not the kernel-bandwidth bias we hypothesised.

Two regression tests pin the new oracle:

1. **Test 1 — analytic LG agreement.**  On a small ``continuous_lg``
   network, forward-with-clamp must match the closed-form Gaussian
   posterior to within 5 % of the analytic posterior std.

2. **Test 2 — non-Gaussian sanity.**  On ``continuous_nongauss`` with
   evidence at the marginal mean, forward-with-clamp and tight
   rejection must agree within rejection-sampling noise — provided
   both index ``bn.ground_truth_samples`` via topological-sort
   order (the corrected convention).
"""
from __future__ import annotations

import math

import networkx as nx
import torch

from benchmarking.crash_test_runner import _forward_with_clamp_posterior_samples
from benchmarking.synthetic import make_synthetic_bn


def _scm_matrices(bn) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract ``A``, ``b``, ``d`` from a true_model whose every node
    is a ``LinearGaussianMechanism``."""
    n = len(list(bn.dag.nodes()))
    nodes = list(bn.dag.nodes())
    A = torch.zeros(n, n)
    b = torch.zeros(n)
    d = torch.zeros(n)
    for node in nodes:
        j = nodes.index(node)
        parents = list(bn.dag.predecessors(node))
        mech = bn.true_model.mechanisms[node]
        b[j] = mech._bias.item()
        d[j] = torch.exp(mech._log_scale).item() ** 2
        for k, p in enumerate(parents):
            A[j, nodes.index(p)] = mech._weight[k].item()
    return A, b, d


def test_forward_clamp_matches_analytic_lg_posterior() -> None:
    """Forward-with-clamp on a 5-node LG must match the closed-form
    Gaussian posterior to within 5 % of analytic std."""
    torch.manual_seed(0)
    bn = make_synthetic_bn(
        family="continuous_lg", n_nodes=5, edge_density=0.40,
        max_in_degree=2, n_train=500, n_test=100, n_reference=200, seed=0,
        device="cpu",
    )
    nodes = list(bn.dag.nodes())
    target = nodes[-1]
    # Pick two evidence nodes with values away from zero so this test
    # would fail under the round-3 ε-ball oracle bias.
    ev_nodes = nodes[:2]
    evidence = {ev_nodes[0]: torch.tensor([1.5]),
                ev_nodes[1]: torch.tensor([-1.2])}

    # Closed-form posterior
    A, b, d = _scm_matrices(bn)
    M = torch.linalg.inv(torch.eye(len(A)) - A)
    mu = M @ b
    Sigma = M @ torch.diag(d) @ M.T
    t_idx = nodes.index(target)
    e_idx = [nodes.index(n) for n in ev_nodes]
    e_vals = torch.tensor([float(evidence[n].item()) for n in ev_nodes])
    post_mean = mu[t_idx] + Sigma[t_idx, e_idx] @ torch.linalg.solve(
        Sigma[e_idx][:, e_idx], e_vals - mu[e_idx],
    )
    post_var = Sigma[t_idx, t_idx] - Sigma[t_idx, e_idx] @ torch.linalg.solve(
        Sigma[e_idx][:, e_idx], Sigma[e_idx, t_idx],
    )
    post_std = post_var.clamp_min(1e-12).sqrt()

    # Forward-with-clamp posterior
    samples = _forward_with_clamp_posterior_samples(
        bn, [target], evidence, n_samples=10_000,
    )
    assert samples is not None and samples.shape[0] == 10_000
    fwc_mean = float(samples.mean().item())
    fwc_std = float(samples.std().item())

    # Within 5 % of analytic posterior std for mean; 10 % for std.
    assert abs(fwc_mean - float(post_mean)) < 0.05 * float(post_std), (
        f"forward-with-clamp mean {fwc_mean:.4f} differs from analytic "
        f"{float(post_mean):.4f} by more than 5 % of analytic std {float(post_std):.4f}"
    )
    assert abs(fwc_std - float(post_std)) < 0.10 * float(post_std), (
        f"forward-with-clamp std {fwc_std:.4f} differs from analytic "
        f"{float(post_std):.4f} by more than 10 %"
    )


def test_forward_clamp_matches_tight_rejection_at_marginal_mean() -> None:
    """On continuous_nongauss with evidence at the marginal mean, the two
    oracles should agree within rejection-sampling noise.  This is the
    one configuration where ε-ball rejection isn't biased — both
    estimators should land on the same answer."""
    torch.manual_seed(0)
    bn = make_synthetic_bn(
        family="continuous_nongauss", n_nodes=5, edge_density=0.40,
        max_in_degree=2, n_train=500, n_test=100, n_reference=200_000,
        seed=0, device="cpu",
    )
    # PR-B round-4: bn.ground_truth_samples columns are in
    # nx.topological_sort order, not bn.dag.nodes() insertion order.
    topo_nodes = list(nx.topological_sort(bn.dag))
    target = topo_nodes[-1]
    ev_node = topo_nodes[0]
    ev_idx = topo_nodes.index(ev_node)
    target_idx = topo_nodes.index(target)

    # Evidence at the marginal mean of the evidence node.
    ev_marginal = bn.ground_truth_samples[:, ev_idx]
    ev_mean = float(ev_marginal.mean().item())
    ev_std = float(ev_marginal.std().item())
    evidence = {ev_node: torch.tensor([ev_mean])}

    # Tight rejection (unbiased at evidence = marginal mean).
    eps = 0.05 * ev_std
    mask = (ev_marginal - ev_mean).abs() < eps
    n_eff = int(mask.sum().item())
    if n_eff < 100:
        # Wider band as a fallback if the synthetic generator's tails
        # didn't produce enough samples at the marginal mean.
        eps = 0.20 * ev_std
        mask = (ev_marginal - ev_mean).abs() < eps
        n_eff = int(mask.sum().item())
    rejection_target = bn.ground_truth_samples[mask, target_idx]
    rejection_mean = float(rejection_target.mean().item())
    rejection_std = float(rejection_target.std().item())
    se = rejection_std / math.sqrt(max(1, n_eff))

    # Forward-with-clamp posterior
    samples = _forward_with_clamp_posterior_samples(
        bn, [target], evidence, n_samples=10_000,
    )
    assert samples is not None
    fwc_mean = float(samples.mean().item())

    # Should agree within 3 standard errors of the rejection estimate
    # plus the same 3-SE tolerance for the forward-clamp side
    # (n=10_000 → SE ≈ rejection_std / 100).
    fwc_se = rejection_std / math.sqrt(10_000)
    tol = 3 * (se + fwc_se)
    assert abs(fwc_mean - rejection_mean) < tol, (
        f"forward-with-clamp {fwc_mean:.4f} differs from tight-rejection "
        f"{rejection_mean:.4f} (n_eff={n_eff}) by more than 3·SE={tol:.4f}"
    )


def test_forward_clamp_handles_degenerate_evidence_dim() -> None:
    """Small smoke check: scalar (0-D) evidence values are accepted."""
    torch.manual_seed(0)
    bn = make_synthetic_bn(
        family="continuous_lg", n_nodes=4, edge_density=0.50,
        max_in_degree=2, n_train=500, n_test=100, n_reference=200, seed=0,
        device="cpu",
    )
    nodes = list(bn.dag.nodes())
    samples = _forward_with_clamp_posterior_samples(
        bn, [nodes[-1]], {nodes[0]: torch.tensor(0.5)}, n_samples=200,
    )
    assert samples is not None and samples.shape == (200, 1)
