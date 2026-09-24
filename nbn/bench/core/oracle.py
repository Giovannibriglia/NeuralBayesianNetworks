"""Ground-truth oracle helpers for v0.13 accuracy measurements.

Extracted from ``nbn/bench/crash_test_runner.py`` (v0.12 private helpers
``_filter_ground_truth`` and ``_conditional_posterior_samples``),
adapted to work against ``BenchmarkProblem`` instead of ``SyntheticBN``.

These helpers are *core* capabilities shared by ``AccuracyAndTiming`` and
potentially by audit scripts, diagnostics, and future measurements.  Placing
them in ``nbn/bench/core/`` rather than inside the measurement package
ensures they remain accessible without importing the full measurement stack.

Reference: docs/v0.13-benchmark-redesign.md §4.1, §5.2
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from nbn.bench.domains.base import BenchmarkProblem


# ---------------------------------------------------------------------------
# Column-order helper (used by both oracle functions)
# ---------------------------------------------------------------------------

def _column_order(problem: BenchmarkProblem) -> tuple[str, ...]:
    """Return the topological-sort column order for ``problem``.

    Mirrors the ``SyntheticBN.column_order`` convention established in
    ``make_synthetic_bn``: columns in ``ground_truth.samples`` are written
    in this order, so any column-indexed access must use this same order.

    Implementation notes
    --------------------
    * ``problem.dag`` may be a ``list[tuple[str, str]]`` (edges only) **or**
      an ``nx.DiGraph`` (the test-helper convention).  Both are handled by
      ``add_edges_from``.
    * Isolated nodes (no edges) are only present in ``problem.variables``,
      *not* in the edge list.  ``add_nodes_from(problem.variables)`` ensures
      they are included before ``topological_sort`` is called.  Omitting
      this step would give a different (shorter) sort than the one used when
      ``ground_truth.samples`` was written.
    * ``nx.topological_sort`` on a fixed graph is deterministic (DFS-based),
      so the result is stable across calls given the same graph.

    Complexity: O(|V| + |E|), negligible for benchmark-scale DAGs.
    """
    import networkx as nx

    dag_nx = nx.DiGraph()
    dag_nx.add_nodes_from(problem.variables)
    dag_nx.add_edges_from(problem.dag)
    return tuple(nx.topological_sort(dag_nx))


# ---------------------------------------------------------------------------
# Oracle 1: discrete — exact-match rejection filter
# ---------------------------------------------------------------------------

def filter_ground_truth(
    problem: BenchmarkProblem,
    evidence_row: dict[str, torch.Tensor],
    target: str,
    *,
    eps_factor: float = 0.50,
    n_eff_min: int = 10,
) -> torch.Tensor | None:
    """Exact-match rejection filter on ``problem.ground_truth.samples``.

    Used for **discrete** targets.  Filters the joint reference-sample pool
    to rows matching the evidence, then returns the target column of the
    surviving rows.

    Adapted from ``crash_test_runner._filter_ground_truth`` (v0.12).
    The main structural change: accepts ``BenchmarkProblem`` instead of
    ``SyntheticBN``; column index is looked up via ``_column_order(problem)``
    rather than ``bn.column_index(name)``.

    Parameters
    ----------
    problem:
        A ``BenchmarkProblem`` whose ``ground_truth.samples`` holds the
        reference pool in ``_column_order(problem)`` column order.
    evidence_row:
        A ``{node_name: scalar_tensor}`` dict for one query row.
    target:
        The target node name.
    eps_factor:
        For continuous evidence nodes: accept reference rows within
        ``eps_factor * std(col)`` of the evidence value.
    n_eff_min:
        Minimum surviving rows required; return ``None`` if fewer survive.

    Returns
    -------
    Filtered target-column samples, shape ``[n_eff]``, on CPU, float32.
    ``None`` if the ground-truth pool is absent or too few rows survive.
    """
    if problem.ground_truth is None or problem.ground_truth.samples is None:
        return None
    samples = problem.ground_truth.samples
    if samples.numel() == 0:
        return None

    col_order = _column_order(problem)
    col_idx = {name: i for i, name in enumerate(col_order)}

    # v0.5c device-safety fix (preserved): mask must live on the same device
    # as samples.  Evidence values may arrive on CPU while samples are on GPU
    # (hybrid/cuda smoke runs).
    mask = torch.ones(samples.shape[0], dtype=torch.bool, device=samples.device)

    for node, val in evidence_row.items():
        if val is None:
            continue  # empty-mode evidence: marginalize, don't filter (Phase 3)
        if node not in col_idx:
            continue  # node not in column map — skip safely
        idx = col_idx[node]
        col = samples[:, idx]
        kind = problem.variables[node][0]
        raw = val.item() if isinstance(val, torch.Tensor) else val
        if kind == "discrete":
            mask &= (col.long() == int(raw))
        else:
            v = torch.as_tensor(raw, device=col.device, dtype=col.dtype)
            sigma = col.std().clamp_min(1e-3)
            mask &= (col - v).abs() < eps_factor * sigma

    if int(mask.sum().item()) < n_eff_min:
        return None

    target_idx = col_idx[target]
    return samples[mask, target_idx].cpu().float()


# ---------------------------------------------------------------------------
# Oracle 2: continuous/hybrid — likelihood weighting on the true model
# ---------------------------------------------------------------------------

# Resampled ESS below this → the oracle cannot represent p(T | e) and returns
# None (cell absent, not zero), the continuous analogue of ``n_eff_min``.
_ORACLE_MIN_ESS = 100.0
# Particle cap as a multiple of ``n_samples``: particles are drawn in chunks
# until the ESS reaches ``n_samples`` or this cap is hit.
_ORACLE_MAX_PARTICLE_FACTOR = 50
# Rows per sampler call — bounds peak memory on large networks.
_ORACLE_MAX_CHUNK = 20_000


def conditional_posterior_samples(
    problem: BenchmarkProblem,
    targets: list[str],
    evidence: dict[str, torch.Tensor],
    *,
    n_samples: int = 2000,
    max_particles: int | None = None,
    min_ess: float = _ORACLE_MIN_ESS,
) -> torch.Tensor | None:
    """Samples from ``p(T | E=e)`` under ``problem.true_model``.

    Used for **continuous** and **hybrid-continuous-target** queries.

    Likelihood weighting on the ground-truth parameters: particles are drawn
    by ancestral sampling with the evidence clamped (the proposal) and each is
    weighted by ``Π_{j∈E} p(e_j | pa_j)``; ``n_samples`` draws are then
    resampled by weight.  Clamping alone samples ``p(T | do(E=e))``, which
    differs from the conditional whenever an evidence node has an unobserved
    ancestor d-connected to T — e.g. evidence downstream of the target (#288).
    When the weights are constant (no such ancestor) the first ``n_samples``
    particles are returned as is: exact, at the old clamped-sampling cost.

    Only the ancestral closure of ``targets ∪ E`` is sampled — no other node
    can affect the posterior — and particles are drawn until the ESS reaches
    ``n_samples`` or ``max_particles`` (default ``50 · n_samples``).

    Parameters
    ----------
    problem:
        A ``BenchmarkProblem`` with ``true_model`` populated: a
        ``NeuralBayesianNetwork`` (synthetic) or a model exposing
        ``weighted_sample(n, evidence, keep)`` (bnlearn Gaussian / CLG).
    targets:
        Target node names.
    evidence:
        ``{node: scalar_tensor_or_value}`` for one query row; ``None`` values
        (Phase 3 empty mode) are marginalized.
    n_samples:
        Posterior samples returned.

    Returns
    -------
    Tensor ``[n_samples, len(targets)]`` on CPU, float32, or ``None`` when the
    true model is unavailable / unsupported, sampling raised, or the ESS stayed
    below ``min_ess``.
    """
    tm = problem.true_model
    if tm is None:
        return None

    ev = {
        k: (v.reshape(1) if isinstance(v, torch.Tensor) and v.dim() == 0 else v)
        for k, v in evidence.items()
        if v is not None
    }
    cap = int(max_particles or _ORACLE_MAX_PARTICLE_FACTOR * n_samples)
    with torch.no_grad():
        try:
            draw = _weighted_sampler(tm, list(targets), ev)
            if draw is None:
                return None
            cols, logws = [], []
            drawn, chunk = 0, int(n_samples)
            while True:
                c, lw = draw(min(chunk, _ORACLE_MAX_CHUNK))
                cols.append(c)
                logws.append(lw)
                drawn += c.shape[0]
                logw = torch.cat(logws)
                w = torch.softmax(logw, dim=0)
                ess = float(1.0 / w.pow(2).sum())
                if ess >= n_samples or drawn >= cap:
                    break
                # Grow towards the particle count the current ESS rate implies.
                need = int(drawn * (n_samples / max(ess, 1.0) - 1.0)) + 1
                chunk = max(1, min(cap - drawn, max(chunk, need)))
        except Exception:
            return None

    x = torch.cat(cols)
    if float(logw.max() - logw.min()) < 1e-9:
        return x[:n_samples]
    if ess < min_ess:
        return None
    idx = torch.multinomial(w, n_samples, replacement=True)
    return x[idx]


def _ancestral_closure(parents_of, nodes) -> set[str]:
    need, stack = set(), list(nodes)
    while stack:
        nd = stack.pop()
        if nd in need:
            continue
        need.add(nd)
        stack.extend(parents_of(nd))
    return need


def _weighted_sampler(tm, targets: list[str], ev: dict):
    """``draw(n) -> (target columns [n, T] float32 CPU, log-weights [n])``.

    ``None`` when evidence is present but the model cannot score it (unknown
    true-model type) — returning clamped samples there would be the #288 bias.
    """
    from nbn.core.network import NeuralBayesianNetwork

    if isinstance(tm, NeuralBayesianNetwork):
        return _nbn_sampler(tm, targets, ev)
    if hasattr(tm, "weighted_sample"):
        return lambda n: tm.weighted_sample(n, ev, targets)
    if ev:
        return None

    def prior(n):
        s = tm.sample(n=n, evidence={})
        return _target_columns(s, targets, n), torch.zeros(n)
    return prior


def _target_columns(samples: dict, targets: list[str], n: int) -> torch.Tensor:
    cols = []
    for t in targets:
        col = samples[t]
        if col.dim() >= 2 and col.shape[-1] == 1:
            col = col.squeeze(-1)
        cols.append(col.float().cpu().reshape(n, -1))
    return torch.cat(cols, dim=-1)


def _nbn_sampler(model, targets: list[str], ev: dict):
    """Pruned ancestral sampler with LW weights for a ``NeuralBayesianNetwork``.

    Mirrors ``nbn.sampling.ancestral.ancestral_sample`` node by node, over the
    ancestral closure of ``targets ∪ E`` only.
    """
    dag = model.dag
    need = _ancestral_closure(dag.parents, list(targets) + list(ev))
    order = [nd for nd in dag.topological_order() if nd in need]
    dev = model.device

    def draw(n: int):
        out: dict[str, torch.Tensor] = {}
        logw = torch.zeros(n, device=dev)
        for node in order:
            parents = dag.parents(node)
            pa = (torch.cat([out[p].reshape(n, -1) for p in parents], dim=-1)
                  if parents else None)
            mech = model.mechanisms[node]
            if node in ev:
                val = torch.as_tensor(ev[node]).to(dev).reshape(1, -1).expand(n, -1)
                out[node] = val
                logw = logw + mech.log_prob(val, pa).reshape(n)
            elif pa is None:
                out[node] = mech.sample(None, n=n).squeeze(0).reshape(n, -1)
            else:
                out[node] = mech.sample(pa, n=1).squeeze(1).reshape(n, -1)
        return _target_columns(out, targets, n), logw.float().cpu()

    return draw
