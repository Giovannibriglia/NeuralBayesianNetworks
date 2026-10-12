"""Ground-truth oracle helpers for v0.13 accuracy measurements.

Extracted from ``nbn/bench/crash_test_runner.py`` (v0.12 private helpers
``_filter_ground_truth`` and ``_conditional_posterior_samples``),
adapted to work against ``BenchmarkProblem`` instead of ``SyntheticBN``.

These helpers are *core* capabilities shared by ``AccuracyAndTiming`` and
potentially by audit scripts, diagnostics, and future measurements.  Placing
them in ``nbn/bench/core/`` rather than inside the measurement package
ensures they remain accessible without importing the full measurement stack.
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


def _lw_particles(
    tm,
    targets: list[str],
    ev: dict,
    *,
    target_ess: float,
    cap: int,
    first_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    """Draw LW particles until the ESS reaches ``target_ess`` or ``cap``.

    Returns ``(target columns [N, T], log-weights [N], ess)`` or ``None`` when
    the model cannot score the evidence / sampling raised.
    """
    with torch.no_grad():
        try:
            draw = _weighted_sampler(tm, list(targets), ev)
            if draw is None:
                return None
            cols, logws = [], []
            drawn, chunk = 0, int(first_chunk)
            while True:
                c, lw = draw(min(chunk, _ORACLE_MAX_CHUNK))
                cols.append(c)
                logws.append(lw)
                drawn += c.shape[0]
                logw = torch.cat(logws)
                w = torch.softmax(logw, dim=0)
                ess = float(1.0 / w.pow(2).sum())
                if ess >= target_ess or drawn >= cap:
                    break
                # Grow towards the particle count the current ESS rate implies.
                need = int(drawn * (target_ess / max(ess, 1.0) - 1.0)) + 1
                chunk = max(1, min(cap - drawn, max(chunk, need)))
        except Exception:
            return None
    return torch.cat(cols), logw, ess


def conditional_posterior_samples(
    problem: BenchmarkProblem,
    targets: list[str],
    evidence: dict[str, torch.Tensor],
    *,
    n_samples: int = 2000,
    max_particles: int | None = None,
    min_ess: float = _ORACLE_MIN_ESS,
    seed: int | None = None,
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
    seed:
        When given, every draw (particles and resampling) runs under
        :func:`nbn.bench.core.seeding.rng_scope` with this seed: the result is
        reproducible and the caller's RNG streams are left untouched (D6).

    Returns
    -------
    Tensor ``[n_samples, len(targets)]`` on CPU, float32, or ``None`` when the
    true model is unavailable / unsupported, sampling raised, or the ESS stayed
    below ``min_ess``.
    """
    if seed is not None:
        from nbn.bench.core.seeding import rng_scope

        with rng_scope(seed, _model_device(problem.true_model)):
            return conditional_posterior_samples(
                problem, targets, evidence, n_samples=n_samples,
                max_particles=max_particles, min_ess=min_ess, seed=None,
            )
    tm = problem.true_model
    if tm is None:
        return None

    ev = _clean_evidence(evidence)
    cap = int(max_particles or _ORACLE_MAX_PARTICLE_FACTOR * n_samples)
    res = _lw_particles(tm, list(targets), ev, target_ess=n_samples, cap=cap,
                        first_chunk=n_samples)
    if res is None:
        return None
    x, logw, ess = res
    if float(logw.max() - logw.min()) < 1e-9:
        return x[:n_samples]
    if ess < min_ess:
        return None
    w = torch.softmax(logw, dim=0)
    idx = torch.multinomial(w, n_samples, replacement=True)
    return x[idx]


def _model_device(tm) -> str:
    """Device the true model samples on (its torch device; CPU for pgmpy /
    numpy models). ``rng_scope`` seeds and restores only that device."""
    return str(getattr(tm, "device", "cpu") or "cpu")


def _clean_evidence(evidence: dict) -> dict:
    return {
        k: (v.reshape(1) if isinstance(v, torch.Tensor) and v.dim() == 0 else v)
        for k, v in evidence.items()
        if v is not None
    }


# ---------------------------------------------------------------------------
# Oracle 3: discrete targets — exact VE, LW, legacy rejection fallback (D1)
# ---------------------------------------------------------------------------

#: ``CellResult.oracle`` values, one per discrete-oracle tier.
ORACLE_EXACT = "exact_ve"
ORACLE_LW = "lw"
ORACLE_REJECTION = "rejection"

# Particle budget of the discrete LW tier when the problem carries no
# reference pool to size it from.
_DEFAULT_DISCRETE_LW_PARTICLES = 50_000
# Particle cap of the discrete LW tier, as a multiple of the ESS target. Lower
# than the continuous tier's factor because the ESS target is ~25x larger.
_DISCRETE_LW_MAX_PARTICLE_FACTOR = 10


def discrete_posterior_lw(
    problem: BenchmarkProblem,
    target: str,
    evidence: dict,
    *,
    n_particles: int,
    min_ess: float = _ORACLE_MIN_ESS,
) -> torch.Tensor | None:
    """``P(target | e)`` as the self-normalised LW histogram of the target
    over particles from ``problem.true_model`` (no resampling noise).

    Particles are drawn until the ESS reaches ``n_particles`` or ``10 ×
    n_particles`` were drawn. ``None`` when the model cannot score the
    evidence or the ESS stayed below ``min_ess``. Uses the caller's RNG (wrap
    in ``rng_scope`` for determinism).
    """
    tm = problem.true_model
    if tm is None:
        return None
    k = int(problem.variables[target][1])
    ev = _clean_evidence(evidence)
    if target in ev:
        out = torch.zeros(k, dtype=torch.float64)
        out[int(torch.as_tensor(ev[target]).reshape(-1)[0].item())] = 1.0
        return out
    res = _lw_particles(
        tm, [target], ev, target_ess=n_particles,
        cap=_DISCRETE_LW_MAX_PARTICLE_FACTOR * int(n_particles),
        first_chunk=min(int(n_particles), _ORACLE_MAX_CHUNK),
    )
    if res is None:
        return None
    x, logw, ess = res
    if ess < min_ess:
        return None
    w = torch.softmax(logw.double(), dim=0)
    idx = x.reshape(-1).round().long().clamp(0, k - 1)
    probs = torch.zeros(k, dtype=torch.float64).scatter_add_(0, idx, w)
    return probs / probs.sum()


def discrete_oracle(
    problem: BenchmarkProblem,
    target: str,
    evidence: dict,
    *,
    n_particles: int | None = None,
    seed: int = 0,
    max_entries: int | None = None,
    eps_factor: float = 0.50,
    n_eff_min: int = 10,
    source: object | None = None,
) -> tuple[torch.Tensor | None, str | None]:
    """Reference ``P(target | e)`` for a discrete target, and how it was made.

    Tiers, first that applies wins:

    1. ``"exact_ve"`` — exact variable elimination on the true model's CPTs
       (:mod:`nbn.bench.core.exact_discrete`) when the ancestral closure of
       ``{target} ∪ E`` is all-discrete and the elimination fits the cap.
    2. ``"lw"`` — likelihood weighting on the true model (hybrid networks with
       continuous ancestors/evidence, or exact VE intractable); the target's
       weighted histogram over ``n_particles`` (default: the problem's
       ``n_reference``, i.e. the reference-pool size) effective particles.
    3. ``"rejection"`` — the legacy exact-match rejection filter over
       ``problem.ground_truth.samples``; only when the true model is absent or
       unusable. Its Monte-Carlo error is large; rows scored this way are
       flagged so they can be excluded.

    Returns ``(probs float64 [K] or None, tier or None)``. All sampling runs
    under ``rng_scope(seed)``: identical for every baseline of a cell.
    """
    from nbn.bench.core import exact_discrete as xd
    from nbn.bench.core.seeding import rng_scope

    ev = _clean_evidence(evidence)
    src = source if source is not None else xd.tabular_source(problem)
    if src is not None:
        try:
            ev_int = {
                k: int(round(float(torch.as_tensor(v).reshape(-1)[0].item())))
                for k, v in ev.items()
            }
            return xd.exact_posterior(
                src, target, ev_int,
                max_entries=max_entries or xd.DEFAULT_MAX_ENTRIES,
            ), ORACLE_EXACT
        except (xd.Intractable, KeyError):
            pass
        except xd.ZeroProbabilityEvidence:
            # No conditional exists. LW would return None for the same
            # reason, so stop here (the measurement emits no_result).
            return None, ORACLE_EXACT
        # xd.InvalidEvidence (a state outside 0..K-1) propagates: it is a
        # selector / data bug, recorded as an error row with its message.

    if problem.true_model is not None:
        if n_particles is None:
            gt = problem.ground_truth
            pool = getattr(gt, "samples", None) if gt is not None else None
            n_particles = (int(pool.shape[0]) if pool is not None and pool.numel()
                           else _DEFAULT_DISCRETE_LW_PARTICLES)
        with rng_scope(seed, _model_device(problem.true_model)):
            probs = discrete_posterior_lw(problem, target, ev, n_particles=n_particles)
        if probs is not None:
            return probs, ORACLE_LW
        if _weighted_sampler_available(problem.true_model, target, ev):
            # The model could score the evidence but the ESS collapsed: an
            # honest "no reference", not a reason to fall back to rejection.
            return None, ORACLE_LW

    samples = filter_ground_truth(
        problem, evidence, target, eps_factor=eps_factor, n_eff_min=n_eff_min,
    )
    if samples is None:
        return None, ORACLE_REJECTION
    k = int(problem.variables[target][1])
    counts = torch.zeros(k, dtype=torch.float64).scatter_add_(
        0, samples.long().clamp(0, k - 1),
        torch.ones(samples.shape[0], dtype=torch.float64),
    )
    return counts / counts.sum(), ORACLE_REJECTION


def _weighted_sampler_available(tm, target: str, ev: dict) -> bool:
    try:
        return _weighted_sampler(tm, [target], ev) is not None
    except Exception:
        return False


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
    if hasattr(tm, "get_cpds") and hasattr(tm, "nodes"):
        return _pgmpy_sampler(tm, targets, ev)
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


def _pgmpy_sampler(model, targets: list[str], ev: dict):
    """LW sampler over a pgmpy ``DiscreteBayesianNetwork``'s ``TabularCPD``s.

    States are integer indices in ``state_names`` order (the encoding of
    ``BnlearnProblemSource``). Draws use the global torch RNG, so callers
    seed it (``rng_scope``) for reproducibility.
    """
    import numpy as np

    parents_of = {nd: list(model.get_cpds(nd).variables[1:]) for nd in model.nodes()}
    need = _ancestral_closure(lambda nd: parents_of[nd], list(targets) + list(ev))
    order: list[str] = []
    done: set[str] = set()

    def visit(nd: str) -> None:
        if nd in done:
            return
        for p in parents_of[nd]:
            visit(p)
        done.add(nd)
        order.append(nd)

    for nd in sorted(need):
        visit(nd)
    tables = {}
    for nd in order:
        vals = np.asarray(model.get_cpds(nd).values, dtype=np.float64)
        t = torch.from_numpy(np.moveaxis(vals, 0, -1).copy())
        tables[nd] = t / t.sum(dim=-1, keepdim=True).clamp_min(1e-300)

    def draw(n: int):
        out: dict[str, torch.Tensor] = {}
        logw = torch.zeros(n, dtype=torch.float64)
        for nd in order:
            t = tables[nd]
            pa = parents_of[nd]
            probs = t[tuple(out[p] for p in pa)] if pa else t.expand(n, -1)
            if nd in ev:
                v = int(round(float(torch.as_tensor(ev[nd]).reshape(-1)[0].item())))
                out[nd] = torch.full((n,), v, dtype=torch.long)
                logw = logw + torch.log(probs[:, v].clamp_min(1e-300))
            else:
                out[nd] = torch.multinomial(probs, 1).reshape(n)
        cols = torch.stack([out[t].float() for t in targets], dim=-1)
        return cols, logw.float()

    return draw
