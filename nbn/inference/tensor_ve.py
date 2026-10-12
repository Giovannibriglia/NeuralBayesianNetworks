"""Log-domain einsum-based variable elimination for discrete BNs.

The v0.3.1 work introduces:

* A small per-model factor cache so repeated queries don't rebuild log-CPTs.
* A per-(targets, evidence_keys) plan cache so the elimination order is
  determined once.
* A *truly batched* ``query_batch`` that produces ``[B, K]`` from a single
  pass of the elimination procedure, rather than looping in Python over the
  batch dim.  Evidence is applied factor-by-factor via fancy indexing,
  introducing a leading B axis on every factor that touched any evidence
  variable; subsequent factor products and marginalisations broadcast over
  that B axis natively.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Mapping

import torch

from nbn.core.factor import LogFactor
from nbn.inference._cache import (
    cache_get,
    cache_put,
    structure_version,
    tensor_versions,
)
from nbn.inference._elimination_order import get_order
from nbn.inference._memory import (
    OutOfMemoryError,
    device_capacity_bytes,
    device_headroom_bytes,
)
from nbn.inference._ve_memory import (
    _estimate_peak_bytes,  # noqa: F401  (re-exported: plan-level helper)
    _factor_dtype_bytes,
    _max_fitting_rows,
    _PeakModel,
    _plan_cost,
    _PlanCost,
    _Scope,
    embed_tape,
    evidence_prelude,
    tables_tape,
)
from nbn.inference._validation import check_discrete_values
from nbn.inference._pruning import relevant_subnetwork, requisite_nodes
from nbn.inference.base import InferenceEngine
from nbn.utils.device import to_device_dtype

logger = logging.getLogger(__name__)

# The guard's peak-memory model lives in ``nbn.inference._ve_memory`` (#178):
# a tape of every allocation the elimination makes, replayed per device.

# Plans whose estimated peak is below this fraction of the stable memory
# budget skip the live-headroom check that sizes chunks (it reads memory
# statistics, cgroup usage files on Linux): they cannot need chunking.
_HEADROOM_CHECK_FRACTION = 0.01

# MPS reductions are wrong on large tensors (torch 2.14.1, macOS 26, Apple
# M-series).  ``torch.logsumexp`` on MPS silently returns wrong values (inf
# rows, or rows off by ~10) while element-wise ops stay correct, and VE's
# every elimination step is a broadcast sum followed by ``logsumexp`` — so it
# returned plausible wrong posteriors once a batch grew large.  Measured
# (2026-10-12, probe in the R4 review): on a *strided* input — VE's factors
# are permuted views — every element at flat offset >= 2^27 comes back wrong
# ([40, 4^12] wrong from row 8, [160, 4^11] from row 32, [640, 4^10] from row
# 128); contiguous inputs held up to ~2^30 elements but failed below 2^31 for
# some shapes.  On MPS, row-chunks are therefore capped so the largest
# intermediate factor stays below this many elements, and a query whose
# single row is already beyond it is refused (``HybridRouter`` answers it
# with LW).  MPS-only; lift or remove once the torch / Metal reduction is fixed.
_MPS_MAX_ELEMENTS = 2 ** 27


class MPSTensorLimitError(RuntimeError):
    """A VE plan needs an intermediate factor larger than MPS reduces correctly."""


class TensorVariableElimination(InferenceEngine):
    """Log-domain VE for discrete BNs (single-row + truly batched queries)."""

    _transient_caches = ("_factor_cache", "_plan_cache", "_cost_cache")

    def __init__(
        self,
        treewidth_threshold: int = 25,
        memory_budget: float | None = None,
        memory_safety: float = 0.9,
        host_memory_fraction: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        memory_budget:
            Optional hard cap, in bytes, on the estimated elimination peak,
            combined (``min``) with the device default below; with neither
            known no check is made.
        memory_safety:
            CUDA default: this fraction of the live device headroom
            (``nbn.inference._memory.device_headroom_bytes``).
        host_memory_fraction:
            CPU / MPS default: this fraction of the *stable* memory capacity
            (RAM capped by the cgroup limit; Metal's recommended working
            set).  The default 1.0 rejects only plans whose estimated peak
            exceeds the whole pool — plans that could not have completed
            without swapping or being OOM-killed.  ``HybridRouter`` prices
            routes with a smaller fraction of its own.
        """
        self.treewidth_threshold = treewidth_threshold
        self.memory_budget = memory_budget
        self.memory_safety = float(memory_safety)
        self.host_memory_fraction = float(host_memory_fraction)
        # Memoised per-plan memory profile (``_PlanCost``), keyed like
        # ``_plan_cache`` (see ``_plan_cost``).
        self._cost_cache: Dict[tuple, _PlanCost] = {}
        # ``(id(model), factors | None)`` validated by this call's guard and
        # consumed by its ``_extract_factors`` (see there).
        self._validated: tuple | None = None
        # ``id(factors) -> (factors, widest element size)``; one entry.
        self._dtype_memo: dict[int, tuple] = {}
        # Memoised log-CPT factors per model: ``{id(model): entry}``, see
        # ``nbn.inference._cache`` and ``_extract_factors`` for validation.
        self._factor_cache: Dict[int, tuple] = {}
        # Memoised (elimination_order, relevant_set) per
        # (id(model), target, evidence_keys, do_keys, order).
        self._plan_cache: Dict[tuple, tuple] = {}

    def invalidate_cache(self, model=None) -> None:
        if model is None:
            self._factor_cache.clear(); self._plan_cache.clear()
            self._cost_cache.clear()
            return
        self._factor_cache.pop(id(model), None)
        for cache in (self._plan_cache, self._cost_cache):
            for k in list(cache):
                if k[0] == id(model):
                    cache.pop(k, None)

    # ------------------------------------------------------------------ #
    # Cache validity
    # ------------------------------------------------------------------ #
    #
    # Entries hold a weakref to the model (recycled ids miss; a dead
    # model's entries are evicted by the weakref callback) and a version
    # (see ``nbn.inference._cache``).  Plans depend only on the graph and the
    # cardinalities, so they are versioned by ``structure_version``.  Factors
    # also depend on every parameter value and on the device, and the
    # model's ``_cache_version`` counter alone misses in-place edits
    # (``load_state_dict``, optimiser steps, ``no_grad`` writes) and
    # ``model.to(device)`` — which served stale CPTs, or crashed with a
    # device mismatch after ``model.to("mps")`` (A7/B4).  So factor entries
    # additionally record the device and the ``_version`` counter of every
    # parameter and buffer; re-reading those counters costs ~30 us on a
    # 200-node network, negligible next to a query.

    @staticmethod
    def _model_version(model) -> tuple:
        """Version of the structural memos (plans)."""
        return structure_version(model)

    def _cache_get(self, cache: Dict, key, model):
        return cache_get(cache, key, model, self._model_version(model))

    def _cache_put(self, cache: Dict, key, model, payload):
        return cache_put(cache, key, model, self._model_version(model), payload)

    # ------------------------------------------------------------------ #
    # Factor and plan extraction (cached)
    # ------------------------------------------------------------------ #

    def _cached_factors(self, model) -> Dict[str, LogFactor] | None:
        """The model's tabulated factors if the cache holds a valid entry."""
        version = (structure_version(model), str(model.device))
        hit = cache_get(self._factor_cache, id(model), model, version)
        if hit is not None:
            tensors, versions, factors = hit
            if versions is not None and tensor_versions(tensors) == versions:
                return factors
        return None

    def _cached_dtype_bytes(self, factors: Dict[str, LogFactor]) -> int:
        """Bytes per element of the widest cached factor: the products take the
        widest dtype (a ``DeterministicMechanism`` tabulates float32 even in a
        float64 model, #301 review 3).  Memoised per cached factor dict."""
        hit = self._dtype_memo.get(id(factors))
        if hit is None or hit[0] is not factors:
            hit = (factors, max(f.log_values.element_size() for f in factors.values()))
            self._dtype_memo = {id(factors): hit}
        return hit[1]

    def _extract_factors(self, model) -> Dict[str, LogFactor]:
        # The guard of this same call may just have validated the cache
        # (``_peak_model``): reuse that answer instead of re-reading every
        # parameter's version counter (#301 review D).  Single use.
        token, self._validated = self._validated, None
        if token is not None and token[0] == id(model) and token[1] is not None:
            return token[1]
        cached = self._cached_factors(model)
        if cached is not None:
            return cached
        version = (structure_version(model), str(model.device))
        factors: Dict[str, LogFactor] = {}
        for node in model.dag.topological_order():
            mech = model.mechanisms[node]
            if not mech.is_discrete:
                raise ValueError(
                    f"TensorVariableElimination only supports discrete mechanisms; "
                    f"node '{node}' has a continuous mechanism."
                )
            # v0.8-#26: was ``not hasattr(mech, "_logits") or mech._logits
            # is None`` — that check incorrectly reported
            # NeuralCategoricalMechanism as "not fitted" because it has
            # no flat ``_logits`` attribute (the CPD is computed via
            # MLP forward pass per-call).  ``Mechanism.is_fitted``
            # delegates to each mechanism's own definition of
            # fittedness; ``mech.tabulate()`` then materialises the
            # CPD by enumeration when needed.
            if not mech.is_fitted:
                raise RuntimeError(f"Mechanism for node '{node}' has not been fitted.")
            parents = model.dag.parents(node)
            var_scope = parents + [node]
            cards = {n: model.mechanisms[n].n_classes for n in var_scope}
            parent_cards = [model.mechanisms[p].n_classes for p in parents]
            # v0.8-#26: tabulate() returns [*parent_cards, K] (or [K]
            # for root) in logit space; log_softmax along the last
            # axis recovers the conditional log-CPT regardless of
            # whether the mechanism stores logits or log-probs.
            logits = mech.tabulate(parent_cards).detach()
            log_cpt = torch.log_softmax(logits, dim=-1)
            factors[node] = LogFactor(log_cpt, var_scope, cards)
        # Snapshot the counters *after* tabulating, so a mechanism that
        # lazily caches into a buffer does not invalidate its own entry.
        tensors = [*model.parameters(), *model.buffers()]
        cache_put(self._factor_cache, id(model), model, version,
                  (tensors, tensor_versions(tensors), factors))
        return factors

    def _plan(
        self,
        model,
        target: str,
        evidence_keys: tuple[str, ...],
        *,
        do_keys: tuple[str, ...] = (),
        order: str = "min_fill",
    ) -> list[str]:
        """Compute (or look up cached) elimination order.

        Thin wrapper over :meth:`_plan_with_relevant` returning just the
        order, preserving the historical list-valued contract used by
        external callers (the memory-guard test and the
        ``ve_profile_n20`` diagnostic).
        """
        return self._plan_with_relevant(
            model, target, evidence_keys, do_keys=do_keys, order=order,
        )[0]

    def _plan_with_relevant(
        self,
        model,
        target: str,
        evidence_keys: tuple[str, ...],
        *,
        do_keys: tuple[str, ...] = (),
        order: str = "min_fill",
    ) -> tuple[list[str], set[str]]:
        """Compute (or look up cached) elimination plan + relevant set.

        Returns a ``(elimination_order, relevant_set)`` tuple where
        ``relevant_set`` is the Bayes-ball relevant subnetwork for
        ``(target, evidence_keys)`` (Bug 2 of #127) minus the evidence nodes
        whose own CPT is not requisite (#132) — i.e. exactly the nodes
        whose CPTs enter the elimination.  The elimination
        order is computed on the *induced subgraph* of that relevant set,
        not the full DAG, so barren and m-separated nodes never enter the
        ordering.  Callers also use ``relevant_set`` to drop factors for
        non-relevant nodes before elimination (see ``query`` /
        ``query_batch``).

        This is the single source of truth for both the order and the
        relevant set: both are derived once per cached
        ``(id(model), target, evidence_keys, order)`` key, so
        ``relevant_subnetwork`` runs at most once per distinct query.

        Parameters
        ----------
        order:
            Strategy name dispatched through
            :mod:`nbn.inference._elimination_order`.  Default
            ``'min_fill'`` (Kjaerulff 1990 / Koller & Friedman §9.4.3) —
            the v0.6b round-2 fix for the v0.5b §A.5 OOM.  Pass
            ``'topological'`` for the legacy naive ordering kept for
            comparability and as a fallback.

        Notes
        -----
        Cache key includes ``order`` so different strategies cache
        independently; the same ``(model, target, evidence_keys)``
        query under two orders does *not* invalidate either entry.  The
        relevant set is a deterministic function of
        ``(model, target, evidence_keys)`` so it is cached alongside the
        order under the same key.
        """
        key = (id(model), target, evidence_keys, do_keys, order)
        cached = self._cache_get(self._plan_cache, key, model)
        if cached is not None:
            return cached
        # Bug 2 (#127): restrict elimination to the variables m-connected
        # to the target given evidence.  The min-fill order is then
        # computed on the induced subgraph rather than the full DAG —
        # same algorithm, far smaller input on dense networks.
        graph = model.dag.networkx_graph
        # ``target`` is one node, or a tuple of nodes for a joint query (R6):
        # Bayes-ball and min-fill take the target *set* unchanged.
        target_list = [target] if isinstance(target, str) else list(target)
        # Mutilate before Bayes-ball: under do(X), X's incoming edges are cut,
        # so its ancestors are no longer d-connected to anything through X.
        # Running relevance on the observational graph would still be *sound*
        # (it can only over-include, and a summed-out CPT contributes a factor
        # of 1), but the mutilated graph is both correct and cheaper.
        if do_keys:
            graph = graph.copy()
            graph.remove_edges_from(
                [(u, v) for (u, v) in list(graph.in_edges(do_keys))]
            )
        # do-nodes are clamped, exactly like evidence, for the purpose of
        # d-separation on the mutilated graph.
        relevant = relevant_subnetwork(
            graph, target if isinstance(target, str) else set(target_list),
            tuple(evidence_keys) + tuple(do_keys),
        )
        # #132 (Option B): an observed node the ball never enters from a
        # parent (e.g. an observed parent/ancestor of the target) is
        # relevant as *evidence* — its value is clamped wherever it appears
        # in a kept CPT — but its own CPT is not requisite: evaluated at the
        # observed value it depends only on its parents and cancels in the
        # final normalisation.  Dropping it keeps those parents out of the
        # elimination (barley: ``sorttkv``'s CPT dragged in card-67 ``sort``).
        non_requisite = set(evidence_keys) - requisite_nodes(
            graph, set(target_list), tuple(evidence_keys) + tuple(do_keys),
        )
        relevant = relevant - non_requisite
        # Bug 2 Stage 2b (#127): the kept factors are the CPTs of the
        # relevant nodes, whose scopes are {node} ∪ parents(node).  Some
        # of those parents lie *outside* the relevant set — e.g. an
        # evidence node's CPT references its own non-relevant parents.
        # Those parent variables must be eliminated too; otherwise they
        # survive into query()/query_batch()'s final "multiply remaining
        # factors" step and re-form the very joint pruning is meant to
        # avoid (barley aks_m2 hub-given-MB blew up to a 118M-element
        # factor when `sort` (card 67) et al. were left un-eliminated).
        # The elimination scope is therefore the union of the kept
        # factors' scopes = relevant ∪ parents(relevant), and the min-fill
        # order is computed over that induced subgraph — the same ordering
        # unpruned VE would use over exactly these factors.
        factor_scope = set(relevant)
        for node in relevant:
            factor_scope.update(graph.predecessors(node))
        elimination_order = get_order(
            order,
            graph.subgraph(factor_scope),
            targets=target_list,
            # do-nodes are clamped like evidence, so they are conditioned out
            # rather than eliminated — prune them from the moral graph too.
            evidence=list(evidence_keys) + list(do_keys),
        )
        result = (elimination_order, relevant)
        return self._cache_put(self._plan_cache, key, model, result)

    # ------------------------------------------------------------------ #
    # Memory budget (every device)
    # ------------------------------------------------------------------ #

    def memory_budget_bytes(
        self, device, *, host_fraction: float | None = None,
    ) -> float | None:
        """Bytes an elimination on ``device`` may plan to use (``None``: unchecked).

        The guard used to run on CUDA only, reading driver-free memory; CPU
        and MPS allocated blindly, so a terabyte plan was attempted and died
        in the allocator (or the OS OOM killer) instead of being refused or
        routed to LW (B5).

        CUDA compares with live headroom (as the original guard did, now
        counting PyTorch's reusable cache, #264).  CPU and MPS compare with a
        stable capacity × ``host_fraction`` (default
        ``self.host_memory_fraction``): momentary free host memory made the
        same query exact on an idle machine and LW under pressure (R5).
        """
        device = torch.device(device)
        if device.type == "cuda":
            head = device_headroom_bytes(device)
            budget = None if head is None else self.memory_safety * head
        else:
            cap = device_capacity_bytes(device)
            frac = self.host_memory_fraction if host_fraction is None else host_fraction
            budget = None if cap is None else frac * cap
        if self.memory_budget is not None:
            cap = float(self.memory_budget)
            budget = cap if budget is None else min(budget, cap)
        return budget

    def _cached_plan_cost(
        self,
        model,
        target,
        evidence_keys: tuple[str, ...],
        do_keys: tuple[str, ...],
        order: str,
    ) -> _PlanCost:
        """The plan's allocation tape (``_ve_memory._plan_cost``), memoised.

        Computed from the DAG and cardinalities alone — no CPT is tabulated —
        so the router can price a query before committing to VE.  The walk
        sees the factors in the order the engine builds them (topological,
        intervened CPTs dropped), so product chains are priced as they run.
        """
        key = (id(model), target, evidence_keys, do_keys, order)
        hit = self._cache_get(self._cost_cache, key, model)
        if hit is not None:
            return hit
        plan, relevant = self._plan_with_relevant(
            model, target, evidence_keys, do_keys=do_keys, order=order,
        )
        scopes = {}
        for node in model.dag.topological_order():
            if node in relevant and node not in do_keys:
                # (graph surgery: an intervened CPT is dropped)
                var_scope = list(model.dag.parents(node)) + [node]
                scopes[node] = _Scope(
                    var_scope, {n: model.mechanisms[n].n_classes for n in var_scope},
                )
        targets = (target,) if isinstance(target, str) else tuple(target)
        cost = _plan_cost(
            plan, scopes, evidence_keys=tuple(evidence_keys) + tuple(do_keys),
            targets=targets,
            target_cards={t: model.mechanisms[t].n_classes for t in targets},
            joint=not isinstance(target, str),
        )
        return self._cache_put(self._cost_cache, key, model, cost)

    def _peak_profile(
        self,
        model,
        target,
        evidence_keys: tuple[str, ...],
        do_keys: tuple[str, ...],
        order: str,
    ) -> tuple[int, int]:
        """``(batch_free, per_row)`` elements of the plan's largest tensor.

        With ``c`` rows the largest single tensor has
        ``max(batch_free, c * per_row)`` elements — what the MPS element cap
        is checked against.  Memory is priced by ``_peak_model``.
        """
        return self._cached_plan_cost(model, target, evidence_keys, do_keys, order).tensor

    def _peak_model(
        self,
        model,
        target,
        evidence=(),
        do=(),
        *,
        B: int = 1,
        order: str = "min_fill",
        price_evidence: bool = True,
        remember: bool = False,
    ) -> _PeakModel | None:
        """Peak-bytes model of this query (``None``: nothing to eliminate).

        ``target`` is a node name or a sequence of names (joint query).
        ``evidence`` / ``do`` are the call's values (``{node: value}``) —
        what they cost depends on whether the engine must materialise or
        copy them (``_ve_memory.evidence_prelude``) — or bare node names,
        priced as lists of floats, the most expensive input.  On a cold
        factor cache the model also prices tabulating every CPT.  ``query``
        passes ``price_evidence=False``: it reads one Python int per node.
        ``remember`` (``query`` / ``query_batch`` only) hands the cache check
        to the same call's ``_extract_factors``; read-only callers
        (``estimate_peak_bytes``, ``fits_memory``) must not leave it behind,
        or an in-place parameter edit before the next extraction would be
        served stale factors (#301 review 3).
        """
        values = {**_as_values(evidence), **_as_values(do)}
        do_set = set(_as_values(do))
        ev = tuple(sorted(k for k in _as_values(evidence) if k not in do_set))
        do_t = tuple(sorted(do_set))
        plan_target = _plan_target(target, set(ev) | do_set)
        if plan_target is None:
            return None  # every target clamped: a point mass, nothing eliminated
        cost = self._cached_plan_cost(model, plan_target, ev, do_t, order)
        names = (target,) if isinstance(target, str) else tuple(target)
        cards = [model.mechanisms[t].n_classes for t in names]
        clamped = [t in ev or t in do_set for t in names]
        persistent, transient = (evidence_prelude(values, B, model.device)
                                 if price_evidence else ((), ()))
        cached = self._cached_factors(model)
        if remember:
            self._validated = (id(model), cached)
        dtype_bytes = (self._cached_dtype_bytes(cached) if cached
                       else _factor_dtype_bytes(model))
        return _PeakModel(
            cost,
            dtype_bytes=dtype_bytes,
            device_type=torch.device(model.device).type,
            prelude=persistent,
            transient=transient,
            tables=tables_tape(model) if cached is None else (),
            embed=embed_tape(cards, clamped) if not isinstance(target, str) and any(clamped)
            else (),
            out_per_row=math.prod(cards),
            n_nodes=len(values),
        )

    def estimate_peak_bytes(
        self,
        model,
        target: str,
        evidence_keys=(),
        do_keys=(),
        *,
        B: int = 1,
        order: str = "min_fill",
    ) -> int:
        """Estimated peak bytes of answering this query (``B`` rows) by VE.

        ``target`` is a node name, or a sequence of names for a joint query.
        An upper bound of the memory the call makes the device hold
        (``nbn.inference._ve_memory``).

        ``evidence_keys`` / ``do_keys`` are the ``{node: value}`` mappings of
        the call, or bare node names.  The difference matters (#178): with
        values, the evidence is priced exactly — int64 tensors already on the
        model's device cost nothing, other tensors an int64 copy, Python
        lists the materialised tensor plus the copy.  With names alone each
        node is priced as the costliest input, a list of ``B`` floats (4 + 8
        bytes per row, plus the validation temporaries), which can be many
        times the real peak on evidence-heavy, factor-light queries (e.g. 16x
        on a naive Bayes with 29 observed children and ``B=20000``).  Pass
        the values whenever they are known.
        """
        pm = self._peak_model(model, target, evidence_keys, do_keys, B=B, order=order)
        return 0 if pm is None else pm.bytes(B)

    def _chunk_budget_bytes(self, device, budget: float) -> float:
        """Budget for sizing row-chunks: the stable budget, capped by live headroom.

        CUDA's budget is already live headroom.  On CPU / MPS the stable
        capacity-based budget decides *whether* VE runs; chunks additionally
        fit ``memory_safety`` × what is free now (N1).
        """
        if torch.device(device).type == "cuda":
            return budget
        head = device_headroom_bytes(device)
        return budget if head is None else min(budget, self.memory_safety * head)

    def _stable_budget_bytes(self, device, budget: float) -> float:
        """The part of ``budget`` that does not move with live usage: all of it
        on CPU / MPS; on CUDA only an explicit ``memory_budget``."""
        if torch.device(device).type != "cuda":
            return budget
        return float("inf") if self.memory_budget is None else float(self.memory_budget)

    def _live_room_bytes(self, device) -> float:
        """``memory_safety`` x what is free on ``device`` now (inf: unknown)."""
        head = device_headroom_bytes(device)
        return float("inf") if head is None else self.memory_safety * head

    def fits_memory(self, model, target, evidence_keys=(), do_keys=(), *,
                    order: str = "min_fill",
                    host_fraction: float | None = None) -> bool:
        """Whether one row of this query fits the budget (larger B is chunked).

        On MPS a row must also keep every intermediate factor under
        ``_MPS_MAX_ELEMENTS`` (beyond it MPS reductions are wrong).
        """
        pm = self._peak_model(model, target, evidence_keys, do_keys, order=order)
        if pm is None:
            return True  # answered as a point mass, no elimination
        elem_cap = _max_tensor_elements(model.device)
        if elem_cap is not None and _rows_within_elements(pm.cost.tensor, elem_cap) == 0:
            return False
        budget = self.memory_budget_bytes(model.device, host_fraction=host_fraction)
        return budget is None or pm.bytes(1) <= budget

    # ------------------------------------------------------------------ #
    # do-operator
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_do(model, do: Mapping[str, torch.Tensor], evidence) -> None:
        """Reject do-specifications VE cannot answer, before any work is done.

        Until this existed, ``do=`` reached ``query`` inside ``**kwargs`` and
        was *dropped on the floor*: ``model.query(["Y"], do={"X": 1})`` on an
        all-discrete network returned the observational answer (the prior,
        when no evidence was given) with no warning at all — the single most
        dangerous failure mode in the engine, because the number returned was
        plausible and wrong.  LW/AIS have always honoured ``do=``, so the two
        engines silently disagreed on the same call.
        """
        overlap = sorted(set(do) & set(evidence or {}))
        if overlap:
            raise ValueError(
                f"Node(s) {overlap} given as both evidence and do-intervention; "
                f"a node cannot be simultaneously observed and set."
            )
        unknown = sorted(n for n in do if n not in model.mechanisms)
        if unknown:
            raise ValueError(f"Unknown do-intervention target(s): {unknown}.")
        for node in do:
            mech = model.mechanisms[node]
            if not mech.is_discrete:
                raise ValueError(
                    f"TensorVariableElimination requires discrete "
                    f"do-targets; '{node}' is continuous."
                )

    # ------------------------------------------------------------------ #
    # Single-row query (kept identical to the v0.2 path)
    # ------------------------------------------------------------------ #

    def query(
        self,
        model,
        targets: List[str],
        evidence: Dict[str, torch.Tensor] | None = None,
        *,
        do: Mapping[str, torch.Tensor] | None = None,
        order: str = "min_fill",
        host_fraction: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Posterior for ``targets`` given one row of evidence / do-values.

        ``host_fraction`` overrides ``host_memory_fraction`` for this call
        (``HybridRouter`` passes its ``ve_host_fraction``).
        """
        self._validated = None  # never reuse another call's cache check
        # do-specific errors first (they name the offending do-target), then
        # the whole-model discreteness check, then the shape of the row.
        self._validate_do(model, dict(do or {}), evidence)
        _require_discrete(model)
        _require_single_row(evidence, do)
        if len(targets) != 1:
            # Joint query (R6): one row of the batched path, ``[K_1, …, K_n]``.
            row = {k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)).reshape(1)
                   for k, v in (evidence or {}).items()}
            do_row = {k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)).reshape(1)
                      for k, v in (do or {}).items()}
            return self.query_batch(model, targets, row, do=do_row, order=order,
                                    host_fraction=host_fraction)[0]
        target = targets[0]
        evidence = evidence or {}
        do = dict(do or {})
        self._validate_do(model, do, evidence)
        device = model.device

        def _as_int(v):
            return int(v.item()) if isinstance(v, torch.Tensor) else int(v)

        # Validate before ``int()``: it would truncate 1.5 to state 1, and
        # indexing would wrap -1 to the last state (R7 / C).
        check_discrete_values(model, {**evidence, **do}, do_keys=do)
        ev_int: Dict[str, int] = {k: _as_int(v) for k, v in evidence.items()}
        do_int: Dict[str, int] = {k: _as_int(v) for k, v in do.items()}

        # Observing or intervening on the target itself makes the answer a
        # point mass by definition.  Handled up front because conditioning
        # removes the target from every factor's scope, so elimination would
        # fall through to the uniform default (A5) — LW, which clamps and then
        # histograms, has always returned the delta here.
        clamped_target = {**ev_int, **do_int}
        if target in clamped_target:
            return _delta_probs(
                model.mechanisms[target].n_classes, clamped_target[target], device,
            )

        # Pre-allocation guard (single row), on every device — see
        # ``memory_budget_bytes``.  Raised before tabulating any CPT.
        budget = self.memory_budget_bytes(device, host_fraction=host_fraction)
        elem_cap = _max_tensor_elements(device)
        if budget is not None or elem_cap is not None:
            pm = self._peak_model(model, target, ev_int, do_int,
                                  order=order, price_evidence=False, remember=True)
            assert pm is not None  # the target is not clamped (handled above)
            peak = pm.bytes(1)
            if budget is not None and peak > budget:
                raise OutOfMemoryError(
                    _guard_message(peak, budget, device, order)
                )
            if elem_cap is not None and _rows_within_elements(pm.cost.tensor,
                                                              elem_cap) == 0:
                raise MPSTensorLimitError(_mps_limit_message(pm.cost.tensor, elem_cap))

        factors = self._extract_factors(model)

        # Pearl's do-operator on a factorised model is exactly: drop the
        # intervened node's CPT (severing its incoming edges) and clamp its
        # value everywhere else.  The remaining factors are the mutilated
        # model's joint, so conditioning + elimination on them yields
        # P(target | do(...)) after the final normalisation.
        factors = {n: f for n, f in factors.items() if n not in do_int}
        clamped = {**ev_int, **do_int}

        to_eliminate, relevant = self._plan_with_relevant(
            model, target, tuple(sorted(ev_int.keys())),
            do_keys=tuple(sorted(do_int.keys())), order=order,
        )

        # Bug 2 (#127): drop factors for non-relevant nodes.  Pruning the
        # elimination *order* alone is not enough — the final
        # "multiply remaining factors" step below would otherwise re-form
        # the giant joint over the dropped barren/m-separated variables
        # and OOM exactly as before.  The relevant set (requisite
        # probability nodes) is precisely the CPTs needed to preserve
        # P(target | evidence) up to the final softmax normalisation.
        factors = {n: f for n, f in factors.items() if n in relevant}

        # Condition each factor on evidence + clamped do-values
        conditioned: list[LogFactor] = []
        for _node, factor in factors.items():
            f = factor
            for ev_node, ev_val in clamped.items():
                if ev_node in f.variables:
                    f = f.condition({ev_node: ev_val})
            conditioned.append(f)
        for var in to_eliminate:
            rel = [f for f in conditioned if var in f.variables]
            if not rel:
                continue
            conditioned = [f for f in conditioned if var not in f.variables]
            # ``_multiply`` consumes ``rel``, so each input is released as
            # soon as it is multiplied in — the live set ``_plan_cost`` prices.
            conditioned.append(_multiply(rel).marginalise(var))

        if not conditioned:
            raise RuntimeError("No factors remaining after elimination.")
        result = _multiply(conditioned)

        k = model.mechanisms[target].n_classes
        if target not in result.variables:
            return torch.ones(k, device=device) / k

        if result.variables[-1] != target:
            dim = result.variables.index(target)
            perm = list(range(len(result.variables)))
            perm.append(perm.pop(dim))
            result = LogFactor(
                result.log_values.permute(*perm),
                [result.variables[p] for p in perm],
                result.cardinalities,
            )
        lv = result.log_values
        del result  # each reduction below may free its input
        while lv.dim() > 1:
            lv = torch.logsumexp(lv, dim=0)
        return torch.softmax(lv, dim=0).to(device)

    # ------------------------------------------------------------------ #
    # v0.3.1 — TRULY BATCHED query_batch
    # ------------------------------------------------------------------ #

    def query_batch(
        self,
        model,
        targets: List[str],
        evidence: Dict[str, torch.Tensor],
        *,
        do: Mapping[str, torch.Tensor] | None = None,
        order: str = "min_fill",
        host_fraction: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Vectorised batched VE.

        ``evidence`` is a dict ``{var: [B] long tensor}`` (or ``[B, 1]``).
        Returns ``[B, K]`` from a single pass of the elimination procedure.
        Evidence is applied factor-by-factor by fancy-indexing the
        per-axis evidence value, which introduces a leading B axis on every
        factor whose scope contained any evidence variable.  Subsequent
        factor products / marginalisations broadcast over that B axis
        natively.

        Output is bit-identical (≤1e-6) to looping ``query()`` over the
        batch on the same evidence — verified by
        ``tests/unit/test_vectorized_query_batch_correctness.py``.

        Several targets (R6) return the exact *joint* posterior
        ``[B, K_1, …, K_n]`` (axes in ``targets`` order, each row summing to
        1): all targets are kept and every other variable eliminated.
        """
        self._validated = None  # never reuse another call's cache check
        targets = list(targets)
        if not targets or len(set(targets)) != len(targets):
            raise ValueError(f"targets must be distinct node names, got {targets}.")
        joint = len(targets) > 1
        target = targets[0]
        do = dict(do or {})
        self._validate_do(model, do, evidence)
        _require_discrete(model)
        device = model.device

        # Normalise evidence + do to [B] long tensors on `device`.  do-values
        # may vary per row: the batched conditioner fancy-indexes each factor
        # with a [B] index tensor, so a per-row intervention costs nothing
        # extra over a per-row observation.
        raw: Dict[str, torch.Tensor] = {}
        B = 1
        for k, v in {**evidence, **do}.items():
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            if t.dim() == 0:
                t = t.reshape(1)
            elif t.dim() >= 2 and t.shape[-1] == 1:
                t = t.reshape(-1)
            raw[k] = t
            B = max(B, t.shape[0])
        # Validate before the cast to long, which would truncate 1.5 to 1.
        check_discrete_values(model, raw, do_keys=do)
        ev_norm: Dict[str, torch.Tensor] = {
            k: to_device_dtype(t, device, torch.long) for k, t in raw.items()
        }
        # Broadcast any [1] evidence to [B]
        ev_norm = {k: (v.expand(B) if v.shape[0] == 1 else v) for k, v in ev_norm.items()}

        if not joint and target in ev_norm:
            # Point mass per row (target observed or intervened) — see ``query``.
            k = model.mechanisms[target].n_classes
            out = torch.zeros(B, k, device=device)
            out.scatter_(1, ev_norm[target].reshape(B, 1), 1.0)
            return out
        cards = {t: model.mechanisms[t].n_classes for t in targets}
        free = [t for t in targets if t not in ev_norm]
        if joint and not free:
            return _embed_clamped(torch.ones(B, device=device), targets, free,
                                  ev_norm, cards)
        plan_target = tuple(free) if joint else target

        # Pre-allocation memory-budget guard (v0.6b round-2; every device
        # since B5).  The query's peak is priced from a tape of every
        # allocation the elimination makes, replayed through the device's
        # allocator (``nbn.inference._ve_memory``, #178), and compared with
        # the device budget (``memory_budget_bytes``).  Rejecting up front
        # beats letting the allocator fail mid-elimination: a partial CUDA OOM
        # fragments the heap, a CPU one can summon the OS OOM killer.  The
        # runner's ``run_with_guard`` classifies the raised
        # ``OutOfMemoryError`` as ``status='oom'``; ``HybridRouter`` answers
        # the query with LW instead.
        #
        # It runs before any CPT is tabulated: conditioning is itself an
        # allocation, so a query that will not fit must be refused before it.
        # (The evidence is already normalised by then; its copies are priced.)
        #
        # PR C: rather than rejecting a batch whose single row fits, split it
        # into the largest row-chunks that fit the same budget.  Next to every
        # chunk stay the batch's evidence tensors (chunks index views of
        # them), the finished answers and their final concatenation: that
        # reserve is subtracted from the budget the chunks are sized for.  If
        # not even one-row chunks fit the stable budget next to it, the batch
        # is refused (#301 review: it ran 1-row chunks over budget).  On a
        # cold cache the CPTs are tabulated first, so every chunk is priced
        # warm; chunks then run in a loop, not by re-entering this guard.
        #
        # N1: the *reject* decision uses the stable budget (so whether VE can
        # answer at all does not depend on momentary memory pressure), but the
        # *chunk size* also respects live headroom on CPU / MPS: chunking is
        # exact, so smaller chunks change only the speed, never the answer —
        # whereas sizing chunks to total RAM on a busy machine meant swap or
        # the OOM killer.
        guarded = kwargs.pop("_guarded", False)   # a chunk the caller already sized
        budget = None if guarded else self.memory_budget_bytes(
            device, host_fraction=host_fraction)
        elem_cap = None if guarded else _max_tensor_elements(device)
        if budget is not None or elem_cap is not None:
            pm = self._peak_model(
                model, tuple(targets) if joint else target,
                {k: v for k, v in evidence.items() if k not in do}, do,
                B=B, order=order, remember=True,
            )
            assert pm is not None  # a free target exists (point masses returned above)
            peak_b = pm.bytes(B)
            chunk, reason, sizer = B, None, None
            if elem_cap is not None:
                rows = _rows_within_elements(pm.cost.tensor, elem_cap)
                if rows == 0:
                    raise MPSTensorLimitError(_mps_limit_message(pm.cost.tensor, elem_cap))
                if rows < chunk:
                    chunk, reason = rows, f"the MPS {elem_cap}-element tensor limit"
            # Live headroom costs a syscall / cgroup usage read; a plan this
            # small next to the stable budget cannot need chunking.
            if budget is not None and peak_b > _HEADROOM_CHECK_FRACTION * budget:
                chunk_budget = self._chunk_budget_bytes(device, budget)
                tabulated = 0
                if peak_b > chunk_budget and pm.tables and B > 1:
                    # Cold cache: tabulate the CPTs once, here, if they fit next
                    # to the batch's evidence; then every chunk is warm and
                    # priced as such (#301 review B: each chunk re-entered cold,
                    # re-chunked ~150 levels deep, and the batch was refused).
                    if pm.tables_bytes() + pm.reserve_bytes(B) > budget:
                        raise OutOfMemoryError(_guard_message(
                            pm.tables_bytes() + pm.reserve_bytes(B), budget, device, order))
                    tabulated = pm.tables_bytes(resident=True)  # now held by this call
                    self._extract_factors(model)
                    pm = self._peak_model(
                        model, tuple(targets) if joint else target,
                        {k: v for k, v in evidence.items() if k not in do}, do,
                        B=B, order=order, remember=True,
                    )
                    peak_b = pm.bytes(B) + tabulated
                if peak_b > chunk_budget:   # chunk_budget <= budget
                    child = pm.child()
                    held = pm.reserve_bytes(B) + tabulated
                    one_row = peak_b if B == 1 else held + child.bytes(1)
                    if one_row > budget:
                        if peak_b > budget:
                            # Not even one-row chunks fit the stable budget.
                            raise OutOfMemoryError(
                                _guard_message(one_row, budget, device, order))
                        # Only live headroom is short and chunking would cost
                        # more than one pass: attempt the pass (N1).
                    else:
                        # At least one row per chunk: a row that fits the
                        # stable budget is always attempted (N1).  Chunks are
                        # re-sized against live headroom as the batch runs.
                        stable = self._stable_budget_bytes(device, budget)
                        outputs = 2 * pm.output_bytes(B)

                        def sizer(limit, child=child, held=held, stable=stable,
                                  outputs=outputs):
                            # The stable budget does not see what this call
                            # already holds (evidence, tabulated CPTs) or will
                            # hold (finished answers, their concatenation):
                            # subtract it.  Live headroom already reflects
                            # what is allocated; only the answers still to
                            # come are subtracted from it (#301 review 3).
                            room = min(stable - held,
                                       self._live_room_bytes(device) - outputs)
                            return max(1, _max_fitting_rows(child.bytes, limit, room))
                        rows = sizer(B - 1)
                        if rows < chunk:
                            chunk = rows
                            reason = (f"the {chunk_budget / 1024 ** 3:.2f} GiB chunk "
                                      f"budget (estimated peak "
                                      f"{peak_b / 1024 ** 3:.2f} GiB for B={B})")
            if chunk < B:
                logger.info(
                    "TensorVariableElimination.query_batch on %s: splitting B=%d "
                    "into chunks of %d rows to respect %s.", device, B, chunk, reason,
                )
                return self._query_batch_chunked(
                    model, targets, ev_norm, B=B, chunk=chunk,
                    order=order, do_keys=tuple(do),
                    host_fraction=host_fraction, sizer=sizer, **kwargs,
                )

        factors = self._extract_factors(model)
        # See ``query``: mutilation = drop the intervened CPTs, clamp the rest.
        factors = {n: f for n, f in factors.items() if n not in do}

        to_eliminate, relevant = self._plan_with_relevant(
            model, plan_target, tuple(sorted(k for k in ev_norm if k not in do)),
            do_keys=tuple(sorted(do.keys())), order=order,
        )

        # Bug 2 (#127): drop factors for non-relevant nodes before
        # conditioning.  See the rationale in ``query`` — order pruning
        # alone would relocate, not fix, the barley OOM because the
        # final factor product re-forms the giant joint.
        factors = {n: f for n, f in factors.items() if n in relevant}

        # Condition each factor batchwise.
        # `conditioned` holds tuples `(log_values_tensor, vars, has_batch)`.
        # Built by a comprehension so no loop variable keeps a factor alive
        # after elimination consumes it (``_plan_cost`` prices that).
        conditioned: list[tuple[torch.Tensor, list[str], bool]] = [
            _condition_factor_batched(f.log_values, f.variables, ev_norm)
            for f in factors.values()
        ]
        del factors

        for var in to_eliminate:
            rel = [f for f in conditioned if var in f[1]]
            if not rel:
                continue
            conditioned = [f for f in conditioned if var not in f[1]]
            # ``_multiply_batched`` consumes ``rel``: each input is released as
            # soon as it is multiplied in.
            prod_lv, prod_vars, prod_has_b = _multiply_batched(rel)
            dim = prod_vars.index(var) + (1 if prod_has_b else 0)
            conditioned.append((
                torch.logsumexp(prod_lv, dim=dim),
                [v for v in prod_vars if v != var],
                prod_has_b,
            ))
            del prod_lv  # the product dies here, not at the next step

        if joint:
            return _joint_from_factors(conditioned, targets, free, ev_norm,
                                       cards, B, device)

        # Multiply remaining factors (only target should be in scope)
        if not conditioned:
            return torch.full((B, model.mechanisms[target].n_classes),
                              1.0 / model.mechanisms[target].n_classes, device=device)
        prod_lv, prod_vars, prod_has_b = _multiply_batched(conditioned)

        k = model.mechanisms[target].n_classes
        # Move target axis to last.
        if target not in prod_vars:
            return torch.full((B, k), 1.0 / k, device=device)
        offset = 1 if prod_has_b else 0
        # Reduce any non-target, non-batch axes via logsumexp (shouldn't
        # exist after elimination but defensive).
        for i in range(len(prod_vars) - 1, -1, -1):
            if prod_vars[i] != target:
                prod_lv = torch.logsumexp(prod_lv, dim=i + offset)
                prod_vars.pop(i)
        if not prod_has_b:
            # No evidence variable was in any factor — broadcast a B leading dim.
            prod_lv = prod_lv.unsqueeze(0).expand(B, *prod_lv.shape)
        # prod_lv shape: [B, K]
        return torch.softmax(prod_lv, dim=-1).to(device)

    def _query_batch_chunked(
        self,
        model,
        targets: List[str],
        ev_norm: Dict[str, torch.Tensor],
        *,
        B: int,
        chunk: int,
        order: str,
        do_keys: tuple[str, ...] = (),
        host_fraction: float | None = None,
        sizer=None,
        **kwargs,
    ) -> torch.Tensor:
        """Run ``query_batch`` over row-chunks of ``ev_norm`` and concatenate.

        ``ev_norm`` is the already-normalised evidence *and do-values*
        (``[B]`` long tensors, broadcast applied), so slicing rows is exact.
        ``do_keys`` names the entries that are interventions: they must go
        back in as ``do=``, not as evidence — re-entering with them merged
        into ``evidence`` silently turned ``P(Y | do(X))`` into the
        observational ``P(Y | X)`` on exactly the large-B CUDA batches that
        get chunked (A8).  Chunks run without the guard, in a loop (no
        recursion): the caller sized them, and ``sizer(limit)``, when given,
        re-sizes each against live headroom before it runs.  Output is
        ``[B, K]``, row-identical to the single-pass result.
        """
        outs = []
        i = 0
        while i < B:
            c = min(chunk, B - i)
            if sizer is not None:
                c = min(c, sizer(c))
            rows = {k: v[i:i + c] for k, v in ev_norm.items()}
            do_rows = {k: rows.pop(k) for k in do_keys}
            outs.append(self.query_batch(
                model, targets, rows, do=do_rows, order=order,
                host_fraction=host_fraction, _guarded=True, **kwargs,
            ))
            i += c
        return torch.cat(outs, dim=0)


def _max_tensor_elements(device) -> int | None:
    """Largest intermediate factor (elements) VE may build on ``device``."""
    return _MPS_MAX_ELEMENTS if torch.device(device).type == "mps" else None


def _rows_within_elements(profile: tuple[int, int], cap: int) -> int:
    """Rows per chunk keeping every intermediate factor below ``cap`` elements.

    ``profile`` is ``(batch_free, per_row)``, ``_PlanCost.tensor``: with ``c``
    rows the largest factor has ``max(batch_free, c * per_row)`` elements.
    0 when not even one row fits.
    """
    batch_free, per_row = profile
    if batch_free >= cap or per_row >= cap:
        return 0
    if per_row == 0:
        return 1 << 62  # nothing carries the batch axis: any B fits
    return (cap - 1) // per_row


def _mps_limit_message(profile: tuple[int, int], cap: int) -> str:
    return (
        f"TensorVariableElimination: this query needs an intermediate factor of "
        f"{max(profile):,} elements, but MPS reductions return wrong values past "
        f"{cap:,} elements (torch 2.14.1); refusing rather than answering "
        f"wrongly.  Run it on CPU/CUDA, or use HybridRouter / likelihood "
        f"weighting, which answers it approximately."
    )


def _require_discrete(model) -> None:
    """VE needs every mechanism discrete — checked before any planning.

    The memory guard prices the plan from cardinalities, so on a hybrid
    network it used to fail first with an opaque ``AttributeError``
    (``'MDNMechanism' has no 'n_classes'``) instead of this message (N3).
    """
    for node, mech in model.mechanisms.items():
        if not mech.is_discrete:
            raise ValueError(
                f"TensorVariableElimination only supports discrete mechanisms; "
                f"node '{node}' has a continuous mechanism."
            )


def _require_single_row(evidence, do) -> None:
    """``query`` takes one value per node; a batch belongs in ``query_batch``."""
    for kind, values in (("evidence", evidence), ("do", do)):
        for node, v in (values or {}).items():
            n = v.numel() if isinstance(v, torch.Tensor) else torch.as_tensor(v).numel()
            if n != 1:
                raise ValueError(
                    f"TensorVariableElimination.query takes one {kind} value per "
                    f"node, got {n} values for {node!r}; use query_batch for a "
                    f"batch of rows."
                )


def _as_values(x) -> dict:
    """``{node: value}`` from a mapping, or ``{node: None}`` from bare names."""
    return dict(x) if isinstance(x, Mapping) else dict.fromkeys(x)


def _plan_target(target, clamped: set):
    """Plan key for ``target`` (name or names) with clamped targets removed.

    ``None`` when every target is clamped.  A single name stays a ``str`` so
    single-target plans keep their historical cache keys.
    """
    if isinstance(target, str):
        return None if target in clamped else target
    free = tuple(t for t in target if t not in clamped)
    if not free:
        return None
    return free[0] if len(free) == 1 and len(tuple(target)) == 1 else free


def _joint_from_factors(conditioned, targets, free, ev_norm, cards, B, device):
    """Normalised joint ``[B, K_1, …, K_n]`` from the post-elimination factors.

    After eliminating every non-target variable, the remaining factors mention
    only the free (unclamped) targets; their product is the unnormalised
    joint over them.  Clamped targets are point masses and are inserted as
    one-hot axes, so the output always has one axis per requested target.
    """
    if conditioned:
        prod_lv, prod_vars, prod_has_b = _multiply_batched(conditioned)
        offset = 1 if prod_has_b else 0
        # Defensive: sum out anything that is not a free target.
        for i in range(len(prod_vars) - 1, -1, -1):
            if prod_vars[i] not in free:
                prod_lv = torch.logsumexp(prod_lv, dim=i + offset)
                prod_vars = prod_vars[:i] + prod_vars[i + 1:]
        if not prod_has_b:
            prod_lv = prod_lv.unsqueeze(0).expand(B, *prod_lv.shape)
        # A free target absent from every factor is uniform: broadcast it.
        for t in free:
            if t not in prod_vars:
                prod_lv = prod_lv.unsqueeze(-1).expand(*prod_lv.shape, cards[t])
                prod_vars = prod_vars + [t]
        perm = [0] + [1 + prod_vars.index(t) for t in free]
        prod_lv = prod_lv.permute(*perm)
    else:
        prod_lv = torch.zeros(B, *[cards[t] for t in free], device=device)
    flat = prod_lv.reshape(B, -1)
    probs = torch.softmax(flat, dim=-1).reshape(prod_lv.shape).to(device)
    return _embed_clamped(probs, targets, free, ev_norm, cards)


def _embed_clamped(probs, targets, free, ev_norm, cards):
    """Insert a one-hot axis per clamped target into ``probs`` ``[B, *free]``.

    Axes are inserted in ``targets`` order, so at each insertion every
    earlier target already has its axis and the result ends as
    ``[B, K_1, …, K_n]``.
    """
    B = probs.shape[0]
    for i, t in enumerate(targets):
        if t in free:
            continue
        onehot = torch.nn.functional.one_hot(ev_norm[t], cards[t]).to(probs.dtype)
        tail = probs.dim() - 1 - i
        probs = probs.unsqueeze(i + 1) * onehot.reshape(B, *([1] * i), cards[t], *([1] * tail))
    return probs


def _delta_probs(k: int, value: int, device) -> torch.Tensor:
    """One-hot ``[K]`` distribution — the answer to ``P(X | do(X = value))``."""
    out = torch.zeros(k, device=device)
    out[int(value)] = 1.0
    return out


def _guard_message(peak: float, budget: float, device, order: str) -> str:
    return (
        f"TensorVariableElimination: query out of memory "
        f"pre-allocation guard — plan would need "
        f"~{peak / 1024 ** 3:.2f} GiB peak "
        f"intermediate factor at order={order!r}, but the memory budget "
        f"is {budget / 1024 ** 3:.2f} GiB on {device}.  Query rejected; "
        f"try a coarser query (fewer evidence variables), a different "
        f"elimination order, or likelihood weighting (HybridRouter falls "
        f"back to it automatically)."
    )


# ---------------------------------------------------------------------- #
# Batched factor helpers
# ---------------------------------------------------------------------- #

def _condition_factor_batched(
    log_values: torch.Tensor,
    factor_vars: list[str],
    evidence_batch: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, list[str], bool]:
    """Slice ``log_values`` by per-variable evidence and add a leading B axis.

    Returns ``(out_log_values, remaining_vars, has_batch)``. ``out_log_values``
    has shape ``[B, *remaining_card]`` when ``has_batch`` is True, else
    ``[*factor_card]`` (no evidence variable touched this factor).
    """
    # Identify which evidence variables sit in this factor's scope
    evidence_in_scope = [v for v in factor_vars if v in evidence_batch]
    if not evidence_in_scope:
        return log_values, list(factor_vars), False

    # Build a per-axis indexer:
    #   evidence axis  -> the [B] long tensor (advanced index)
    #   unobserved axis -> slice(None)
    # PyTorch advanced-indexing rules place the broadcast batch axis:
    #   * at the position of the first advanced index when all advanced
    #     indexes are contiguous (e.g. ``[idx_a, idx_b, :]`` -> [B, c]).
    #   * at the front when advanced indexes are separated by slices
    #     (e.g. ``[idx_a, :, idx_c]`` -> [B, b]).
    # We normalise to "batch axis at front" with an explicit ``movedim``.
    indexers: list = []
    remaining: list[str] = []
    adv_positions: list[int] = []
    for i, v in enumerate(factor_vars):
        if v in evidence_batch:
            indexers.append(evidence_batch[v])
            adv_positions.append(i)
        else:
            indexers.append(slice(None))
            remaining.append(v)
    out = log_values[tuple(indexers)]
    if adv_positions:
        # Determine where PyTorch placed the broadcast (batch) axis.
        contiguous = (max(adv_positions) - min(adv_positions) + 1
                      == len(adv_positions))
        batch_axis = min(adv_positions) if contiguous else 0
        if batch_axis != 0:
            out = out.movedim(batch_axis, 0)
    return out, remaining, True


def _log_factor_product_batched(
    a_lv: torch.Tensor, a_vars: list[str], a_has_b: bool,
    b_lv: torch.Tensor, b_vars: list[str], b_has_b: bool,
) -> tuple[torch.Tensor, list[str], bool]:
    """Multiply two log-factors, possibly with a leading B batch dim.

    Output keeps ``has_batch = a_has_b or b_has_b`` and aligns axes by
    variable name.  Broadcasting handles size-1 padding.
    """
    all_vars = list(dict.fromkeys(a_vars + b_vars))
    out_has_b = a_has_b or b_has_b

    def _align(lv: torch.Tensor, vars_: list[str], has_b: bool) -> torch.Tensor:
        offset = 1 if has_b else 0
        present = [v for v in all_vars if v in vars_]
        if present:
            perm = [vars_.index(v) + offset for v in present]
            if has_b:
                lv = lv.permute(0, *perm)
            else:
                lv = lv.permute(*perm)
        # Insert size-1 axes for the missing variables
        full_lv = lv
        for i, v in enumerate(all_vars):
            if v not in vars_:
                axis = i + (1 if has_b else 0)
                full_lv = full_lv.unsqueeze(axis)
        # If the output should have B but this factor doesn't, prepend size-1
        if out_has_b and not has_b:
            full_lv = full_lv.unsqueeze(0)
        return full_lv

    a_aligned = _align(a_lv, a_vars, a_has_b)
    b_aligned = _align(b_lv, b_vars, b_has_b)
    return a_aligned + b_aligned, all_vars, out_has_b


def _multiply_batched(factors: list) -> tuple[torch.Tensor, list[str], bool]:
    """Product of batched log-factors, *consuming* ``factors``.

    Each input is popped before it is multiplied in, so (with no other
    reference held by the caller) it is freed as soon as it has been used —
    the live set ``_plan_cost`` prices.  Iterating over a list the caller
    kept held every input until the elimination step ended.
    """
    factors.reverse()
    prod = factors.pop()
    while factors:
        prod = _log_factor_product_batched(*prod, *factors.pop())
    return prod


def _multiply(factors: list[LogFactor]) -> LogFactor:
    """Single-row ``_multiply_batched``: consumes ``factors``."""
    factors.reverse()
    prod = factors.pop()
    while factors:
        prod = _log_factor_product(prod, factors.pop())
    return prod


def _log_factor_product(a: LogFactor, b: LogFactor) -> LogFactor:
    """Single-row factor product (kept for ``query()``)."""
    all_vars = list(dict.fromkeys(a.variables + b.variables))
    cards = {**a.cardinalities, **b.cardinalities}

    def _align(f: LogFactor) -> torch.Tensor:
        lv = f.log_values
        present = [v for v in all_vars if v in f.variables]
        if present:
            perm = [f.variables.index(v) for v in present]
            lv = lv.permute(*perm)
        full_lv = lv
        for i, v in enumerate(all_vars):
            if v not in f.variables:
                full_lv = full_lv.unsqueeze(i)
        return full_lv

    return LogFactor(_align(a) + _align(b), all_vars, cards)
