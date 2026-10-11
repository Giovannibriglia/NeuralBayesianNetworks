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
from nbn.inference._memory import OutOfMemoryError, device_headroom_bytes
from nbn.inference._validation import check_discrete_values
from nbn.inference._pruning import relevant_subnetwork, requisite_nodes
from nbn.inference.base import InferenceEngine

logger = logging.getLogger(__name__)

# Empirical broadcast live-set multiplier (issue #177).
#
# The per-step max-factor estimate in ``_estimate_peak_bytes`` captures
# only one union-scope tensor, but ``_log_factor_product_batched``
# materialises three simultaneously (``a_aligned`` + ``b_aligned`` +
# their sum), and conditioned factors persist across elimination steps.
# Measurement (2026-06-08, nbn-neuralcat-ve, B=16..128) found
# actual/predicted = 2.85× rock-stable across batch sizes; 3.0 rounds
# conservative. See #177 for the measurement and #178 for the deeper
# live-set model that would replace this scalar.
_LIVESET_MULTIPLIER = 3.0


class TensorVariableElimination(InferenceEngine):
    """Log-domain VE for discrete BNs (single-row + truly batched queries)."""

    _transient_caches = ("_factor_cache", "_plan_cache", "_cost_cache")

    def __init__(
        self,
        treewidth_threshold: int = 25,
        memory_budget: float | None = None,
        memory_safety: float = 0.9,
    ) -> None:
        """
        Parameters
        ----------
        memory_budget:
            Optional hard cap, in bytes, on the estimated elimination peak.
            The effective budget is ``min(memory_budget, memory_safety ×
            device headroom)``; with neither known no check is made.
        memory_safety:
            Fraction of the device headroom (``nbn.inference._memory``) a
            query may plan to use.
        """
        self.treewidth_threshold = treewidth_threshold
        self.memory_budget = memory_budget
        self.memory_safety = float(memory_safety)
        # Memoised per-plan peak profile ``(batch_free, per_row)`` element
        # counts, keyed like ``_plan_cache`` (see ``_peak_profile``).
        self._cost_cache: Dict[tuple, tuple] = {}
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

    def _extract_factors(self, model) -> Dict[str, LogFactor]:
        version = (structure_version(model), str(model.device))
        hit = cache_get(self._factor_cache, id(model), model, version)
        if hit is not None:
            tensors, versions, factors = hit
            if versions is not None and tensor_versions(tensors) == versions:
                return factors
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
            graph, target, tuple(evidence_keys) + tuple(do_keys),
        )
        # #132 (Option B): an observed node the ball never enters from a
        # parent (e.g. an observed parent/ancestor of the target) is
        # relevant as *evidence* — its value is clamped wherever it appears
        # in a kept CPT — but its own CPT is not requisite: evaluated at the
        # observed value it depends only on its parents and cancels in the
        # final normalisation.  Dropping it keeps those parents out of the
        # elimination (barley: ``sorttkv``'s CPT dragged in card-67 ``sort``).
        non_requisite = set(evidence_keys) - requisite_nodes(
            graph, target, tuple(evidence_keys) + tuple(do_keys),
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
            targets=[target],
            # do-nodes are clamped like evidence, so they are conditioned out
            # rather than eliminated — prune them from the moral graph too.
            evidence=list(evidence_keys) + list(do_keys),
        )
        result = (elimination_order, relevant)
        return self._cache_put(self._plan_cache, key, model, result)

    # ------------------------------------------------------------------ #
    # Memory budget (every device)
    # ------------------------------------------------------------------ #

    def memory_budget_bytes(self, device) -> float | None:
        """Bytes an elimination on ``device`` may plan to use (``None``: unchecked).

        The guard used to run on CUDA only, reading driver-free memory; CPU
        and MPS allocated blindly, so a terabyte plan was attempted and died
        in the allocator (or the OS OOM killer) instead of being refused or
        routed to LW (B5).
        """
        head = device_headroom_bytes(device)
        budget = None if head is None else self.memory_safety * head
        if self.memory_budget is not None:
            cap = float(self.memory_budget)
            budget = cap if budget is None else min(budget, cap)
        return budget

    def _peak_profile(
        self,
        model,
        target: str,
        evidence_keys: tuple[str, ...],
        do_keys: tuple[str, ...],
        order: str,
    ) -> tuple[int, int]:
        """``(batch_free, per_row)`` peak element counts of this query's plan.

        Computed from the DAG and cardinalities alone — no CPT is tabulated —
        so the router can price a query before committing to VE.  The peak
        in bytes for any ``B`` follows from the pair (see
        ``_peak_bytes_from_profile``); memoised per plan.
        """
        key = (id(model), target, evidence_keys, do_keys, order)
        hit = self._cache_get(self._cost_cache, key, model)
        if hit is not None:
            return hit
        plan, relevant = self._plan_with_relevant(
            model, target, evidence_keys, do_keys=do_keys, order=order,
        )
        scopes = {}
        for node in relevant:
            if node in do_keys:
                continue  # graph surgery: the intervened CPT is dropped
            var_scope = list(model.dag.parents(node)) + [node]
            scopes[node] = _Scope(
                var_scope, {n: model.mechanisms[n].n_classes for n in var_scope},
            )
        prof = _peak_profile(
            plan, scopes, evidence_keys=tuple(evidence_keys) + tuple(do_keys),
        )
        return self._cache_put(self._cost_cache, key, model, prof)

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
        """Estimated peak bytes of answering this query (``B`` rows) by VE."""
        ev = tuple(sorted(k for k in evidence_keys if k not in set(do_keys)))
        prof = self._peak_profile(model, target, ev, tuple(sorted(do_keys)), order)
        return _peak_bytes_from_profile(prof, B)

    def fits_memory(self, model, target, evidence_keys=(), do_keys=(), *,
                    order: str = "min_fill") -> bool:
        """Whether one row of this query fits the budget (larger B is chunked)."""
        budget = self.memory_budget_bytes(model.device)
        if budget is None:
            return True
        return self.estimate_peak_bytes(
            model, target, evidence_keys, do_keys, B=1, order=order,
        ) <= budget

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
        **kwargs,
    ) -> torch.Tensor:
        if len(targets) != 1:
            raise NotImplementedError("TensorVE supports exactly one target.")
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
        budget = self.memory_budget_bytes(device)
        if budget is not None:
            peak = self.estimate_peak_bytes(
                model, target, tuple(ev_int), tuple(do_int), B=1, order=order,
            )
            if peak > budget:
                raise OutOfMemoryError(
                    _guard_message(peak, budget, device, order)
                )

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
            relevant = [f for f in conditioned if var in f.variables]
            rest = [f for f in conditioned if var not in f.variables]
            if not relevant:
                continue
            product = relevant[0]
            for f in relevant[1:]:
                product = _log_factor_product(product, f)
            product = product.marginalise(var)
            conditioned = rest + [product]

        if not conditioned:
            raise RuntimeError("No factors remaining after elimination.")
        result = conditioned[0]
        for f in conditioned[1:]:
            result = _log_factor_product(result, f)

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
        """
        if len(targets) != 1:
            raise NotImplementedError("TensorVE supports exactly one target.")
        target = targets[0]
        do = dict(do or {})
        self._validate_do(model, do, evidence)
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
            k: t.to(device=device, dtype=torch.long) for k, t in raw.items()
        }
        # Broadcast any [1] evidence to [B]
        ev_norm = {k: (v.expand(B) if v.shape[0] == 1 else v) for k, v in ev_norm.items()}

        if target in ev_norm:
            # Point mass per row (target observed or intervened) — see ``query``.
            k = model.mechanisms[target].n_classes
            out = torch.zeros(B, k, device=device)
            out.scatter_(1, ev_norm[target].reshape(B, 1), 1.0)
            return out

        # Pre-allocation memory-budget guard (v0.6b round-2; every device
        # since B5).  The peak intermediate-factor size is estimated by
        # walking the plan algebraically (``_peak_profile``, same algebra as
        # ``nbn/bench/diagnostics/ve_profile_n20.py``) and compared with the
        # device budget (``memory_budget_bytes``).  Rejecting up front beats
        # letting the allocator fail mid-elimination: a partial CUDA OOM
        # fragments the heap, a CPU one can summon the OS OOM killer.  The
        # runner's ``run_with_guard`` classifies the raised
        # ``OutOfMemoryError`` as ``status='oom'``; ``HybridRouter`` answers
        # the query with LW instead.
        #
        # It runs before any CPT is tabulated or any tensor touches
        # ``device``: conditioning is itself an allocation, so a query that
        # will not fit must be refused before it.
        #
        # PR C: the peak is linear in B in the regime where the guard fires
        # (the peak step carries the batch axis), so rather than rejecting
        # the whole batch it is split into the largest row-chunks that fit
        # the same budget (``_LIVESET_MULTIPLIER`` unchanged — separately
        # calibrated, #177/#178).
        budget = self.memory_budget_bytes(device)
        if budget is not None:
            ev_keys = tuple(sorted(k for k in ev_norm if k not in do))
            do_keys = tuple(sorted(do))
            prof = self._peak_profile(model, target, ev_keys, do_keys, order)
            peak_b = _peak_bytes_from_profile(prof, B)
            if peak_b > budget:
                peak_1 = _peak_bytes_from_profile(prof, 1)
                chunk = _max_chunk_rows(peak_b, peak_1, B, budget)
                if chunk == 0:
                    # Even a single row exceeds the budget: chunking cannot help.
                    raise OutOfMemoryError(
                        _guard_message(peak_1, budget, device, order)
                    )
                logger.info(
                    "TensorVariableElimination.query_batch: estimated peak "
                    "%.2f GiB for B=%d exceeds the memory budget (%.2f GiB) "
                    "on %s; splitting into chunks of %d rows.",
                    peak_b / 1024 ** 3, B, budget / 1024 ** 3, device, chunk,
                )
                return self._query_batch_chunked(
                    model, targets, ev_norm, B=B, chunk=chunk,
                    order=order, do_keys=tuple(do), **kwargs,
                )

        factors = self._extract_factors(model)
        # See ``query``: mutilation = drop the intervened CPTs, clamp the rest.
        factors = {n: f for n, f in factors.items() if n not in do}

        to_eliminate, relevant = self._plan_with_relevant(
            model, target, tuple(sorted(k for k in ev_norm if k not in do)),
            do_keys=tuple(sorted(do.keys())), order=order,
        )

        # Bug 2 (#127): drop factors for non-relevant nodes before
        # conditioning.  See the rationale in ``query`` — order pruning
        # alone would relocate, not fix, the barley OOM because the
        # final factor product re-forms the giant joint.
        factors = {n: f for n, f in factors.items() if n in relevant}

        # Condition each factor batchwise.
        # `conditioned` holds tuples `(log_values_tensor, vars, has_batch)`.
        conditioned: list[tuple[torch.Tensor, list[str], bool]] = []
        for f in factors.values():
            lv, vars_, has_b = _condition_factor_batched(
                f.log_values, f.variables, ev_norm,
            )
            conditioned.append((lv, vars_, has_b))

        for var in to_eliminate:
            relevant = [f for f in conditioned if var in f[1]]
            rest = [f for f in conditioned if var not in f[1]]
            if not relevant:
                continue
            prod_lv, prod_vars, prod_has_b = relevant[0]
            for lv2, vars2, has_b2 in relevant[1:]:
                prod_lv, prod_vars, prod_has_b = _log_factor_product_batched(
                    prod_lv, prod_vars, prod_has_b, lv2, vars2, has_b2,
                )
            # Marginalise
            dim = prod_vars.index(var) + (1 if prod_has_b else 0)
            prod_lv = torch.logsumexp(prod_lv, dim=dim)
            prod_vars = [v for v in prod_vars if v != var]
            conditioned = rest + [(prod_lv, prod_vars, prod_has_b)]

        # Multiply remaining factors (only target should be in scope)
        if not conditioned:
            return torch.full((B, model.mechanisms[target].n_classes),
                              1.0 / model.mechanisms[target].n_classes, device=device)
        prod_lv, prod_vars, prod_has_b = conditioned[0]
        for lv2, vars2, has_b2 in conditioned[1:]:
            prod_lv, prod_vars, prod_has_b = _log_factor_product_batched(
                prod_lv, prod_vars, prod_has_b, lv2, vars2, has_b2,
            )

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
        **kwargs,
    ) -> torch.Tensor:
        """Run ``query_batch`` over row-chunks of ``ev_norm`` and concatenate.

        ``ev_norm`` is the already-normalised evidence *and do-values*
        (``[B]`` long tensors, broadcast applied), so slicing rows is exact.
        ``do_keys`` names the entries that are interventions: they must go
        back in as ``do=``, not as evidence — re-entering with them merged
        into ``evidence`` silently turned ``P(Y | do(X))`` into the
        observational ``P(Y | X)`` on exactly the large-B CUDA batches that
        get chunked (A8).  Each chunk re-enters ``query_batch`` — plan/factor
        caches hit, and the guard re-checks each chunk against live free
        memory (shrinking further or raising if the budget collapsed
        mid-batch).  Output is ``[B, K]``, row-identical to the single-pass
        result.
        """
        outs = []
        for i in range(0, B, chunk):
            rows = {k: v[i:i + chunk] for k, v in ev_norm.items()}
            do_rows = {k: rows.pop(k) for k in do_keys}
            outs.append(self.query_batch(
                model, targets, rows, do=do_rows, order=order, **kwargs,
            ))
        return torch.cat(outs, dim=0)


def _delta_probs(k: int, value: int, device) -> torch.Tensor:
    """One-hot ``[K]`` distribution — the answer to ``P(X | do(X = value))``."""
    out = torch.zeros(k, device=device)
    out[int(value)] = 1.0
    return out


# ---------------------------------------------------------------------- #
# Memory-budget estimator (v0.6b round-2)
# ---------------------------------------------------------------------- #


def _max_chunk_rows(
    peak_at_B: int, peak_at_1: int, B: int, budget: float,
) -> int:
    """Largest per-chunk row count whose estimated peak fits ``budget``.

    Pure arithmetic over two ``_estimate_peak_bytes`` evaluations (at the
    requested ``B`` and at ``B=1``) so the chunk decision is unit-testable
    without a cuda device.  ``_estimate_peak_bytes`` is
    ``max(A, B·R)·_LIVESET_MULTIPLIER·dtype_bytes`` where ``A`` is the
    largest batch-free step and ``R`` the largest per-row batched step; in
    the regime where the guard fires with ``peak_at_1 <= budget``, the peak
    is batch-dominated (``peak_at_B == B·R``), so the per-row cost is
    exactly ``peak_at_B / B``.

    Returns
    -------
    ``B`` when the full batch already fits (single pass, unchanged
    behaviour); ``0`` when even a single row exceeds the budget (caller
    raises the pre-existing guard error); otherwise the largest fitting
    chunk size in ``[1, B - 1]``.
    """
    if peak_at_B <= budget:
        return B
    if B <= 1 or peak_at_1 > budget:
        return 0
    per_row = peak_at_B / B
    return max(1, min(int(budget // per_row), B - 1))


class _Scope:
    """Scope-only stand-in for a ``LogFactor`` (variables + cardinalities).

    The peak estimate needs only factor scopes, so it can be computed from
    the DAG before any CPT is tabulated.
    """

    __slots__ = ("variables", "cardinalities")

    def __init__(self, variables, cardinalities) -> None:
        self.variables = list(variables)
        self.cardinalities = dict(cardinalities)


def _peak_profile(
    plan: list[str],
    factors: Mapping[str, object],
    *,
    evidence_keys: tuple[str, ...],
) -> tuple[int, int]:
    """Largest elimination-step factor, split by whether it carries the B axis.

    Returns ``(batch_free, per_row)`` element counts: the peak for a batch
    of ``B`` rows is ``max(batch_free, B · per_row)`` elements (before the
    live-set multiplier), so one walk prices every ``B``.  ``factors`` maps
    node → anything with ``variables`` / ``cardinalities`` (``LogFactor`` or
    ``_Scope``).
    """
    ev_set = set(evidence_keys)
    state: list[tuple[set[str], bool, Dict[str, int]]] = []
    for f in factors.values():
        # _condition_factor_batched drops evidence vars from the scope
        # and prepends a B-axis if any evidence touched the factor.
        scope: set[str] = set(f.variables) - ev_set
        has_b = any(v in ev_set for v in f.variables)
        state.append((scope, has_b, dict(f.cardinalities)))

    batch_free = per_row = 0
    for var in plan:
        relevant = [s for s in state if var in s[0]]
        rest = [s for s in state if var not in s[0]]
        if not relevant:
            continue
        union_scope: set[str] = set().union(*(s[0] for s in relevant))
        prod_has_b = any(s[1] for s in relevant)
        cards: Dict[str, int] = {}
        for s in relevant:
            cards.update(s[2])
        elements = 1
        for v in union_scope:
            elements *= cards.get(v, 1)
        if prod_has_b:
            per_row = max(per_row, elements)
        else:
            batch_free = max(batch_free, elements)
        # Marginalise: drop var from the surviving factor.
        union_scope.discard(var)
        state = rest + [(union_scope, prod_has_b, cards)]
    return batch_free, per_row


def _peak_bytes_from_profile(
    profile: tuple[int, int], B: int, dtype_bytes: int = 4,
) -> int:
    """Peak bytes for ``B`` rows from a ``_peak_profile`` pair (see there)."""
    batch_free, per_row = profile
    peak = max(batch_free, B * per_row) * dtype_bytes
    # Scale by the broadcast live-set multiplier (#177): the bare ``peak``
    # is the largest single union-scope factor, but the realised peak holds
    # ~3 such tensors plus persistent conditioned factors.
    return int(peak * _LIVESET_MULTIPLIER)


def _estimate_peak_bytes(
    plan: list[str],
    factors: Dict[str, LogFactor],
    *,
    evidence_keys: tuple[str, ...],
    B: int,
    dtype_bytes: int = 4,
) -> int:
    """Predict the peak intermediate-factor size in bytes.

    Walks the elimination ``plan`` algebraically — the same algebra as
    the inner loop of :meth:`TensorVariableElimination.query_batch` and
    of the diagnostic in
    ``nbn/bench/diagnostics/ve_profile_n20.py::_walk_elimination_shapes``,
    but tracks only ``(scope_set, has_b, cardinalities)`` triples and
    never allocates a tensor.

    The bare per-step walk returns the largest single union-scope
    factor.  That is *not* the realised peak: ``_log_factor_product_batched``
    holds ~3 union-scope tensors live at once and conditioned factors
    persist across steps, so the measured peak runs ~2.85× the bare walk
    (2026-06-08, nbn-neuralcat-ve, B=16..128).  The result is therefore
    scaled by ``_LIVESET_MULTIPLIER`` (#177) before return so the guard
    compares against a realistic peak; #178 tracks a principled live-set
    model that would replace the scalar.
    """
    return _peak_bytes_from_profile(
        _peak_profile(plan, factors, evidence_keys=evidence_keys),
        B, dtype_bytes,
    )


def _memory_budget_message(
    plan: list[str],
    factors: Dict[str, LogFactor],
    *,
    evidence_keys: tuple[str, ...],
    B: int,
    device: torch.device,
    order: str,
    safety: float = 0.9,
) -> str | None:
    """Return the pre-allocation guard's rejection message, or ``None``.

    The guard's *decision* as a pure helper (#177): the estimated peak
    (which includes the ``_LIVESET_MULTIPLIER``) versus ``safety`` × the
    device headroom (``nbn.inference._memory.device_headroom_bytes`` — on
    CUDA driver-free memory plus PyTorch's reusable cached blocks, #264).

    Returns ``None`` (query allowed) when the headroom is unknown or the
    estimate fits; otherwise the formatted guard message.
    """
    estimated_peak = _estimate_peak_bytes(
        plan, factors, evidence_keys=evidence_keys, B=B,
    )
    head = device_headroom_bytes(device)
    if head is None or estimated_peak <= safety * head:
        return None
    return _guard_message(estimated_peak, safety * head, device, order)


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
