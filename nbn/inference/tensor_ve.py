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
from typing import Dict, List

import torch

from nbn.core.factor import LogFactor
from nbn.inference._elimination_order import get_order
from nbn.inference.base import InferenceEngine

logger = logging.getLogger(__name__)


class TensorVariableElimination(InferenceEngine):
    """Log-domain VE for discrete BNs (single-row + truly batched queries)."""

    def __init__(self, treewidth_threshold: int = 25) -> None:
        self.treewidth_threshold = treewidth_threshold
        # Memoise factor build by id(model). Cleared on `invalidate_cache`.
        self._factor_cache: Dict[int, Dict[str, LogFactor]] = {}
        # Memoise the elimination order per (id(model), targets, evidence_keys).
        self._plan_cache: Dict[tuple, list[str]] = {}

    def invalidate_cache(self, model=None) -> None:
        if model is None:
            self._factor_cache.clear(); self._plan_cache.clear()
            return
        self._factor_cache.pop(id(model), None)
        for k in list(self._plan_cache):
            if k[0] == id(model):
                self._plan_cache.pop(k, None)

    # ------------------------------------------------------------------ #
    # Factor and plan extraction (cached)
    # ------------------------------------------------------------------ #

    def _extract_factors(self, model) -> Dict[str, LogFactor]:
        cached = self._factor_cache.get(id(model))
        if cached is not None:
            return cached
        factors: Dict[str, LogFactor] = {}
        for node in model.dag.topological_order():
            mech = model.mechanisms[node]
            if not mech.is_discrete:
                raise ValueError(
                    f"TensorVariableElimination only supports discrete mechanisms; "
                    f"node '{node}' has a continuous mechanism."
                )
            if not hasattr(mech, "_logits") or mech._logits is None:
                raise RuntimeError(f"Mechanism for node '{node}' has not been fitted.")
            parents = model.dag.parents(node)
            var_scope = parents + [node]
            cards = {n: model.mechanisms[n].n_classes for n in var_scope}
            logits = mech._logits.detach()
            parent_cards = [model.mechanisms[p].n_classes for p in parents]
            if parent_cards:
                logits = logits.reshape(*parent_cards, cards[node])
            else:
                logits = logits.reshape(cards[node])
            log_cpt = torch.log_softmax(logits, dim=-1)
            factors[node] = LogFactor(log_cpt, var_scope, cards)
        self._factor_cache[id(model)] = factors
        return factors

    def _plan(
        self,
        model,
        target: str,
        evidence_keys: tuple[str, ...],
        *,
        order: str = "min_fill",
    ) -> list[str]:
        """Compute (or look up cached) elimination plan.

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
        query under two orders does *not* invalidate either entry.
        """
        key = (id(model), target, evidence_keys, order)
        cached = self._plan_cache.get(key)
        if cached is not None:
            return cached
        elimination_order = get_order(
            order,
            model.dag.networkx_graph,
            targets=[target],
            evidence=list(evidence_keys),
        )
        self._plan_cache[key] = elimination_order
        return elimination_order

    # ------------------------------------------------------------------ #
    # Single-row query (kept identical to the v0.2 path)
    # ------------------------------------------------------------------ #

    def query(
        self,
        model,
        targets: List[str],
        evidence: Dict[str, torch.Tensor] | None = None,
        *,
        order: str = "min_fill",
        **kwargs,
    ) -> torch.Tensor:
        if len(targets) != 1:
            raise NotImplementedError("TensorVE supports exactly one target.")
        target = targets[0]
        evidence = evidence or {}
        device = model.device

        ev_int: Dict[str, int] = {
            k: int(v.item()) if isinstance(v, torch.Tensor) else int(v)
            for k, v in evidence.items()
        }
        factors = self._extract_factors(model)

        # Condition each factor on evidence
        conditioned: list[LogFactor] = []
        for _node, factor in factors.items():
            f = factor
            for ev_node, ev_val in ev_int.items():
                if ev_node in f.variables:
                    f = f.condition({ev_node: ev_val})
            conditioned.append(f)

        to_eliminate = self._plan(
            model, target, tuple(sorted(ev_int.keys())), order=order,
        )
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
        device = model.device

        # Normalise evidence to [B] long tensors on `device`.
        ev_norm: Dict[str, torch.Tensor] = {}
        B = 1
        for k, v in evidence.items():
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            if t.dim() == 0:
                t = t.reshape(1)
            elif t.dim() >= 2 and t.shape[-1] == 1:
                t = t.reshape(-1)
            t = t.to(device=device, dtype=torch.long)
            ev_norm[k] = t
            B = max(B, t.shape[0])
        # Broadcast any [1] evidence to [B]
        ev_norm = {k: (v.expand(B) if v.shape[0] == 1 else v) for k, v in ev_norm.items()}

        factors = self._extract_factors(model)

        # Condition each factor batchwise.
        # `conditioned` holds tuples `(log_values_tensor, vars, has_batch)`.
        conditioned: list[tuple[torch.Tensor, list[str], bool]] = []
        for f in factors.values():
            lv, vars_, has_b = _condition_factor_batched(
                f.log_values, f.variables, ev_norm,
            )
            conditioned.append((lv, vars_, has_b))

        to_eliminate = self._plan(
            model, target, tuple(sorted(ev_norm.keys())), order=order,
        )
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
