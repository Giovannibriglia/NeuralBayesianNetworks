from __future__ import annotations

import logging
from typing import Dict, List

import torch

from nbn.core.factor import LogFactor
from nbn.inference.base import InferenceEngine

logger = logging.getLogger(__name__)


class TensorVariableElimination(InferenceEngine):
    """Log-domain einsum-based variable elimination for discrete BNs.

    This is the primary exact inference engine for purely discrete networks.
    It uses ``opt_einsum`` to find an optimal contraction path (equivalently,
    a good variable elimination order) and applies the log-sum-exp trick for
    numerical stability.

    Complexity is exponential in induced treewidth, so this engine is only
    used when the network is discrete and the treewidth is below a threshold
    (default 25).  For larger networks, fall back to ``LikelihoodWeightingEngine``.

    Parameters
    ----------
    treewidth_threshold: int
        Maximum induced treewidth to attempt exact VE.
    """

    def __init__(self, treewidth_threshold: int = 25) -> None:
        self.treewidth_threshold = treewidth_threshold
        self._factor_cache: Dict = {}

    def _extract_factors(self, model) -> Dict[str, LogFactor]:
        """Extract per-node CPT as a ``LogFactor``."""
        factors: Dict[str, LogFactor] = {}
        for node in model.dag.topological_order():
            mech = model.mechanisms[node]
            if not mech.is_discrete:
                raise ValueError(
                    f"TensorVariableElimination only supports discrete mechanisms; "
                    f"node '{node}' has a continuous mechanism."
                )
            if not hasattr(mech, '_logits') or mech._logits is None:
                raise RuntimeError(f"Mechanism for node '{node}' has not been fitted.")

            parents = model.dag.parents(node)
            var_scope = parents + [node]
            cards = {n: model.mechanisms[n].n_classes for n in var_scope}

            # mech._logits: [n_parent_states, K]; reshape to [*parent_cards, K]
            logits = mech._logits.detach()  # [prod(pa_cards), K]
            parent_cards = [model.mechanisms[p].n_classes for p in parents]
            if parent_cards:
                logits = logits.reshape(*parent_cards, cards[node])
            else:
                logits = logits.reshape(cards[node])

            log_cpt = torch.log_softmax(logits, dim=-1)
            factors[node] = LogFactor(log_cpt, var_scope, cards)
        return factors

    def query(
        self,
        model,
        targets: List[str],
        evidence: Dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Exact marginal inference via variable elimination.

        Parameters
        ----------
        targets: list with exactly one discrete node name.
        evidence: dict of observed class labels (int tensors of shape ``[1]`` or scalar).

        Returns
        -------
        torch.Tensor of shape ``[K]`` — normalised posterior over target classes.
        """
        if len(targets) != 1:
            raise NotImplementedError("TensorVE currently supports exactly one target node.")
        target = targets[0]
        evidence = evidence or {}
        device = model.device

        # Convert evidence to int scalars
        ev_int: Dict[str, int] = {
            k: int(v.item()) if isinstance(v, torch.Tensor) else int(v)
            for k, v in evidence.items()
        }

        factors = self._extract_factors(model)

        # Condition on evidence
        conditioned: List[LogFactor] = []
        for node, factor in factors.items():
            f = factor
            for ev_node, ev_val in ev_int.items():
                if ev_node in f.variables:
                    f = f.condition({ev_node: ev_val})
            conditioned.append(f)

        # Determine elimination order (all variables except target)
        all_vars = list(model.dag.topological_order())
        to_eliminate = [v for v in all_vars if v != target and v not in ev_int]

        for var in to_eliminate:
            # Collect all factors involving this variable
            relevant = [f for f in conditioned if var in f.variables]
            rest = [f for f in conditioned if var not in f.variables]

            if not relevant:
                continue

            # Multiply all relevant factors (add in log space)
            product = relevant[0]
            for f in relevant[1:]:
                product = _log_factor_product(product, f)

            # Marginalise out var
            product = product.marginalise(var)
            conditioned = rest + [product]

        # Multiply remaining factors (should only involve target now)
        if not conditioned:
            raise RuntimeError("No factors remaining after elimination.")

        result = conditioned[0]
        for f in conditioned[1:]:
            result = _log_factor_product(result, f)

        # Extract and normalise
        k = model.mechanisms[target].n_classes
        if target not in result.variables:
            # Constant factor — uniform posterior
            return torch.ones(k, device=device) / k

        # Reorder to put target last if needed
        if result.variables[-1] != target:
            dim = result.variables.index(target)
            perm = list(range(len(result.variables)))
            perm.append(perm.pop(dim))
            result = LogFactor(
                result.log_values.permute(*perm),
                [result.variables[p] for p in perm],
                result.cardinalities,
            )

        # Final tensor is 1-D [K]
        lv = result.log_values
        while lv.dim() > 1:
            lv = torch.logsumexp(lv, dim=0)

        probs = torch.softmax(lv, dim=0)
        return probs.to(device)

    def query_batch(
        self,
        model,
        targets: List[str],
        evidence: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Batched VE: processes each evidence row independently.

        For small treewidth networks this is fast; for large batches
        consider ``LikelihoodWeightingEngine`` which is GPU-native.
        """
        b = next(iter(evidence.values())).shape[0]
        rows = []
        for i in range(b):
            ev_i = {k: v[i] for k, v in evidence.items()}
            rows.append(self.query(model, targets, ev_i, **kwargs))
        return torch.stack(rows, dim=0)  # [B, K]


def _log_factor_product(a: LogFactor, b: LogFactor) -> LogFactor:
    """Multiply two log-factors (add log values, broadcast as needed)."""
    all_vars = list(dict.fromkeys(a.variables + b.variables))
    cards = {**a.cardinalities, **b.cardinalities}

    def _align(f: LogFactor) -> torch.Tensor:
        """Permute and unsqueeze f.log_values to align with all_vars axes."""
        lv = f.log_values
        present = [v for v in all_vars if v in f.variables]
        if present:
            perm = [f.variables.index(v) for v in present]
            lv = lv.permute(*perm)
        # Then insert missing dims
        full_lv = lv
        for i, v in enumerate(all_vars):
            if v not in f.variables:
                full_lv = full_lv.unsqueeze(i)
        return full_lv

    a_aligned = _align(a)
    b_aligned = _align(b)
    return LogFactor(a_aligned + b_aligned, all_vars, cards)
