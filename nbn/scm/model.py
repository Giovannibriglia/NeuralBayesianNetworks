"""``StructuralCausalModel``: an NBN whose mechanisms carry explicit noise.

On top of :class:`NeuralBayesianNetwork` it adds

* **noise tying** — :meth:`tie_noise` makes several structural mechanisms share
  one noise *law*; :meth:`fit` then pools them;
* **declared confounding** — ``bidirected=[(a, b)]`` records hidden common
  causes (shared noise *draws*); they are metadata for diagnostics and for
  downstream partial identification, not something NBN fits;
* **abduction and counterfactuals** — :meth:`abduct` and
  :meth:`counterfactual_sample` use the structural equations.

Everything else (``query``, ``do=``, ``sample``, inference engines, devices,
``save``/``load``) is inherited, with the SCM metadata carried along.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import torch

from nbn.core.network import NeuralBayesianNetwork
from nbn.scm.fitting import fit_offset_law
from nbn.scm.mechanisms import (
    AdditiveNoiseMechanism,
    DiscreteAdditiveNoiseMechanism,
    StructuralMechanism,
)
from nbn.scm.noise import GaussianMixtureNoise, GaussianNoise, OffsetNoise
from nbn.utils.batching import pack_parents
from nbn.utils.device import to_device


class StructuralCausalModel(NeuralBayesianNetwork):
    """A Bayesian network read as a structural causal model.

    Parameters
    ----------
    dag, variables, default_engine, device:
        As for :class:`NeuralBayesianNetwork`.
    bidirected:
        Pairs of nodes with a declared hidden common cause.
    context:
        Name of a dataset / regime / policy-id variable, if the data pool
        several regimes (used by ``context_invariance_test``).
    """

    def __init__(
        self,
        dag,
        variables,
        bidirected: Iterable[Tuple[str, str]] = (),
        context: str | None = None,
        default_engine: Any = "auto",
        device: str = "auto",
    ) -> None:
        super().__init__(dag, variables, default_engine=default_engine, device=device)
        self.bidirected: List[Tuple[str, str]] = [(a, b) for a, b in bidirected]
        for a, b in self.bidirected:
            for node in (a, b):
                if node not in self.variables:
                    raise ValueError(f"bidirected edge mentions unknown node '{node}'.")
        if context is not None and context not in self.variables:
            raise ValueError(f"context '{context}' is not a variable of the model.")
        self.context = context
        self.noise_groups: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Noise tying
    # ------------------------------------------------------------------

    def tie_noise(self, nodes: Sequence[str], name: str | None = None) -> str:
        """Make ``nodes`` share one noise law; return the group's name.

        All nodes must hold *unfitted* structural mechanisms of one family
        (all discrete-offset or all continuous-additive): tie first, then
        :meth:`fit`.  A node belongs to at most one group.
        """
        nodes = list(nodes)
        if len(nodes) < 2:
            raise ValueError("tie_noise needs at least two nodes.")
        mechs = [self._structural(node) for node in nodes]
        for node, mech in zip(nodes, mechs):
            for group, members in self.noise_groups.items():
                if node in members:
                    raise ValueError(f"'{node}' is already tied in group '{group}'.")
            if mech.is_fitted:
                raise ValueError(
                    f"'{node}' is already fitted; tie_noise would replace its fitted "
                    "noise law.  Tie before fitting, then call fit()."
                )
        shared = self._shared_noise(nodes, mechs).to(self.device)
        for mech in mechs:
            mech.noise = shared
        name = name or "+".join(nodes)
        self.noise_groups[name] = nodes
        self._cache_version += 1
        return name

    def _shared_noise(self, nodes, mechs):
        if all(isinstance(m, DiscreteAdditiveNoiseMechanism) for m in mechs):
            return OffsetNoise(max(
                m.offset_range(int(self.variables[n].cardinality))
                for n, m in zip(nodes, mechs)
                if isinstance(m, DiscreteAdditiveNoiseMechanism)
            ))
        if all(isinstance(m, AdditiveNoiseMechanism) for m in mechs):
            dims = {int(self.variables[n].dim) for n in nodes}
            if len(dims) != 1:
                raise ValueError(f"Tied continuous nodes must have one dimension, got {dims}.")
            first = mechs[0]
            dim = dims.pop()
            if first.noise_spec == "mixture":
                return GaussianMixtureNoise(dim, n_components=first.n_components)
            return GaussianNoise(dim)
        raise TypeError(
            "tie_noise needs mechanisms of one family: all "
            "DiscreteAdditiveNoiseMechanism or all AdditiveNoiseMechanism."
        )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, data: Dict[str, torch.Tensor], *, weights: torch.Tensor | None = None,
            **kwargs: Any):
        """Fit every mechanism (as NBN does), then refit each tied group jointly.

        Discrete groups refit their locations and the shared law together on
        count tables (cheap).  Continuous groups keep their fitted locations —
        least squares does not depend on the noise law — and refit only the
        shared law on the pooled residuals.  The returned history reports the
        log-likelihood of tied nodes *after* the joint refit.
        """
        history = super().fit(data, weights=weights, **kwargs)
        if not self.noise_groups:
            return history
        data = to_device(data, self.device)
        w = None if weights is None else torch.as_tensor(weights).to(self.device)
        with torch.no_grad():
            for nodes in self.noise_groups.values():
                mechs = [self._structural(node) for node in nodes]
                if isinstance(mechs[0], DiscreteAdditiveNoiseMechanism):
                    self._refit_offset_group(nodes, mechs, data, w)
                else:
                    pooled = torch.cat([
                        m.residual(data[n], self.parent_values(data, n)) for n, m in zip(nodes, mechs)
                    ])
                    mechs[0].noise.fit(pooled, None if w is None else w.repeat(len(nodes)))
                for node, mech in zip(nodes, mechs):
                    lp = mech.log_prob(data[node], self.parent_values(data, node))
                    history.node_log_likelihoods[node][-1] = float(lp.mean())
        self._cache_version += 1
        return history

    def _refit_offset_group(self, nodes, mechs, data, w) -> None:
        counts = [
            m.prepare_counts(
                data[node], self.parent_values(data, node),
                parent_cards=self._parent_cards(node),
                n_classes=int(self.variables[node].cardinality), weights=w,
            )
            for node, m in zip(nodes, mechs)
        ]
        locations, q, _ = fit_offset_law(
            [m.transfer() for m in mechs], counts, n_iter=mechs[0].n_iter, rtol=mechs[0].rtol
        )
        for m, loc in zip(mechs, locations):
            m.locations = loc
        mechs[0].noise.set_probs(q)
        for m, c in zip(mechs, counts):
            m.check_reachable(c)

    # ------------------------------------------------------------------
    # Structural queries
    # ------------------------------------------------------------------

    def parent_values(self, data: Mapping[str, torch.Tensor], node: str) -> torch.Tensor | None:
        """``[N, D_pa]`` parent values of ``node`` in DAG order (``None`` for roots)."""
        return pack_parents(dict(data), self.dag.parents(node))

    @torch.no_grad()
    def abduct(self, data: Mapping[str, torch.Tensor], n_samples: int = 1) -> Dict[str, torch.Tensor]:
        """Noise values of every structural node given a full observation.

        ``{node: [n_samples, N, D]}``: exact residuals for continuous nodes,
        posterior draws for discrete ones.  With every variable observed the
        noises of different nodes are independent given the data (tied nodes
        share a law, not a draw), so node-wise abduction is exact.
        """
        data = to_device(dict(data), self.device)
        return {
            node: mech.abduct(data[node], self.parent_values(data, node), n_samples)
            for node, mech in self.mechanisms.items()
            if isinstance(mech, StructuralMechanism)
        }

    @torch.no_grad()
    def counterfactual_sample(
        self,
        evidence: Mapping[str, torch.Tensor],
        do: Mapping[str, Any],
        n_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Draw ``X_do`` for the units in ``evidence`` (abduction–action–prediction).

        ``evidence`` observes every variable (``[B]`` or ``[B, D]`` each).
        Nodes that do not descend from a ``do`` target keep their observed
        value; for the others the noise is abducted from the evidence and the
        structural equations are re-run in topological order.  Only those
        descendants need structural mechanisms.  Discrete noise is drawn from
        its posterior, so the answer is a sample; continuous additive noise is
        recovered exactly.

        ``do`` values may be a scalar, a ``[D]`` vector (all units) or a
        ``[B, D]`` tensor (one value per unit; ``[B]`` for 1-D variables).

        Counterfactuals rest on the structural form chosen for each mechanism
        (data identify only ``P(X | pa)``): read them as answers *under that
        model*.

        Returns ``{node: [n_samples, B, D]}``.
        """
        if self.bidirected:
            raise NotImplementedError(
                "counterfactual_sample assumes independent noises; this model "
                "declares hidden common causes (bidirected edges)."
            )
        evidence = to_device(dict(evidence), self.device)
        batch = next(iter(evidence.values())).shape[0]
        affected = self._descendants_of(do)
        world: Dict[str, torch.Tensor] = {}
        for node in self.dag.topological_order():
            dim = int(self.variables[node].dim)
            if node in do:
                world[node] = self._do_value(do[node], batch, dim).expand(n_samples, -1, -1)
            elif node not in affected:
                observed = evidence[node].to(torch.float32).reshape(batch, -1)
                world[node] = observed.unsqueeze(0).expand(n_samples, -1, -1)
            else:
                mech = self._structural(node)
                noise = mech.abduct(evidence[node], self.parent_values(evidence, node), n_samples)
                parents = self.dag.parents(node)
                pa = torch.cat([world[p] for p in parents], -1).reshape(n_samples * batch, -1)
                value = mech.structural(pa, noise.reshape(n_samples * batch, -1))
                world[node] = value.to(torch.float32).reshape(n_samples, batch, -1)
        return world

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save as NBN does, plus the SCM metadata (tying, bidirected edges, context).

        Tied mechanisms are pickled together, so a reloaded model shares one
        noise module per group again.  (The checkpoint is written by NBN and
        then extended, so the SCM layer needs no change to NBN's core.)
        """
        super().save(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["scm"] = {
            "noise_groups": self.noise_groups,
            "bidirected": self.bidirected,
            "context": self.context,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> StructuralCausalModel:
        model = super().load(path, map_location=map_location)
        extras = torch.load(path, map_location=map_location, weights_only=False).get("scm", {})
        model.noise_groups = {k: list(v) for k, v in extras.get("noise_groups", {}).items()}
        model.bidirected = [(a, b) for a, b in extras.get("bidirected", [])]
        model.context = extras.get("context")
        return model  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _structural(self, node: str) -> StructuralMechanism:
        # nn.ModuleDict has no .get(); keep the explicit membership test.
        mech = self.mechanisms[node] if node in self.mechanisms else None  # noqa: SIM401
        if not isinstance(mech, StructuralMechanism):
            raise TypeError(
                f"Node '{node}' needs a structural mechanism (set_mechanism first), "
                f"got {type(mech).__name__}."
            )
        return mech

    def _parent_cards(self, node: str) -> List[int] | None:
        cards = [self.variables[p].cardinality for p in self.dag.parents(node)]
        return None if any(c is None for c in cards) else [int(c) for c in cards]

    def _descendants_of(self, targets: Iterable[str]) -> Set[str]:
        out: Set[str] = set()
        for t in targets:
            out.update(self.dag.descendants(t))
        return out - set(targets)

    def _do_value(self, value: Any, batch: int, dim: int) -> torch.Tensor:
        """A ``do`` value as ``[1, B, D]``: one value for all units, or one per unit."""
        t = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if t.dim() == 0:
            return t.expand(1, batch, dim)
        if t.dim() == 1 and dim == 1 and t.numel() == batch:
            return t.reshape(1, batch, 1)
        if t.dim() == 1 and t.numel() == dim:
            return t.reshape(1, 1, dim).expand(1, batch, dim)
        if t.dim() == 2 and tuple(t.shape) == (batch, dim):
            return t.unsqueeze(0)
        raise ValueError(
            f"do value of shape {tuple(t.shape)} fits neither one value ([{dim}]) "
            f"nor one per unit ([{batch}, {dim}])."
        )
