from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import torch
import torch.nn as nn

from nbn.core.dag import DAG
from nbn.core.variables import Variable
from nbn.mechanisms.base import Mechanism
from nbn.mechanisms.parametric.categorical_table import CategoricalTableMechanism
from nbn.mechanisms.parametric.linear_gaussian import LinearGaussianMechanism
from nbn.mechanisms.parametric.mdn import MDNMechanism
from nbn.utils.device import resolve_device, to_device

logger = logging.getLogger(__name__)


class _AliasModuleDict(nn.ModuleDict):
    """An ``nn.ModuleDict`` that tolerates node names containing ``.``.

    ``nn.ModuleDict`` keys map to submodule names, which forbid ``.`` (it would
    be read as attribute nesting), so a bnlearn node like ``pm2.5`` or
    ``YR.GLASS`` crashes registration. This subclass transparently stores keys
    with ``.`` replaced by a reserved token and reverses the substitution on
    read, so every call site keeps using the original node name unchanged.
    The transform is a pure, deterministic, reversible string op (no side map
    to keep in sync across deepcopy / load_state_dict).
    """

    _TOKEN = "__DOT__"

    @classmethod
    def _safe(cls, key: str) -> str:
        return key.replace(".", cls._TOKEN)

    @classmethod
    def _orig(cls, key: str) -> str:
        return key.replace(cls._TOKEN, ".")

    def __setitem__(self, key, module):
        super().__setitem__(self._safe(key), module)

    def __getitem__(self, key):
        return super().__getitem__(self._safe(key))

    def __delitem__(self, key):
        super().__delitem__(self._safe(key))

    def __contains__(self, key):
        return super().__contains__(self._safe(key))

    def __iter__(self):
        return (self._orig(k) for k in super().__iter__())

    def keys(self):
        return list(self)  # __iter__ already yields original (de-aliased) names

    def items(self):
        return [(self._orig(k), v) for k, v in super().items()]


class NeuralBayesianNetwork(nn.Module):
    """A Bayesian Network with a *given* DAG and learnable, GPU-resident,
    autograd-compatible per-node neural mechanisms.

    Design goals (mirroring GPyTorch):
    * Every CPD is a ``Mechanism`` (``nn.Module`` subclass).
    * ``mechanism.forward(parents)`` returns a ``torch.distributions.Distribution``.
    * The whole network is itself an ``nn.Module``, so ``.to(device)``,
      ``.parameters()``, ``.state_dict()``, ``torch.compile``, and AMP work.

    Parameters
    ----------
    dag:
        Either a list of ``(parent, child)`` edge tuples or a
        ``networkx.DiGraph``.
    variables:
        Dict mapping node name → ``("discrete"|"continuous", dim)`` spec or
        a ``Variable`` instance.
    default_engine:
        Inference engine to use for ``query()``/``query_batch()``.
        ``"auto"`` selects ``HybridRouter``.

    Examples
    --------
    >>> from nbn import NeuralBayesianNetwork as NBN
    >>> from nbn.mechanisms import CategoricalTableMechanism, MDNMechanism
    >>> model = NBN(
    ...     [("A", "C"), ("B", "C"), ("C", "D")],
    ...     variables={"A": ("discrete", 2), "B": ("discrete", 3),
    ...                "C": ("discrete", 4), "D": ("continuous", 1)},
    ... )
    >>> model.set_mechanism("A", CategoricalTableMechanism())
    >>> model.fit(data, device="cuda")
    >>> probs = model.query(["C"], evidence={"A": torch.tensor([0])})
    """

    def __init__(
        self,
        dag: Union[List[Tuple[str, str]], Any],
        variables: Mapping[str, Union[Tuple, Variable]],
        default_engine: Union[str, Any] = "auto",
        device: str = "auto",
    ) -> None:
        super().__init__()

        # Build DAG
        if isinstance(dag, DAG):
            self.dag = dag
        else:
            self.dag = DAG(dag, extra_nodes=list(variables.keys()))

        # Build variable specs
        self.variables: Dict[str, Variable] = {}
        for name, spec in variables.items():
            if isinstance(spec, Variable):
                self.variables[name] = spec
            else:
                kind, dim = spec
                from nbn.core.variables import ContinuousVariable, DiscreteVariable
                if kind == "discrete":
                    self.variables[name] = DiscreteVariable(name, cardinality=dim)
                else:
                    self.variables[name] = ContinuousVariable(name, dim=dim)

        missing = set(self.dag.nodes()) - set(self.variables)
        if missing:
            raise ValueError(f"Variables spec missing nodes: {sorted(missing)}")

        # Mechanism registry (populated by set_mechanism / auto_mechanisms / fit).
        # _AliasModuleDict tolerates dotted node names (e.g. bnlearn's pm2.5).
        self.mechanisms: nn.ModuleDict = _AliasModuleDict()

        self._engine_spec = default_engine
        self._engine = None
        self._device = resolve_device(device)
        self._mixed_precision = False

    # ------------------------------------------------------------------
    # Mechanism registration
    # ------------------------------------------------------------------

    def set_mechanism(self, node: str, mech: Mechanism) -> None:
        """Register a mechanism for a node and move it to the model's device."""
        if node not in self.variables:
            raise ValueError(f"Unknown node '{node}'.")
        mech.to(self._device)
        self.mechanisms[node] = mech

    def auto_mechanisms(
        self,
        default_discrete: str = "categorical_table",
        default_continuous: str = "mdn",
        mdn_components: int = 5,
    ) -> None:
        """Assign default mechanisms based on variable types (each on model device)."""
        for node, var in self.variables.items():
            if node in self.mechanisms:
                continue
            if var.is_discrete:
                if default_discrete == "categorical_table":
                    mech: Mechanism = CategoricalTableMechanism()
                else:
                    from nbn.mechanisms.parametric.neural_categorical import NeuralCategoricalMechanism
                    mech = NeuralCategoricalMechanism(n_classes=var.cardinality or 2)
            else:
                if default_continuous == "mdn":
                    mech = MDNMechanism(num_components=mdn_components)
                else:
                    mech = LinearGaussianMechanism()
            self.set_mechanism(node, mech)

    def set_mixed_precision(self, enabled: bool) -> None:
        """Toggle ``torch.amp.autocast`` for forward passes (default: off)."""
        self._mixed_precision = bool(enabled)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        data: Dict[str, torch.Tensor],
        *,
        method: str = "local",
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
        consolidate: bool = True,
        **kwargs: Any,
    ):
        """Fit all node mechanisms to data on the model's device.

        Parameters
        ----------
        data:
            Dict of node_name → tensor ``[N, D]`` or ``[N]``. Auto-moved.
        method:
            ``"local"`` (node-wise, default) or ``"joint"`` (shared optimiser).
        epochs, batch_size, lr:
            Training hyperparameters. ``None`` (default) = each mechanism
            uses its own designed budget (flow 300 epochs @ lr 5e-4, MDN 200,
            neural-categorical 100, ...); explicit values override globally
            for every mechanism.
        consolidate:
            If True (default), neural mechanisms snapshot post-fit EWC state
            so ``update()`` works later; False skips the Fisher pass for
            fit-only workloads (``update()`` then raises until refit with
            ``consolidate=True``).

        Returns
        -------
        TrainHistory
        """
        if "device" in kwargs:
            raise TypeError(
                "device is set at NeuralBayesianNetwork construction time; "
                "use model.to(new_device) to move."
            )
        from nbn.learning.fit import fit as _fit
        data = to_device(data, self._device)
        return _fit(
            self, data,
            method=method, epochs=epochs,
            batch_size=batch_size, lr=lr,
            consolidate=consolidate,
            device=str(self._device),
            **kwargs,
        )

    def update(
        self,
        data: Dict[str, torch.Tensor],
        *,
        forgetting: float = 1.0,
        **kwargs: Any,
    ):
        """Fold new data into the fitted CPDs without retraining or rehearsal.

        The graph and cardinalities are fixed; only CPD parameters change.
        Each mechanism that supports it (set via ``supports_update``) folds in
        the new data using sufficient statistics it persisted during
        ``fit`` — the posterior left by ``fit`` becomes the prior for
        ``update`` (recursive Bayes).  Mechanisms that don't support it are
        skipped (recorded in the returned ``UpdateHistory``).

        Parameters
        ----------
        data:
            Dict of node_name → tensor ``[N, D]`` or ``[N]``. Auto-moved.
        forgetting:
            Exponential-forgetting factor in ``(0, 1]`` applied to each
            mechanism's persisted prior state (``1.0`` = pure recursive Bayes).

        Returns
        -------
        UpdateHistory
        """
        if "device" in kwargs:
            raise TypeError(
                "device is set at NeuralBayesianNetwork construction time; "
                "use model.to(new_device) to move."
            )
        from nbn.update.orchestrate import update as _update
        data = to_device(data, self._device)
        return _update(
            self, data,
            forgetting=forgetting,
            device=str(self._device),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _get_engine(self):
        if self._engine is not None:
            return self._engine
        if self._engine_spec == "auto" or self._engine_spec is None:
            from nbn.inference.hybrid import HybridRouter
            self._engine = HybridRouter()
        elif isinstance(self._engine_spec, str):
            eng_map = {
                "likelihood_weighting": "nbn.inference.likelihood_weighting.LikelihoodWeightingEngine",
                "tensor_ve": "nbn.inference.tensor_ve.TensorVariableElimination",
                "hybrid": "nbn.inference.hybrid.HybridRouter",
            }
            if self._engine_spec not in eng_map:
                raise ValueError(f"Unknown engine '{self._engine_spec}'. Available: {list(eng_map)}")
            module_path, cls_name = eng_map[self._engine_spec].rsplit(".", 1)
            import importlib
            mod = importlib.import_module(module_path)
            self._engine = getattr(mod, cls_name)()
        else:
            self._engine = self._engine_spec
        return self._engine

    def query(
        self,
        targets: Sequence[str],
        evidence: Mapping[str, Any] | None = None,
        engine: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Posterior inference for ``targets`` given ``evidence``.

        Parameters
        ----------
        targets:
            List of target node names.
        evidence:
            Dict node → observed value (int scalar / tensor ``[D]``).
        engine:
            Override inference engine for this call.

        Returns
        -------
        torch.Tensor
            For a single discrete target: ``[K]`` probability vector.
            For continuous or multi-target: ``(weights, samples)`` tuple.
        """
        if "device" in kwargs:
            raise TypeError(
                "device is set at NeuralBayesianNetwork construction time; "
                "use model.to(new_device) to move."
            )
        eng = engine if engine is not None else self._get_engine()
        ev: Dict[str, torch.Tensor] = {}
        for k, v in (evidence or {}).items():
            t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
            if t.dim() == 0:
                t = t.unsqueeze(0)
            ev[k] = t.to(self._device)
        return eng.query(self, list(targets), ev, **kwargs)

    def query_batch(
        self,
        targets: Sequence[str],
        evidence: Mapping[str, torch.Tensor],
        engine: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Batched posterior inference.

        Parameters
        ----------
        targets:
            Target node names.
        evidence:
            Dict node → ``[B, D]`` tensor (or ``[B]`` for scalar nodes).

        Returns
        -------
        torch.Tensor of shape ``[B, K]`` for discrete targets.
        """
        if "device" in kwargs:
            raise TypeError(
                "device is set at NeuralBayesianNetwork construction time; "
                "use model.to(new_device) to move."
            )
        eng = engine if engine is not None else self._get_engine()
        ev = {k: v.to(self._device) for k, v in evidence.items()}
        return eng.query_batch(self, list(targets), ev, **kwargs)

    def map_query(
        self,
        targets: Sequence[str],
        evidence: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """MAP query: return argmax of the posterior for each target."""
        result = self.query(targets, evidence, **kwargs)
        if isinstance(result, torch.Tensor) and result.dim() == 1:
            return {targets[0]: result.argmax()}
        raise NotImplementedError("MAP query for non-discrete / multi-target not implemented.")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        n: int = 1,
        evidence: Dict[str, torch.Tensor] | None = None,
        do: Mapping[str, torch.Tensor] | None = None,
        return_log_prob: bool = False,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Draw ``n`` joint samples on the model's device.

        Parameters
        ----------
        n: number of samples.
        evidence: clamped observed values (auto-moved to model device).
        do: do-intervention values (auto-dispatched to deterministic /
            dirac-gaussian per variable type).
        return_log_prob: if True return ``(samples, log_prob)``.
        """
        if "device" in kwargs:
            raise TypeError(
                "device is set at NeuralBayesianNetwork construction time; "
                "use model.to(new_device) to move."
            )
        from nbn.sampling.ancestral import ancestral_sample
        evidence = to_device(evidence, self._device) if evidence else None
        do = to_device(do, self._device) if do else None
        return ancestral_sample(
            self, n=n, evidence=evidence, do=do,
            device=str(self._device),
            return_log_prob=return_log_prob,
        )

    # ------------------------------------------------------------------
    # Causal extensions
    # ------------------------------------------------------------------

    def intervene(self, do: Mapping[str, Any]) -> NeuralBayesianNetwork:
        """Return a deep-copied NBN with do-interventions applied.

        Dispatches by variable type:
        * discrete  → ``DeterministicMechanism`` (delta-Categorical at value)
        * continuous → ``DiracGaussianMechanism`` (tight Gaussian at value)

        The mutilated graph (incoming edges to do-targets removed) is reflected
        in the working DAG used by inference engines via ``self._do_targets``.
        """
        import copy

        from nbn.mechanisms.parametric.deterministic import DeterministicMechanism
        from nbn.mechanisms.parametric.dirac_gaussian import DiracGaussianMechanism

        new_model = copy.deepcopy(self)
        new_model._do_targets = set(do.keys())
        for node, val in do.items():
            if node not in new_model.variables:
                raise ValueError(f"Unknown intervention target '{node}'.")
            var = new_model.variables[node]
            val_t = val if isinstance(val, torch.Tensor) else torch.tensor(val, dtype=torch.float)
            val_t = val_t.to(new_model._device)
            if var.is_discrete:
                mech: Mechanism = DeterministicMechanism(val_t)
            else:
                mech = DiracGaussianMechanism(val_t, output_dim=var.dim)
            mech.to(new_model._device)
            new_model.mechanisms[node] = mech
        return new_model

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, *args, **kwargs) -> NeuralBayesianNetwork:
        """Move model + every registered mechanism + cached engine to a device."""
        super().to(*args, **kwargs)
        # Update cached device
        new_device: torch.device | None = None
        if args and isinstance(args[0], (str, torch.device)):
            new_device = resolve_device(args[0])
        elif "device" in kwargs:
            new_device = resolve_device(kwargs["device"])
        else:
            try:
                new_device = next(self.parameters()).device
            except StopIteration:
                pass
        if new_device is not None:
            self._device = new_device
        # Propagate to a cached engine (clear so next call rebuilds on new device)
        self._engine = None
        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model to a ``.pt`` checkpoint."""
        payload = {
            "dag_edges": self.dag.edges(),
            "dag_nodes": self.dag.nodes(),
            "variables": {
                n: (v.kind, v.dim, v.cardinality)
                for n, v in self.variables.items()
            },
            "state_dict": self.state_dict(),
            "mechanism_types": {
                n: type(m).__name__ for n, m in self.mechanisms.items()
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> NeuralBayesianNetwork:
        """Load a model saved with ``save()``."""
        payload = torch.load(path, map_location=map_location, weights_only=False)
        edges = payload["dag_edges"]
        nodes = payload["dag_nodes"]
        variables = {
            n: (kind, dim) for n, (kind, dim, _) in payload["variables"].items()
        }
        model = cls(edges, variables)
        # We cannot reconstruct mechanism hyperparams without saving them;
        # this loads just state_dict — full round-trip requires mechanism classes.
        # For now we delegate to the caller to re-register mechanisms.
        return model

    @classmethod
    def from_bif(cls, path: str) -> NeuralBayesianNetwork:
        """Load a discrete BN from a .bif file (uses pgmpy if available)."""
        try:
            from pgmpy.readwrite import BIFReader
            reader = BIFReader(path)
            model_pgmpy = reader.get_model()
        except ImportError as e:
            raise ImportError("from_bif requires pgmpy: pip install pgmpy") from e

        edges = list(model_pgmpy.edges())
        variables = {}
        for node in model_pgmpy.nodes():
            card = len(model_pgmpy.get_cpds(node).state_names[node])
            variables[node] = ("discrete", card)
        nbn_model = cls(edges, variables)

        # Install fitted CategoricalTableMechanisms from the pgmpy CPDs
        import numpy as np
        for node in model_pgmpy.topological_order:
            cpd = model_pgmpy.get_cpds(node)
            mech = CategoricalTableMechanism()
            parents = list(model_pgmpy.get_parents(node))
            k = cpd.variable_card
            # Build dummy data for fit_local to infer cardinalities
            # Use the CPT directly instead
            parent_cards = [cpd.evidence_card[cpd.evidence.index(p)] for p in parents] if parents else []
            n_parent_states = int(np.prod(parent_cards)) if parent_cards else 1

            log_cpt = torch.log(
                torch.tensor(cpd.values.reshape(k, n_parent_states).T, dtype=torch.float).clamp_min(1e-9)
            )
            mech._logits = nn.Parameter(log_cpt)
            mech._n_classes = k
            mech._parent_cards = parent_cards
            strides = []
            stride = 1
            for c in reversed(parent_cards):
                strides.append(stride)
                stride *= c
            mech._parent_strides = list(reversed(strides))
            mech._class_values = torch.arange(k, dtype=torch.float)
            mech.output_dim = 1
            nbn_model.mechanisms[node] = mech

        return nbn_model

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self.dag.nodes())
        e = len(self.dag.edges())
        fitted = len(self.mechanisms)
        return (
            f"NeuralBayesianNetwork("
            f"nodes={n}, edges={e}, fitted_mechanisms={fitted}, "
            f"device={self._device})"
        )
