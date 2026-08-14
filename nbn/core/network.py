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
        # Nodes whose incoming edges have been severed by ``intervene()``.
        # Empty for an observational model; see ``intervene``.
        self._do_targets: set = set()
        # Monotonic counter bumped whenever the CPDs change (fit / update /
        # set_mechanism).  Inference engines memoise per-model factors and
        # plans; they key that memo on this counter so a refit can never be
        # served a stale posterior.  See ``TensorVariableElimination``.
        self._cache_version = 0
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
        self._cache_version += 1

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
        try:
            return _fit(
                self, data,
                method=method, epochs=epochs,
                batch_size=batch_size, lr=lr,
                consolidate=consolidate,
                device=str(self._device),
                **kwargs,
            )
        finally:
            # Bump even on failure: a partially-completed fit has already
            # mutated some CPDs, so any memoised factors are stale either way.
            self._cache_version += 1

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
        try:
            return _update(
                self, data,
                forgetting=forgetting,
                device=str(self._device),
                **kwargs,
            )
        finally:
            self._cache_version += 1

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
        do: Mapping[str, Any] | None = None,
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
        do:
            Dict node → interventional value, i.e. ``P(targets | do(...))``.
            The node's incoming edges are severed and its value clamped
            (Pearl's do-operator); unlike ``evidence`` it carries no
            likelihood.  Honoured by every engine.  Combine with ``evidence``
            to condition and intervene in the same query.

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

        def _norm(d):
            out: Dict[str, torch.Tensor] = {}
            for k, v in (d or {}).items():
                t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
                if t.dim() == 0:
                    t = t.unsqueeze(0)
                out[k] = t.to(self._device)
            return out

        ev = _norm(evidence)
        if do:
            kwargs["do"] = _norm(do)
        return eng.query(self, list(targets), ev, **kwargs)

    def query_batch(
        self,
        targets: Sequence[str],
        evidence: Mapping[str, torch.Tensor],
        engine: Any | None = None,
        do: Mapping[str, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Batched posterior inference.

        Parameters
        ----------
        targets:
            Target node names.
        evidence:
            Dict node → ``[B, D]`` tensor (or ``[B]`` for scalar nodes).
        do:
            Dict node → ``[B]`` interventional values.  The intervention may
            vary per row, so a whole dose-response sweep is one batched call.

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
        if do:
            kwargs["do"] = {k: v.to(self._device) for k, v in do.items()}
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
            dirac-gaussian per variable type).  One value per node — this
            path has no batch axis to vary the intervention along; use
            ``query_batch(do=...)`` for a per-row sweep.
        return_log_prob: if True return ``(samples, log_prob)``.

        Unlike ``intervene()``, this applies the intervention against the
        live parameters, so the returned samples stay differentiable w.r.t.
        the model's parameters.
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

        Implements Pearl's do-operator by *graph surgery*: every incoming edge
        to a do-target is removed from the returned model's DAG, and the
        target's CPD is replaced by a point mass at the intervened value.

        * discrete  → ``DeterministicMechanism`` (delta-Categorical at value)
        * continuous → ``DiracGaussianMechanism`` (tight Gaussian at value)

        The mutilated graph is the returned model's ``dag``, so every consumer
        — variable elimination, likelihood weighting, ancestral sampling —
        sees the intervention without needing to know about it.
        ``self._do_targets`` records which nodes were cut.

        Gradients
        ---------
        The returned model is a ``copy.deepcopy``, so its parameters are fresh
        leaf tensors: backpropagating through it does **not** reach the
        original model's parameters.  When you need ``d/dtheta`` of an
        interventional quantity, use ``model.sample(n, do=...)``, which
        applies the intervention in-place on the live parameters and stays
        differentiable.

        Interventions are single-valued per node.  To sweep a range of
        intervention values, loop — a batched do-value is rejected (the
        mutilated model has one CPD per node, not one per batch row).
        """
        import copy

        from nbn.core.dag import DAG
        from nbn.mechanisms.parametric.deterministic import DeterministicMechanism
        from nbn.mechanisms.parametric.dirac_gaussian import DiracGaussianMechanism

        unknown = [n for n in do if n not in self.variables]
        if unknown:
            raise ValueError(f"Unknown intervention target(s): {sorted(unknown)}.")

        new_model = copy.deepcopy(self)
        targets = set(do.keys())
        # Graph surgery: drop every edge INTO a do-target.  Previously the
        # edges survived and only the CPD was swapped, which left the
        # mutilation implicit — correct for samplers (the delta CPD ignores
        # its parents) but wrong for variable elimination, which builds a
        # factor over ``parents + [node]`` and needs the severed scope.
        # Rebuild from ``ordered_edges()``, which preserves every *untouched*
        # node's parent order.  Filtering ``edges()`` instead would silently
        # permute some other node's parent list, transposing its CPT axes —
        # see ``DAG.ordered_edges``.
        kept_edges = [
            (u, v) for (u, v) in self.dag.ordered_edges() if v not in targets
        ]
        new_model.dag = DAG(kept_edges, extra_nodes=list(self.dag.nodes()))
        new_model._do_targets = set(targets)

        for node, val in do.items():
            var = new_model.variables[node]
            val_t = val if isinstance(val, torch.Tensor) else torch.tensor(val, dtype=torch.float)
            val_t = val_t.reshape(-1) if val_t.dim() == 0 else val_t
            if val_t.dim() > 1:
                raise ValueError(
                    f"Batched intervention value for '{node}' (shape "
                    f"{tuple(val_t.shape)}); intervene() applies one value per "
                    f"node.  Loop over the values, or pass do= to "
                    f"query()/query_batch(), which do accept a batch."
                )
            val_t = val_t.to(new_model._device)
            if var.is_discrete:
                card = var.cardinality
                idx = int(val_t.reshape(-1)[0].item())
                if card is not None and not 0 <= idx < int(card):
                    raise ValueError(
                        f"Intervention value {idx} for discrete node '{node}' "
                        f"is outside its {card} declared states."
                    )
                mech: Mechanism = DeterministicMechanism(
                    val_t.to(torch.float), cardinality=card,
                )
            else:
                mech = DiracGaussianMechanism(val_t, output_dim=var.dim)
            mech.to(new_model._device)
            new_model.mechanisms[node] = mech
        # set_mechanism is bypassed above (mechanisms are written directly), so
        # bump explicitly: the returned model's CPDs differ from the copy's.
        new_model._cache_version += 1
        new_model._engine = None
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

    # Checkpoint format version.  1 = graph + variables + state_dict only
    # (mechanisms were NOT restorable); 2 = adds the fitted mechanism modules
    # themselves, so ``load`` round-trips.
    _CHECKPOINT_FORMAT = 2

    def save(self, path: str) -> None:
        """Save model to a ``.pt`` checkpoint.

        Stores the graph, the variable specs (including declared
        cardinalities) and the *fitted mechanism modules*, so ``load`` returns
        a directly queryable model.

        The mechanisms are pickled by ``torch.save`` rather than reduced to a
        ``state_dict``: most mechanisms build their parameters lazily inside
        ``fit_local`` (``LinearGaussianMechanism._weight``,
        ``MDNMechanism.net``, ``NormalizingFlowMechanism._flow`` are all
        ``None`` until then), so a freshly constructed mechanism has no
        parameter to load a ``state_dict`` into.  Reconstructing them would
        require every mechanism to serialise its own hyperparameters — the
        pickle carries that for free.  A ``state_dict`` is still written for
        inspection and backward compatibility with format-1 readers.

        Loading a checkpoint therefore executes the pickled classes' code:
        load only checkpoints you trust, as with any ``torch.save`` of a
        module.
        """
        payload = {
            "format": self._CHECKPOINT_FORMAT,
            # ordered_edges(), not edges(): rebuilding from edges() can
            # permute a node's parent list and transpose its CPT axes.
            "dag_edges": self.dag.ordered_edges(),
            "dag_nodes": self.dag.nodes(),
            "variables": {
                n: (v.kind, v.dim, v.cardinality)
                for n, v in self.variables.items()
            },
            "state_dict": self.state_dict(),
            "mechanism_types": {
                n: type(m).__name__ for n, m in self.mechanisms.items()
            },
            "mechanisms": dict(self.mechanisms.items()),
            "do_targets": sorted(self._do_targets),
            "engine_spec": self._engine_spec if isinstance(self._engine_spec, str) else None,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> NeuralBayesianNetwork:
        """Load a model saved with ``save()``.

        Returns a model with its mechanisms re-registered and ready to query.
        A format-1 checkpoint (written before mechanisms were persisted) has
        no mechanisms to restore; it loads with an empty mechanism registry
        and a warning, exactly as it always did.
        """
        payload = torch.load(path, map_location=map_location, weights_only=False)
        edges = payload["dag_edges"]
        nodes = payload["dag_nodes"]
        # Rebuild full Variable specs — the cardinality (third element) was
        # previously dropped on load, silently downgrading declared discrete
        # ranges to whatever ``dim`` happened to be.
        from nbn.core.variables import ContinuousVariable, DiscreteVariable
        variables: Dict[str, Variable] = {}
        for n, (kind, dim, card) in payload["variables"].items():
            if kind == "discrete":
                variables[n] = DiscreteVariable(n, cardinality=card)
            else:
                variables[n] = ContinuousVariable(n, dim=dim)

        from nbn.core.dag import DAG
        dag = DAG(edges, extra_nodes=nodes)
        engine_spec = payload.get("engine_spec") or "auto"
        model = cls(dag, variables, default_engine=engine_spec, device=str(map_location))

        mechanisms = payload.get("mechanisms")
        if mechanisms is None:
            logger.warning(
                "Checkpoint '%s' predates format %d and carries no mechanism "
                "modules; the loaded model has an empty mechanism registry and "
                "must have them re-registered before it can be queried.",
                path, cls._CHECKPOINT_FORMAT,
            )
            return model
        for node, mech in mechanisms.items():
            model.set_mechanism(node, mech)
        model._do_targets = set(payload.get("do_targets", ()))
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
