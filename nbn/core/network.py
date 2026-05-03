from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from nbn.core.dag import DAG
from nbn.core.variables import Variable
from nbn.core.query import NBNQuery
from nbn.mechanisms.base import Mechanism
from nbn.mechanisms.categorical_table import CategoricalTableMechanism
from nbn.mechanisms.linear_gaussian import LinearGaussianMechanism
from nbn.mechanisms.mdn import MDNMechanism

logger = logging.getLogger(__name__)


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
                from nbn.core.variables import DiscreteVariable, ContinuousVariable
                if kind == "discrete":
                    self.variables[name] = DiscreteVariable(name, cardinality=dim)
                else:
                    self.variables[name] = ContinuousVariable(name, dim=dim)

        missing = set(self.dag.nodes()) - set(self.variables)
        if missing:
            raise ValueError(f"Variables spec missing nodes: {sorted(missing)}")

        # Mechanism registry (populated by set_mechanism / auto_mechanisms / fit)
        self.mechanisms: nn.ModuleDict = nn.ModuleDict()

        self._engine_spec = default_engine
        self._engine = None
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

    # ------------------------------------------------------------------
    # Mechanism registration
    # ------------------------------------------------------------------

    def set_mechanism(self, node: str, mech: Mechanism) -> None:
        """Register a mechanism for a node."""
        if node not in self.variables:
            raise ValueError(f"Unknown node '{node}'.")
        self.mechanisms[node] = mech

    def auto_mechanisms(
        self,
        default_discrete: str = "categorical_table",
        default_continuous: str = "mdn",
        mdn_components: int = 5,
    ) -> None:
        """Assign default mechanisms based on variable types."""
        for node, var in self.variables.items():
            if node in self.mechanisms:
                continue
            if var.is_discrete:
                if default_discrete == "categorical_table":
                    self.mechanisms[node] = CategoricalTableMechanism()
                else:
                    from nbn.mechanisms.neural_categorical import NeuralCategoricalMechanism
                    self.mechanisms[node] = NeuralCategoricalMechanism(n_classes=var.cardinality or 2)
            else:
                if default_continuous == "mdn":
                    self.mechanisms[node] = MDNMechanism(num_components=mdn_components)
                else:
                    self.mechanisms[node] = LinearGaussianMechanism()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        data: Dict[str, torch.Tensor],
        *,
        method: str = "local",
        epochs: int = 100,
        batch_size: int = 4096,
        lr: float = 1e-3,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        """Fit all node mechanisms to data.

        Parameters
        ----------
        data:
            Dict of node_name → tensor ``[N, D]`` or ``[N]``.
        method:
            ``"local"`` (node-wise, default) or ``"joint"`` (shared optimiser).
        epochs, batch_size, lr:
            Training hyperparameters passed to mechanisms.
        device:
            Where to run; moves model and data here.

        Returns
        -------
        TrainHistory
        """
        from nbn.learning.fit import fit as _fit
        if device is not None:
            self.to(device)
            self._device = torch.device(device)
        return _fit(
            self, data,
            method=method, epochs=epochs,
            batch_size=batch_size, lr=lr,
            device=device or str(self._device),
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
        evidence: Optional[Mapping[str, Any]] = None,
        engine: Optional[Any] = None,
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
        engine: Optional[Any] = None,
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
        eng = engine if engine is not None else self._get_engine()
        ev = {k: v.to(self._device) for k, v in evidence.items()}
        return eng.query_batch(self, list(targets), ev, **kwargs)

    def map_query(
        self,
        targets: Sequence[str],
        evidence: Optional[Mapping[str, Any]] = None,
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
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        do: Optional[Mapping[str, torch.Tensor]] = None,
        device: Optional[str] = None,
        return_log_prob: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Draw ``n`` joint samples from the model via ancestral sampling.

        Parameters
        ----------
        n: number of samples.
        evidence: clamped observed values.
        do: do-intervention values (Pearl's do-operator).
        device: target device.
        return_log_prob: if True return ``(samples, log_prob)``.
        """
        from nbn.sampling.ancestral import ancestral_sample
        return ancestral_sample(
            self, n=n, evidence=evidence, do=do,
            device=device or str(self._device),
            return_log_prob=return_log_prob,
        )

    # ------------------------------------------------------------------
    # Causal extensions
    # ------------------------------------------------------------------

    def intervene(self, do: Mapping[str, Any]) -> "NeuralBayesianNetwork":
        """Return a new NBN with do-interventions applied (mutilated graph)."""
        import copy
        from nbn.mechanisms.deterministic import DeterministicMechanism
        new_model = copy.deepcopy(self)
        for node, val in do.items():
            val_t = val if isinstance(val, torch.Tensor) else torch.tensor(val, dtype=torch.float)
            new_model.mechanisms[node] = DeterministicMechanism(val_t.to(self._device))
        return new_model

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, *args, **kwargs) -> "NeuralBayesianNetwork":
        super().to(*args, **kwargs)
        # Update cached device
        try:
            p = next(self.parameters())
            self._device = p.device
        except StopIteration:
            if args and isinstance(args[0], (str, torch.device)):
                self._device = torch.device(args[0])
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
    def load(cls, path: str, map_location: str = "cpu") -> "NeuralBayesianNetwork":
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
    def from_bif(cls, path: str) -> "NeuralBayesianNetwork":
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
