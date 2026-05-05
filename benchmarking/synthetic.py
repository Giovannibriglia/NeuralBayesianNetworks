"""Synthetic Bayesian-network generator for crash-test benchmarks (v0.4).

Produces fully-specified BNs with **known ground-truth parameters** in four
data regimes:

* ``discrete``             — every node is a Categorical CPD.
* ``continuous_lg``        — every node is a Linear-Gaussian CPD.
* ``continuous_nongauss``  — every node is a 3-component MDN where the
  mixture means are linear in the parents but the per-component biases are
  drawn from a skew-normal, so the resulting marginals are non-Gaussian.
* ``hybrid``               — a mix of discrete and continuous nodes; edges
  from continuous parents to discrete children pass through a fixed-quantile
  binner before indexing the categorical CPT.

The generator is the foundation for the v0.4 crash-test framework: every
metric reported by the parameter-learning and inference crash tests is
defined relative to the true mechanism returned here.

Backward compatibility
----------------------
The v0.3 helper ``generate_synthetic_hybrid(...)`` is kept as a thin
deprecation shim that forwards to ``make_synthetic_bn(family='hybrid', ...)``.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import networkx as nx
import torch
import torch.nn as nn

from nbn.core.network import NeuralBayesianNetwork
from nbn.mechanisms.categorical_table import CategoricalTableMechanism
from nbn.mechanisms.linear_gaussian import LinearGaussianMechanism
from nbn.mechanisms.mdn import MDNMechanism, _build_mlp

Family = Literal["discrete", "continuous_lg", "continuous_nongauss", "hybrid"]
_FAMILIES: tuple[Family, ...] = (
    "discrete",
    "continuous_lg",
    "continuous_nongauss",
    "hybrid",
)
_REFERENCE_LARGE_NETWORK_THRESHOLD = 1_000


# ---------------------------------------------------------------------- #
# Public dataclass
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class SyntheticBN:
    """A synthetic BN with known ground-truth parameters and matching data.

    Attributes
    ----------
    name:
        Stable, descriptive identifier of the form
        ``f"{family}_n{n_nodes}_d{int(edge_density*1000):03d}_k{cardinality}_seed{seed}"``.
    dag:
        The generated networkx DAG.
    variable_specs:
        Mapping ``node -> (kind, dim_or_card)`` matching the
        ``NeuralBayesianNetwork`` constructor's ``variables`` argument.
    true_model:
        The data-generating ``NeuralBayesianNetwork``: every mechanism has
        its parameters set to the ground-truth values used to draw the
        samples below.  No ``fit_local`` calls have been made; you can
        evaluate accuracy of any *fitted* model against this one directly.
    train_data, test_data:
        Independent ancestral samples from ``true_model`` keyed by node
        name; each is shape ``[n_train, ...]`` / ``[n_test, ...]``.
    ground_truth_samples:
        ``[n_reference, |V|]`` joint samples from ``true_model`` used as
        the reference for distributional metrics (``W₁``, energy, JS-norm)
        on continuous and hybrid problems.  ``None`` for fully discrete
        families where pgmpy-style exact marginals are the natural
        ground-truth target instead.
    column_order:
        Canonical column order for any tensor that follows this BN's
        column convention — including ``ground_truth_samples`` and any
        downstream cached pool the runner constructs.  Always equals
        ``tuple(nx.topological_sort(dag))``.  Single source of truth:
        do not recompute the topological sort elsewhere; use
        ``column_index(name)`` to map a node name to its column.

        Established in v0.6a after PR #14 round-4 surfaced a schema
        bug — ``bn.ground_truth_samples`` is written in topological-sort
        order, but the runner was indexing via insertion order, which
        differs for non-trivial DAGs.
    """

    name: str
    dag: nx.DiGraph
    variable_specs: Dict[str, Tuple[str, int]]
    true_model: NeuralBayesianNetwork
    train_data: Dict[str, torch.Tensor]
    test_data: Dict[str, torch.Tensor]
    ground_truth_samples: torch.Tensor | None
    n_nodes: int
    n_edges: int
    avg_in_degree: float
    cardinality: int
    family: Family
    column_order: Tuple[str, ...]

    def column_index(self, name: str) -> int:
        """Return the column index of ``name`` in tensors that follow
        this BN's column order (e.g. ``ground_truth_samples``).

        Single source of truth — never compute the index from
        ``list(self.dag.nodes())`` elsewhere.

        Raises
        ------
        KeyError
            If ``name`` is not a node in this BN.
        """
        try:
            return self.column_order.index(name)
        except ValueError as exc:
            raise KeyError(
                f"node {name!r} not in BN {self.name!r} "
                f"(known nodes: {self.column_order})",
            ) from exc


# ---------------------------------------------------------------------- #
# Public entry point
# ---------------------------------------------------------------------- #


def make_synthetic_bn(
    *,
    n_nodes: int,
    family: Family,
    edge_density: float = 0.20,
    max_in_degree: int = 4,
    cardinality: int = 4,
    fraction_continuous: float = 0.5,
    n_train: int = 10_000,
    n_test: int = 2_000,
    n_reference: int = 50_000,
    seed: int = 0,
    device: str | torch.device = "auto",
) -> SyntheticBN:
    """Generate a synthetic BN with known parameters and matching data.

    Topology
    --------
    Topologically-ordered Erdős-Rényi: nodes are indexed ``0..n-1``; edge
    ``(i, j)`` is admitted with probability ``edge_density`` iff ``i < j``
    *and* ``parents(j)`` does not yet hold ``max_in_degree`` parents.

    Mechanisms (per family)
    -----------------------
    * ``discrete`` — every node is a ``CategoricalTableMechanism`` with
      ``K = cardinality``.  Logits are drawn ``~ N(0, 1)`` per row of the
      CPT, independently of the parent configuration.
    * ``continuous_lg`` — every node is a ``LinearGaussianMechanism`` with
      weights ``~ N(0, 1/√P)`` (``P`` = number of parents) and noise
      ``σ ~ U[0.3, 1.0]``.
    * ``continuous_nongauss`` — every node is an ``MDNMechanism`` with
      ``K = 3`` Gaussian components.  The MDN is configured with a *zero
      hidden-layer* MLP, i.e. a single linear projection from parents to
      ``K + K * D + K * D`` outputs.  Component-wise: means are linear in
      parents with biases drawn from a skew-normal; sigmas are constant in
      parents and drawn ``~ U[0.3, 1.0]``; mixing weights are constant in
      parents and drawn from a ``Dirichlet(1, 1, 1)``.  This is the "hard"
      continuous regime.
    * ``hybrid`` — each node is independently flagged continuous with
      probability ``fraction_continuous`` (and discrete otherwise); each
      continuous node uses the LG mechanism; each discrete node uses a
      ``CategoricalTableMechanism`` whose continuous parents are first
      passed through a fixed-quantile binner attached to the mechanism
      (so the runtime indexing becomes integer-valued and the CPT shape
      is well-defined).

    Determinism
    -----------
    All sampling sources use ``torch.Generator(device=cpu).manual_seed(seed)``;
    the global RNG is left untouched.

    Memory
    ------
    For ``n_nodes >= 1000`` the requested ``n_reference`` is automatically
    halved with a ``UserWarning`` to keep the joint-samples tensor under
    a couple of GB.

    Returns
    -------
    A frozen ``SyntheticBN`` containing the DAG, the data-generating model,
    the train/test data, and (for continuous and hybrid families) a pool
    of joint reference samples.
    """
    if family not in _FAMILIES:
        raise ValueError(f"Unknown family {family!r}; expected one of {_FAMILIES}.")
    if n_nodes < 1:
        raise ValueError("n_nodes must be ≥ 1")
    if not (0.0 <= edge_density <= 1.0):
        raise ValueError("edge_density must be in [0, 1]")
    if max_in_degree < 1:
        raise ValueError("max_in_degree must be ≥ 1")
    if family == "hybrid" and not (0.0 < fraction_continuous < 1.0):
        raise ValueError(
            "fraction_continuous must be in (0, 1) for the hybrid family",
        )

    device_t = _resolve_device(device)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    if n_nodes >= _REFERENCE_LARGE_NETWORK_THRESHOLD and n_reference > 10_000:
        warnings.warn(
            f"make_synthetic_bn(n_nodes={n_nodes}): halving n_reference from "
            f"{n_reference} to {n_reference // 2} to bound memory.",
            UserWarning,
            stacklevel=2,
        )
        n_reference = n_reference // 2

    # 1. DAG topology -------------------------------------------------- #
    nodes = [f"X{i}" for i in range(n_nodes)]
    edges = _random_topo_dag(
        nodes, edge_density=edge_density, max_in_degree=max_in_degree, gen=gen,
    )
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    # 2. Per-node kind decision (only relevant for hybrid) ------------- #
    if family == "hybrid":
        kinds = {
            n: ("continuous" if torch.rand((), generator=gen).item() < fraction_continuous
                else "discrete")
            for n in nodes
        }
    elif family == "discrete":
        kinds = dict.fromkeys(nodes, "discrete")
    else:
        kinds = dict.fromkeys(nodes, "continuous")

    variable_specs: Dict[str, Tuple[str, int]] = {
        n: ("discrete", cardinality) if kinds[n] == "discrete" else ("continuous", 1)
        for n in nodes
    }

    # 3. Build the empty model ----------------------------------------- #
    model = NeuralBayesianNetwork(
        list(g.edges()),
        variables=variable_specs,
        device=str(device_t),
    )

    # 4. Populate mechanisms with ground-truth parameters -------------- #
    # column_order is the *single* place topological order is computed
    # in this function — all subsequent column-indexed work derives
    # from it (mechanism population order, sample-tensor cat order, the
    # SyntheticBN field exposed to callers).  See SyntheticBN docstring.
    column_order: Tuple[str, ...] = tuple(nx.topological_sort(g))
    for node in column_order:
        parents = list(model.dag.parents(node))
        kind = kinds[node]
        if kind == "discrete":
            mech: object = _build_discrete_mechanism(
                node, parents, kinds, cardinality, gen, device_t,
                family=family,
            )
        elif family == "continuous_lg" or (family == "hybrid" and kind == "continuous"):
            mech = _build_lg_mechanism(parents, gen, device_t)
        elif family == "continuous_nongauss":
            mech = _build_mdn_mechanism(parents, gen, device_t)
        else:
            raise AssertionError(
                f"Unhandled (family={family}, kind={kind}) combination",
            )
        model.set_mechanism(node, mech)  # type: ignore[arg-type]

    # 5. Sample data --------------------------------------------------- #
    # ``ancestral_sample`` reads from the global torch RNG; we snapshot,
    # seed it deterministically (same seed used for topology + params),
    # take all three sample passes consecutively, then restore the prior
    # state so the caller's RNG is untouched.
    with torch.no_grad(), _scoped_seed(seed, device_t):
        train_dict = _take_samples(model, n=n_train, device=device_t)
        test_dict = _take_samples(model, n=n_test, device=device_t)
        ground_truth_samples: torch.Tensor | None = None
        if family != "discrete":
            ref_dict = _take_samples(model, n=n_reference, device=device_t)
            ground_truth_samples = torch.cat(
                [ref_dict[n].reshape(n_reference, -1).float() for n in column_order],
                dim=-1,
            )

    name = (
        f"{family}_n{n_nodes}_d{int(round(edge_density * 1000)):03d}"
        f"_k{cardinality}_seed{int(seed)}"
    )
    avg_in = (sum(len(model.dag.parents(n)) for n in nodes) / n_nodes) if n_nodes else 0.0
    return SyntheticBN(
        name=name,
        dag=g,
        variable_specs=variable_specs,
        true_model=model,
        train_data=train_dict,
        test_data=test_dict,
        ground_truth_samples=ground_truth_samples,
        n_nodes=n_nodes,
        n_edges=len(edges),
        avg_in_degree=float(avg_in),
        cardinality=int(cardinality) if family in ("discrete", "hybrid") else -1,
        family=family,
        column_order=column_order,
    )


# ---------------------------------------------------------------------- #
# Topology
# ---------------------------------------------------------------------- #


def _random_topo_dag(
    nodes: List[str],
    *,
    edge_density: float,
    max_in_degree: int,
    gen: torch.Generator,
) -> List[Tuple[str, str]]:
    """Erdős-Rényi over ``i < j`` with a per-child max-in-degree cap."""
    n = len(nodes)
    edges: List[Tuple[str, str]] = []
    in_deg = [0] * n
    for j in range(n):
        if in_deg[j] >= max_in_degree:
            continue
        for i in range(j):
            if in_deg[j] >= max_in_degree:
                break
            if torch.rand((), generator=gen).item() < edge_density:
                edges.append((nodes[i], nodes[j]))
                in_deg[j] += 1
    return edges


# ---------------------------------------------------------------------- #
# Mechanism builders — discrete
# ---------------------------------------------------------------------- #


class _BinningCategoricalTable(CategoricalTableMechanism):
    """``CategoricalTableMechanism`` that bins continuous parents at runtime.

    The ground-truth quantile thresholds for each continuous parent are
    fixed at construction time and stored as a buffer so the mechanism
    can be ``.to(device)``-moved with the rest of the network.
    """

    is_discrete: bool = True

    def __init__(
        self,
        *,
        cardinality: int,
        thresholds: torch.Tensor,
        continuous_parent_mask: torch.Tensor,
    ) -> None:
        super().__init__(alpha=0.0)
        # thresholds has shape [n_parents, K-1] with NaN padding for
        # discrete-parent columns; the mask says which columns to bin.
        self.register_buffer("_thresholds", thresholds)
        self.register_buffer("_cont_mask", continuous_parent_mask.bool())
        self._k_for_binning = int(cardinality)

    def _to_int_parents(self, parents: torch.Tensor | None) -> torch.Tensor | None:
        if parents is None or parents.shape[-1] == 0:
            return parents
        out = parents.long().clone()
        if not self._cont_mask.any():
            return out
        for col in torch.where(self._cont_mask)[0].tolist():
            thr = self._thresholds[col]
            valid_thr = thr[~torch.isnan(thr)].to(parents.device)
            if valid_thr.numel() == 0:
                continue
            out[..., col] = torch.bucketize(
                parents[..., col].float().contiguous(), valid_thr,
            ).clamp_max(self._k_for_binning - 1).long()
        return out

    def forward(self, parents):  # type: ignore[override]
        return super().forward(self._to_int_parents(parents))

    def sample(self, parents, n: int = 1):  # type: ignore[override]
        return super().sample(self._to_int_parents(parents), n=n)

    def log_prob(self, x, parents):  # type: ignore[override]
        return super().log_prob(x, self._to_int_parents(parents))


def _build_discrete_mechanism(
    node: str,
    parents: List[str],
    kinds: Dict[str, str],
    cardinality: int,
    gen: torch.Generator,
    device: torch.device,
    *,
    family: Family,
) -> CategoricalTableMechanism:
    """Categorical CPT with i.i.d. ``N(0, 1)`` logits (and binner for hybrid)."""
    k = int(cardinality)
    parent_kinds = [kinds[p] for p in parents]
    parent_cards = [k for _ in parents]
    n_parent_states = 1
    for c in parent_cards:
        n_parent_states *= c
    logits = torch.randn(n_parent_states, k, generator=gen)

    has_continuous_parent = any(pk == "continuous" for pk in parent_kinds)
    if family == "hybrid" and has_continuous_parent:
        # Quantile thresholds for each continuous parent: K-1 cuts at fixed
        # quantiles of N(0, 1).  The continuous LG nodes in this generator
        # have approximately standard-normal marginals, so fixed normal
        # quantiles are a deterministic, reproducible discretiser.
        from torch.distributions import Normal
        normal = Normal(0.0, 1.0)
        quantiles = torch.linspace(1.0 / k, (k - 1) / k, k - 1)
        thr = normal.icdf(quantiles)  # [K-1]
        thresholds = torch.full((len(parents), k - 1), float("nan"))
        cont_mask = torch.zeros(len(parents), dtype=torch.bool)
        for col, pk in enumerate(parent_kinds):
            if pk == "continuous":
                thresholds[col] = thr
                cont_mask[col] = True
        mech: CategoricalTableMechanism = _BinningCategoricalTable(
            cardinality=k, thresholds=thresholds,
            continuous_parent_mask=cont_mask,
        )
    else:
        mech = CategoricalTableMechanism(alpha=0.0)

    mech._logits = nn.Parameter(logits.to(device))
    mech._n_classes = k
    mech._parent_cards = parent_cards
    strides: List[int] = []
    stride = 1
    for c in reversed(parent_cards):
        strides.append(stride)
        stride *= c
    mech._parent_strides = list(reversed(strides))
    mech._class_values = torch.arange(k, dtype=torch.float, device=device)
    mech.output_dim = 1
    return mech.to(device)


# ---------------------------------------------------------------------- #
# Mechanism builders — Linear-Gaussian
# ---------------------------------------------------------------------- #


def _build_lg_mechanism(
    parents: List[str],
    gen: torch.Generator,
    device: torch.device,
) -> LinearGaussianMechanism:
    """LG with ``W ~ N(0, 1/√P)`` weights and ``σ ~ U[0.3, 1.0]`` noise."""
    p = len(parents)
    d_x = 1
    sigma = 0.3 + 0.7 * torch.rand((d_x,), generator=gen)
    bias = torch.randn((d_x,), generator=gen)

    mech = LinearGaussianMechanism(ridge=0.0, learnable=True)
    mech.output_dim = d_x
    mech._input_dim = p
    if p == 0:
        weight = torch.zeros((0, d_x))
    else:
        weight = torch.randn((p, d_x), generator=gen) / math.sqrt(p)
    mech._weight = nn.Parameter(weight.to(device))
    mech._bias = nn.Parameter(bias.to(device))
    mech._log_scale = nn.Parameter(torch.log(sigma).to(device))
    return mech.to(device)


# ---------------------------------------------------------------------- #
# Mechanism builders — non-Gaussian MDN
# ---------------------------------------------------------------------- #


def _build_mdn_mechanism(
    parents: List[str],
    gen: torch.Generator,
    device: torch.device,
) -> MDNMechanism:
    """3-component MDN with linear projection from parents (no hidden layers).

    The MDN's MLP is configured with ``hidden=()`` so its output is a single
    linear projection of parents.  We hand-set the projection's weights and
    biases so that:

    * Mixture logits depend only on the bias (constant in parents),
      seeded from ``Dirichlet(1, 1, 1)``.
    * Mixture means depend linearly on parents with bias drawn from a
      skew-normal — this is the source of the non-Gaussian shape.
    * Mixture log-scales depend only on the bias (constant in parents),
      drawn from ``log U[0.3, 1.0]``.
    """
    k = 3
    d_x = 1
    mech = MDNMechanism(num_components=k, hidden=(), activation="relu")
    mech._d_x = d_x
    mech.output_dim = d_x

    # The bias / weight magnitudes are deliberately tighter than the brief's
    # ``N(0, 1/√P)`` ideal: long topological chains amplify variance at
    # every level, and an unconstrained mixture-of-skew-normals chain
    # diverges by node ~30 in our tests.  After v0.4b PR #12 fixed
    # ancestral sampling to actually respect per-row parent values
    # (closing v0.5 #11), the *correct* per-row chain is more sensitive
    # to bias drift than the buggy [n, n, D] take-samp[0] path was, so
    # we tighten constants again from 0.5 → 0.3 to keep n_nodes=100
    # joint samples bounded.  Strongly-non-Gaussian regime is a v0.5
    # ablation — see #7.
    mean_w_scale = 0.3
    skew_bias_scale = 0.3
    scale_low, scale_high = 0.3, 0.7

    if not parents:
        mech._d_pa = 0
        mech._root_logits = nn.Parameter(_dirichlet_logits(k, gen).to(device))
        mech._root_loc = nn.Parameter(
            (skew_bias_scale * _skew_normal((k, d_x), gen)).to(device),
        )
        mech._root_log_scale = nn.Parameter(
            torch.log(_uniform(scale_low, scale_high, (k, d_x), gen)).to(device),
        )
        return mech.to(device)

    p = len(parents)
    mech._d_pa = p
    out_dim = k + k * d_x + k * d_x
    mech.net = _build_mlp(p, (), out_dim, "relu").to(device)

    # The single linear layer maps parents [B, P] -> [B, out_dim].
    # Layout (matches MDN._params_from_parents):
    #   [logits (k) | locs (k*d_x) | log_scales (k*d_x)]
    linear: nn.Linear = mech.net[0]  # type: ignore[assignment]
    weight = torch.zeros(out_dim, p)
    bias = torch.zeros(out_dim)

    bias[:k] = _dirichlet_logits(k, gen)
    mean_w = torch.randn(k, d_x, p, generator=gen) * (mean_w_scale / math.sqrt(p))
    mean_b = skew_bias_scale * _skew_normal((k, d_x), gen)
    weight[k:k + k * d_x, :] = mean_w.reshape(k * d_x, p)
    bias[k:k + k * d_x] = mean_b.reshape(k * d_x)
    bias[k + k * d_x:] = torch.log(
        _uniform(scale_low, scale_high, (k, d_x), gen),
    ).reshape(k * d_x)

    with torch.no_grad():
        linear.weight.copy_(weight.to(device))
        linear.bias.copy_(bias.to(device))
    return mech.to(device)


def _dirichlet_logits(k: int, gen: torch.Generator) -> torch.Tensor:
    """Sample a ``Dirichlet(1, …, 1)`` vector and return its log."""
    raw = torch.empty(k).exponential_(generator=gen)
    probs = raw / raw.sum()
    return torch.log(probs.clamp_min(1e-12))


def _skew_normal(shape: Tuple[int, ...], gen: torch.Generator) -> torch.Tensor:
    """Skew-normal-ish draws via ``z₁ + 0.7 |z₂|``."""
    a = torch.randn(shape, generator=gen)
    b = torch.randn(shape, generator=gen)
    return a + 0.7 * b.abs()


def _uniform(
    low: float, high: float, shape: Tuple[int, ...], gen: torch.Generator,
) -> torch.Tensor:
    return low + (high - low) * torch.rand(shape, generator=gen)


# ---------------------------------------------------------------------- #
# Sampling helpers
# ---------------------------------------------------------------------- #


_SAMPLE_CHUNK = 2_000


def _take_samples(
    model: NeuralBayesianNetwork, *, n: int, device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Ancestral-sample ``n`` rows; return a dict of node tensors on device.

    The current ``ancestral_sample`` implementation builds intermediate
    ``[n, n, P]`` tensors per node which makes a one-shot ``n=50_000`` call
    O(n²) in memory.  We chunk the request into ``_SAMPLE_CHUNK``-sized
    pieces so the memory footprint scales linearly in ``n``.  Each chunk
    re-seeds nothing (the surrounding ``_scoped_seed`` keeps overall
    determinism).
    """
    if n <= _SAMPLE_CHUNK:
        return {k: v.to(device) for k, v in model.sample(n=n).items()}

    chunks: list[Dict[str, torch.Tensor]] = []
    remaining = n
    while remaining > 0:
        c = min(_SAMPLE_CHUNK, remaining)
        chunks.append(model.sample(n=c))
        remaining -= c
    keys = chunks[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        out[k] = torch.cat([d[k].reshape(d[k].shape[0], -1) for d in chunks], dim=0).to(device)
    return out


class _scoped_seed:
    """Context manager that seeds CPU + CUDA RNGs and restores their state on exit.

    The downstream ``ancestral_sample`` reads from the *global* RNG, so to
    keep ``make_synthetic_bn`` deterministic without permanently leaking
    state to the caller we snapshot, reseed, then restore.
    """

    def __init__(self, seed: int, device: torch.device) -> None:
        self.seed = int(seed)
        self.device = device
        self._cpu_state: torch.Tensor | None = None
        self._cuda_state: torch.Tensor | None = None

    def __enter__(self) -> _scoped_seed:
        self._cpu_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            self._cuda_state = torch.cuda.get_rng_state()
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._cpu_state is not None:
            torch.random.set_rng_state(self._cpu_state)
        if self._cuda_state is not None:
            torch.cuda.set_rng_state(self._cuda_state)


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ---------------------------------------------------------------------- #
# Backwards-compat shim
# ---------------------------------------------------------------------- #


def generate_synthetic_hybrid(
    n_nodes: int = 50,
    discrete_frac: float = 0.5,
    edge_prob: float = 0.15,
    seed: int = 0,
) -> Tuple[NeuralBayesianNetwork, Dict[str, torch.Tensor]]:
    """**Deprecated** — use :func:`make_synthetic_bn` instead.

    This shim preserves the v0.3 signature of ``generate_synthetic_hybrid``
    and forwards to :func:`make_synthetic_bn` with the hybrid family.

    Parameters
    ----------
    n_nodes:
        Total number of nodes in the generated DAG.
    discrete_frac:
        Fraction of *discrete* nodes; the new generator parameterises the
        complementary fraction (``fraction_continuous = 1 - discrete_frac``).
    edge_prob:
        Edge density (passed through as ``edge_density``).
    seed:
        Reproducibility seed.

    Returns
    -------
    A ``(true_model, train_data)`` pair, mirroring the v0.3 return contract.
    """
    warnings.warn(
        "generate_synthetic_hybrid is deprecated; use "
        "make_synthetic_bn(family='hybrid', ...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    bn = make_synthetic_bn(
        family="hybrid",
        n_nodes=n_nodes,
        edge_density=edge_prob,
        fraction_continuous=max(min(1.0 - discrete_frac, 0.99), 0.01),
        seed=seed,
    )
    return bn.true_model, bn.train_data


__all__ = [
    "Family",
    "SyntheticBN",
    "make_synthetic_bn",
    "generate_synthetic_hybrid",
]
