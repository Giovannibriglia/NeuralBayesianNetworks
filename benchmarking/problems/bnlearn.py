"""BnlearnProblemSource: load real Bayesian networks from bnlearn.

Phase 4 of v0.13 (issue #109). Stage 2 implements **discrete** networks
only: on-demand ``.bif`` download with caching to ``~/.cache/nbn/bnlearn/``,
pgmpy-based forward sampling for train/test/reference data, and a
ground-truth reference pool so the existing oracle
(``benchmarking.core.oracle.filter_ground_truth``) scores them with no
oracle changes. Gaussian and CLG networks (from the bundled JSON files
committed in Stage 1) land in Stage 3.

See docs/phase4-design-draft.md for the full design.
"""
from __future__ import annotations

import gzip
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import networkx as nx
import torch

from benchmarking.domains.base import BenchmarkProblem, FailedProblem, GroundTruth

logger = logging.getLogger(__name__)


def _kind_to_family(kind: str) -> str:
    """Map a registry ``kind`` to the v0.13 parquet ``family`` string.

    Mirrors the ``family`` values set by ``_discrete_problems`` /
    ``_continuous_problems`` so a ``FailedProblem`` row carries the same
    ``family`` a successful load would have produced.
    """
    if kind == "gaussian":
        return "continuous_gauss"
    if kind == "clg":
        return "clg"
    return "discrete"

# Cache directory for downloaded .bif files.
_CACHE_DIR = Path("~/.cache/nbn/bnlearn").expanduser()

# Network metadata registry. All 29 canonical networks are listed so the
# registry is the single source of truth; Stage 2 only *loads* the discrete
# ones. Gaussian/CLG entries raise NotImplementedError until Stage 3.
# n_nodes is informational (logging); the real count comes from the model.
_NETWORKS: dict[str, dict] = {
    # ---- Discrete (loaded from on-demand .bif) ----
    # Small
    "asia":       {"kind": "discrete", "size_class": "small",      "n_nodes": 8},
    "cancer":     {"kind": "discrete", "size_class": "small",      "n_nodes": 5},
    "earthquake": {"kind": "discrete", "size_class": "small",      "n_nodes": 5},
    "sachs":      {"kind": "discrete", "size_class": "small",      "n_nodes": 11},
    "survey":     {"kind": "discrete", "size_class": "small",      "n_nodes": 6},
    # Medium
    "alarm":      {"kind": "discrete", "size_class": "medium",     "n_nodes": 37},
    "barley":     {"kind": "discrete", "size_class": "medium",     "n_nodes": 48},
    "child":      {"kind": "discrete", "size_class": "medium",     "n_nodes": 20},
    "insurance":  {"kind": "discrete", "size_class": "medium",     "n_nodes": 27},
    "mildew":     {"kind": "discrete", "size_class": "medium",     "n_nodes": 35},
    "water":      {"kind": "discrete", "size_class": "medium",     "n_nodes": 32},
    # Large
    "hailfinder": {"kind": "discrete", "size_class": "large",      "n_nodes": 56},
    "hepar2":     {"kind": "discrete", "size_class": "large",      "n_nodes": 70},
    "win95pts":   {"kind": "discrete", "size_class": "large",      "n_nodes": 76},
    # Very large
    "andes":      {"kind": "discrete", "size_class": "very_large", "n_nodes": 223},
    "diabetes":   {"kind": "discrete", "size_class": "very_large", "n_nodes": 413},
    "link":       {"kind": "discrete", "size_class": "very_large", "n_nodes": 724},
    "munin1":     {"kind": "discrete", "size_class": "very_large", "n_nodes": 186},
    "pathfinder": {"kind": "discrete", "size_class": "very_large", "n_nodes": 135},
    "pigs":       {"kind": "discrete", "size_class": "very_large", "n_nodes": 441},
    # Massive
    "munin":      {"kind": "discrete", "size_class": "massive",    "n_nodes": 1041},
    "munin2":     {"kind": "discrete", "size_class": "massive",    "n_nodes": 1003},
    "munin3":     {"kind": "discrete", "size_class": "massive",    "n_nodes": 1044},
    "munin4":     {"kind": "discrete", "size_class": "massive",    "n_nodes": 1041},
    # ---- Gaussian (bundled JSON; Stage 3) ----
    "ecoli70":    {"kind": "gaussian", "size_class": "medium",     "n_nodes": 46},
    "magic-niab": {"kind": "gaussian", "size_class": "medium",     "n_nodes": 44},
    "magic-irri": {"kind": "gaussian", "size_class": "large",      "n_nodes": 64},
    "arth150":    {"kind": "gaussian", "size_class": "very_large", "n_nodes": 107},
    # ---- Conditional Linear Gaussian (bundled JSON; Stage 3) ----
    "healthcare": {"kind": "clg",      "size_class": "small",      "n_nodes": 7},
    "sangiovese": {"kind": "clg",      "size_class": "small",      "n_nodes": 15},
    "mehra":      {"kind": "clg",      "size_class": "medium",     "n_nodes": 24},
}


@dataclass
class BnlearnConfig:
    """Configuration for ``BnlearnProblemSource``.

    Attributes
    ----------
    networks:
        bnlearn network names (each must be in ``_NETWORKS``).
    seeds:
        Integer seeds. Each ``(network, seed)`` becomes one
        ``BenchmarkProblem``; the seed drives forward-sampling randomness.
    n_train, n_test:
        Row counts for ``train_data`` / ``test_data``.
    n_reference:
        Row count for the ground-truth reference pool stored on
        ``problem.ground_truth.samples`` (used by the discrete oracle's
        exact-match rejection filter). Mirrors ``SyntheticConfig.n_reference``.
    """

    networks: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [0])
    n_train: int = 5000
    n_test: int = 1000
    n_reference: int = 5000


# bnlearn hosts the munin1/2/3 partitions under the ``munin4`` directory, not a
# per-name directory.  Verified 2026-06-04: ``/bnrepository/munin4/munin{1,2,3}.bif.gz``
# return HTTP 200 while ``/bnrepository/munin{1,2,3}/...`` return 404 (this is what
# silently killed the 2026-06-03 bnlearn_complete run at munin1).  Everything else —
# including ``munin`` and ``munin4`` themselves — uses the per-name directory.
_URL_DIRECTORY_OVERRIDES: dict[str, str] = {
    "munin1": "munin4",
    "munin2": "munin4",
    "munin3": "munin4",
}

# Hard timeout for a single download attempt.  ``urlretrieve`` accepts no timeout
# and would block indefinitely on a stalled connection; ``urlopen(timeout=...)``
# bounds connect + per-read blocking.  No retries: a wrong URL is not transient,
# and a failed load is now recorded as a FailedProblem row (the run continues)
# rather than aborting — so retrying a permanent 404 would only waste time.
_DOWNLOAD_TIMEOUT_S = 30.0


def _bif_url(name: str) -> str:
    """URL for the bnlearn discrete ``.bif`` file (gzipped)."""
    directory = _URL_DIRECTORY_OVERRIDES.get(name, name)
    return f"https://www.bnlearn.com/bnrepository/{directory}/{name}.bif.gz"


def _ensure_cached(name: str) -> Path:
    """Download a discrete ``.bif.gz`` to the cache and decompress if absent.

    Returns the local path to the uncompressed ``.bif`` file.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_bif = _CACHE_DIR / f"{name}.bif"
    if local_bif.exists():
        return local_bif

    url = _bif_url(name)
    logger.info("Downloading bnlearn network %s from %s", name, url)
    local_gz = _CACHE_DIR / f"{name}.bif.gz"
    try:
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_S) as response:
            local_gz.write_bytes(response.read())
    except Exception as e:  # noqa: BLE001 — re-raised with context
        raise RuntimeError(f"Failed to download {name}.bif.gz from {url}: {e}") from e

    with gzip.open(local_gz, "rb") as f_in, open(local_bif, "wb") as f_out:
        f_out.write(f_in.read())
    local_gz.unlink()  # keep only the decompressed .bif
    return local_bif


def _load_discrete_model(name: str) -> Any:
    """Load a discrete bnlearn network via pgmpy's ``BIFReader``."""
    from pgmpy.readwrite import BIFReader

    bif_path = _ensure_cached(name)
    return BIFReader(str(bif_path)).get_model()


def _forward_sample_discrete(
    model: Any, n_samples: int, seed: int,
) -> dict[str, torch.Tensor]:
    """Forward-sample ``n_samples`` rows; return ``{node: long tensor}``.

    Values are integer state indices in the CPD's ``state_names`` order
    (not the raw string state labels), matching the synthetic source's
    discrete encoding.
    """
    from pgmpy.sampling import BayesianModelSampling

    sampler = BayesianModelSampling(model)
    try:
        df = sampler.forward_sample(size=n_samples, seed=seed, show_progress=False)
    except TypeError:
        # Older pgmpy without the seed kwarg.
        import numpy as np

        np.random.seed(seed)
        df = sampler.forward_sample(size=n_samples, show_progress=False)

    result: dict[str, torch.Tensor] = {}
    for node in model.nodes():
        state_names = model.get_cpds(node).state_names[node]
        state_to_idx = {s: i for i, s in enumerate(state_names)}
        col = df[node].map(state_to_idx).to_numpy()
        result[node] = torch.tensor(col, dtype=torch.long)
    return result


def _reference_pool(
    sample_dict: dict[str, torch.Tensor],
    variables: dict[str, tuple[str, int]],
    dag: list[tuple[str, str]],
) -> torch.Tensor:
    """Stack a forward-sampled dict into a ``[n_ref, n_nodes]`` float pool.

    Columns are in topological-sort order — the same order
    ``benchmarking.core.oracle._column_order`` reconstructs, so the discrete
    oracle indexes the right columns.
    """
    g = nx.DiGraph()
    g.add_nodes_from(variables)
    g.add_edges_from(dag)
    col_order = list(nx.topological_sort(g))
    # .float() per column so mixed long (discrete) + float (continuous) columns
    # in CLG networks stack cleanly; the discrete oracle re-casts via col.long().
    return torch.stack([sample_dict[n].float() for n in col_order], dim=1)


_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bnlearn"


def _load_continuous_json(name: str) -> dict:
    """Load a Gaussian or CLG network from its bundled JSON (Stage 1 output)."""
    json_path = _DATA_DIR / f"{name}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Bundled bnlearn JSON not found at {json_path}. Gaussian/CLG "
            f"network JSON is produced by scripts/convert_bnlearn_continuous.R."
        )
    import json

    with open(json_path) as f:
        return json.load(f)


def _decode_config_index(
    i: int, discrete_parents: list[str], dlevels: dict[str, list[str]],
) -> dict[str, str]:
    """Decode a flat CLG config index → discrete-parent state assignment.

    First discrete parent varies fastest — bnlearn's native ``expand.grid``
    encoding (design doc §5.2). Inverse of the encoding used in
    ``_BnlearnContinuousModel._sample_clg_node``.
    """
    cfg: dict[str, str] = {}
    for dp in discrete_parents:
        n_states = len(dlevels[dp])
        cfg[dp] = dlevels[dp][i % n_states]
        i //= n_states
    return cfg


def _json_edges(data: dict) -> list[tuple[str, str]]:
    """Normalise the JSON ``edges`` field to ``[(from, to), ...]``.

    Stage 1 emits dicts ``{"from", "to"}``; tolerate ``[from, to]`` too.
    """
    out: list[tuple[str, str]] = []
    for e in data["edges"]:
        if isinstance(e, dict):
            out.append((e["from"], e["to"]))
        else:
            out.append((e[0], e[1]))
    return out


class _BnlearnContinuousModel:
    """Sampling model for Gaussian / CLG bnlearn networks.

    Doubles as (a) the data generator for train/test/reference rows and
    (b) ``problem.true_model`` for the oracle's
    ``forward_with_clamp_posterior_samples``, which calls
    ``true_model.sample(n=N, evidence=ev)`` and indexes the result by node
    name. ``sample`` therefore returns ``{node: tensor[N]}`` (long for
    discrete nodes, float for continuous), and clamps any node present in
    ``evidence`` to its observed value instead of drawing it.
    """

    def __init__(
        self, data: dict, variables: dict[str, tuple[str, int | None]],
    ) -> None:
        self._kind = data["kind"]                      # "gaussian" | "clg"
        self._cpds = {c["name"]: c for c in data["cpds"]}
        self._variables = variables
        self._nodes = list(data["nodes"])
        g = nx.DiGraph()
        g.add_nodes_from(self._nodes)
        g.add_edges_from(_json_edges(data))
        self._topo = list(nx.topological_sort(g))

    def sample(
        self,
        n: int = 1,
        evidence: dict | None = None,
        seed: int | None = None,
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Ancestral forward sample of ``n`` rows (evidence-clamped)."""
        import numpy as np

        evidence = evidence or {}
        rng = np.random.default_rng(seed)
        values: dict[str, Any] = {}

        for node in self._topo:
            if node in evidence:
                values[node] = self._clamp(node, evidence[node], n)
                continue
            cpd = self._cpds[node]
            ntype = cpd["type"]
            if ntype == "gaussian":
                values[node] = self._sample_gaussian_node(cpd, values, n, rng)
            elif ntype == "discrete":
                values[node] = self._sample_discrete_node(cpd, values, n, rng)
            elif ntype == "clg_continuous":
                values[node] = self._sample_clg_node(cpd, values, n, rng)
            else:
                raise ValueError(f"Unknown node type {ntype!r} for {node!r}")

        out: dict[str, torch.Tensor] = {}
        for node in self._nodes:
            is_discrete = self._variables[node][0] == "discrete"
            dtype = torch.long if is_discrete else torch.float32
            out[node] = torch.as_tensor(values[node], dtype=dtype)
        return out

    # -- per-node samplers (numpy arrays) -----------------------------------

    def _clamp(self, node: str, value: Any, n: int):
        import numpy as np

        if isinstance(value, torch.Tensor):
            scalar = value.detach().cpu().reshape(-1)[0].item()
        else:
            scalar = float(value)
        if self._variables[node][0] == "discrete":
            return np.full(n, int(round(scalar)), dtype=np.int64)
        return np.full(n, float(scalar), dtype=np.float64)

    @staticmethod
    def _sample_gaussian_node(cpd, values, n, rng):
        import numpy as np

        means = np.full(n, float(cpd["intercept"]), dtype=np.float64)
        for parent, beta in cpd["coefficients"].items():
            means += float(beta) * values[parent]
        return rng.normal(means, float(cpd["sd"]))

    def _sample_clg_node(self, cpd, values, n, rng):
        import numpy as np

        dps = cpd["discrete_parents"]
        cps = cpd["continuous_parents"]
        dlevels = cpd["dlevels"]
        intercepts = np.asarray(cpd["intercepts"], dtype=np.float64)
        sds = np.asarray(cpd["sds"], dtype=np.float64)

        # Per-sample flat config index: first discrete parent fastest. The
        # discrete parent's state-index (values[dp]) aligns with dlevels order
        # (== that node's own `states` order; verified for all bundled CLG nets).
        config = np.zeros(n, dtype=np.int64)
        mult = 1
        for dp in dps:
            config += values[dp].astype(np.int64) * mult
            mult *= len(dlevels[dp])

        means = intercepts[config].copy()
        for cp in cps:
            coefs = np.asarray(cpd["coefficients"][cp], dtype=np.float64)
            means += coefs[config] * values[cp]
        return rng.normal(means, sds[config])

    @staticmethod
    def _sample_discrete_node(cpd, values, n, rng):
        import numpy as np

        states = cpd["states"]
        parents = cpd.get("parents", [])
        k = len(states)
        prob = np.asarray(cpd["prob"], dtype=np.float64).reshape(
            cpd["prob_dim"], order="F",
        )  # axis 0 = own states; axes 1.. = parents (in `parents` order)

        if not parents:
            probs = np.broadcast_to(prob.reshape(k, 1), (k, n))
        else:
            # Parent state-indices align with each parent's prob axis (parent
            # levels == that parent's own `states`; verified on healthcare).
            parent_idx = tuple(values[p].astype(np.int64) for p in parents)
            probs = prob[(slice(None),) + parent_idx]  # (k, n)

        probs = probs / probs.sum(axis=0, keepdims=True).clip(min=1e-12)
        # Vectorised inverse-CDF categorical sampling.
        cum = np.cumsum(probs, axis=0)               # (k, n)
        u = rng.random(n)
        return (u[None, :] < cum).argmax(axis=0).astype(np.int64)


class BnlearnProblemSource:
    """Problem source for real Bayesian networks from the bnlearn repository.

    Stage 2: discrete networks only, loaded from cached ``.bif`` files via
    ``pgmpy.readwrite.BIFReader``. Stage 3 adds Gaussian and CLG networks
    from the bundled JSON files. Each ``(network, seed)`` pair in the config
    yields one ``BenchmarkProblem`` with the network's structure and
    seed-specific forward-sampled data.
    """

    def iter_problems(
        self, config: BnlearnConfig,
    ) -> Iterator[BenchmarkProblem | FailedProblem]:
        """Yield one ``BenchmarkProblem`` per ``(network, seed)`` pair.

        A network whose data fails to load (download 404, parse error, …)
        does **not** abort the run: the failure is caught here — keeping this
        generator alive — and a single ``FailedProblem`` sentinel is yielded
        in its place before continuing to the next network.  An unknown
        network name remains a fatal config error (raised), since it signals
        a typo the caller should fix rather than a transient/data problem.
        """
        for net_name in config.networks:
            if net_name not in _NETWORKS:
                raise ValueError(
                    f"Unknown bnlearn network: {net_name!r}. "
                    f"Known networks: {sorted(_NETWORKS.keys())}"
                )
            meta = _NETWORKS[net_name]
            kind = meta["kind"]
            logger.info(
                "Loading bnlearn network %s (kind=%s, n_nodes=%d)",
                net_name, kind, meta["n_nodes"],
            )
            try:
                if kind == "discrete":
                    yield from self._discrete_problems(net_name, config)
                elif kind in ("gaussian", "clg"):
                    yield from self._continuous_problems(net_name, kind, config)
                else:
                    raise ValueError(
                        f"Network {net_name!r} has unsupported kind={kind!r}."
                    )
            except Exception as exc:  # noqa: BLE001 — recorded as a FailedProblem row
                logger.exception(
                    "Failed to load bnlearn network %s; recording an error row "
                    "and continuing with the next network", net_name,
                )
                yield FailedProblem(
                    problem_id=net_name,
                    family=_kind_to_family(kind),
                    error_msg=f"{type(exc).__name__}: {exc}",
                    benchmark="bnlearn",
                )
                continue

    def _discrete_problems(
        self, net_name: str, config: BnlearnConfig,
    ) -> Iterator[BenchmarkProblem]:
        model = _load_discrete_model(net_name)  # cached across seeds
        dag = list(model.edges())
        variables: dict[str, tuple[str, int]] = {}
        for node in model.nodes():
            n_states = len(model.get_cpds(node).state_names[node])
            variables[node] = ("discrete", n_states)

        for seed in config.seeds:
            train_data = _forward_sample_discrete(model, config.n_train, seed)
            # Distinct derived seeds keep train/test/reference independent
            # yet deterministic per (network, seed).
            test_data = _forward_sample_discrete(model, config.n_test, seed + 100_000)
            ref_dict = _forward_sample_discrete(model, config.n_reference, seed + 200_000)
            gt_samples = _reference_pool(ref_dict, variables, dag)

            yield BenchmarkProblem(
                name=net_name,
                dag=dag,
                variables=variables,
                train_data=train_data,
                test_data=test_data,
                queries=[],
                ground_truth=GroundTruth(samples=gt_samples),
                true_model=model,
                family="discrete",
                problem_id=net_name,
                seed=seed,
            )

    def _continuous_problems(
        self, net_name: str, kind: str, config: BnlearnConfig,
    ) -> Iterator[BenchmarkProblem]:
        data = _load_continuous_json(net_name)
        dag = _json_edges(data)
        cpds = {c["name"]: c for c in data["cpds"]}
        # Per-node type uses the canonical labels the rest of the stack
        # expects: "discrete" (with cardinality) or "continuous". The CLG/
        # Gaussian distinction lives at the *family* level, not per node.
        # Adapters' is_applicable infer family from these via
        # ``kinds == {"continuous"}``, so the literal "continuous" matters.
        variables: dict[str, tuple[str, int | None]] = {}
        for node in data["nodes"]:
            c = cpds[node]
            if c["type"] == "discrete":
                variables[node] = ("discrete", len(c["states"]))
            else:
                variables[node] = ("continuous", None)

        # family: pure Gaussian -> continuous_gauss; mixed -> clg (design doc §9).
        family = "continuous_gauss" if kind == "gaussian" else "clg"
        model = _BnlearnContinuousModel(data, variables)  # shared across seeds

        for seed in config.seeds:
            train_data = model.sample(config.n_train, seed=seed)
            test_data = model.sample(config.n_test, seed=seed + 100_000)
            ref_dict = model.sample(config.n_reference, seed=seed + 200_000)
            gt_samples = _reference_pool(ref_dict, variables, dag)

            yield BenchmarkProblem(
                name=net_name,
                dag=dag,
                variables=variables,
                train_data=train_data,
                test_data=test_data,
                queries=[],
                ground_truth=GroundTruth(samples=gt_samples),
                true_model=model,
                family=family,
                problem_id=net_name,
                seed=seed,
            )
