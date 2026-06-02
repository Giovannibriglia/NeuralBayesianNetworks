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

from benchmarking.domains.base import BenchmarkProblem, GroundTruth

logger = logging.getLogger(__name__)

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


def _bif_url(name: str) -> str:
    """URL for the bnlearn discrete ``.bif`` file (gzipped)."""
    return f"https://www.bnlearn.com/bnrepository/{name}/{name}.bif.gz"


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
        urllib.request.urlretrieve(url, local_gz)
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
    return torch.stack([sample_dict[n] for n in col_order], dim=1).float()


class BnlearnProblemSource:
    """Problem source for real Bayesian networks from the bnlearn repository.

    Stage 2: discrete networks only, loaded from cached ``.bif`` files via
    ``pgmpy.readwrite.BIFReader``. Stage 3 adds Gaussian and CLG networks
    from the bundled JSON files. Each ``(network, seed)`` pair in the config
    yields one ``BenchmarkProblem`` with the network's structure and
    seed-specific forward-sampled data.
    """

    def iter_problems(self, config: BnlearnConfig) -> Iterator[BenchmarkProblem]:
        """Yield one ``BenchmarkProblem`` per ``(network, seed)`` pair."""
        for net_name in config.networks:
            if net_name not in _NETWORKS:
                raise ValueError(
                    f"Unknown bnlearn network: {net_name!r}. "
                    f"Known networks: {sorted(_NETWORKS.keys())}"
                )
            meta = _NETWORKS[net_name]
            if meta["kind"] != "discrete":
                raise NotImplementedError(
                    f"Network {net_name!r} has kind={meta['kind']!r}; only "
                    f"'discrete' is supported in Stage 2 of Phase 4. "
                    f"Gaussian/CLG networks land in Stage 3."
                )

            logger.info(
                "Loading bnlearn network %s (kind=%s, n_nodes=%d)",
                net_name, meta["kind"], meta["n_nodes"],
            )
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
                test_data = _forward_sample_discrete(
                    model, config.n_test, seed + 100_000,
                )
                ref_dict = _forward_sample_discrete(
                    model, config.n_reference, seed + 200_000,
                )
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
