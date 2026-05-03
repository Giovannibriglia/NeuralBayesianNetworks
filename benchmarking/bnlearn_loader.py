from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


# All discrete networks from the bnlearn repository.
# (name, n_nodes, n_edges) — metadata for printing / figure annotations.
BNLEARN_NETWORKS = {
    # Small (≤20)
    "cancer":     {"nodes": 5,    "edges": 4,    "url_stem": "cancer"},
    "earthquake": {"nodes": 5,    "edges": 4,    "url_stem": "earthquake"},
    "survey":     {"nodes": 6,    "edges": 6,    "url_stem": "survey"},
    "asia":       {"nodes": 8,    "edges": 8,    "url_stem": "asia"},
    "sachs":      {"nodes": 11,   "edges": 17,   "url_stem": "sachs"},
    "child":      {"nodes": 20,   "edges": 25,   "url_stem": "child"},
    # Medium (20–100)
    "insurance":  {"nodes": 27,   "edges": 52,   "url_stem": "insurance"},
    "water":      {"nodes": 32,   "edges": 66,   "url_stem": "water"},
    "mildew":     {"nodes": 35,   "edges": 46,   "url_stem": "mildew"},
    "alarm":      {"nodes": 37,   "edges": 46,   "url_stem": "alarm"},
    "barley":     {"nodes": 48,   "edges": 84,   "url_stem": "barley"},
    "hailfinder": {"nodes": 56,   "edges": 66,   "url_stem": "hailfinder"},
    "hepar2":     {"nodes": 70,   "edges": 123,  "url_stem": "hepar2"},
    "win95pts":   {"nodes": 76,   "edges": 112,  "url_stem": "win95pts"},
    # Large (100–1000)
    "pathfinder": {"nodes": 109,  "edges": 195,  "url_stem": "pathfinder"},
    "munin1":     {"nodes": 186,  "edges": 273,  "url_stem": "munin1"},
    "andes":      {"nodes": 223,  "edges": 338,  "url_stem": "andes"},
    "diabetes":   {"nodes": 413,  "edges": 602,  "url_stem": "diabetes"},
    "pigs":       {"nodes": 441,  "edges": 592,  "url_stem": "pigs"},
    "link":       {"nodes": 724,  "edges": 1125, "url_stem": "link"},
    # Very large (>1000)
    "munin2":     {"nodes": 1003, "edges": 1244, "url_stem": "munin2"},
    "munin4":     {"nodes": 1038, "edges": 1388, "url_stem": "munin4"},
    "munin3":     {"nodes": 1041, "edges": 1306, "url_stem": "munin3"},
    "munin":      {"nodes": 1041, "edges": 1397, "url_stem": "munin"},
}


def _bnlearn_url(stem: str) -> str:
    """The official bnlearn .bif download URL."""
    return f"https://www.bnlearn.com/bnrepository/{stem}/{stem}.bif"


def load_bnlearn(
    name: str,
    cache_dir: str | None = None,
    download: bool = True,
):
    """Download (and cache) a bnlearn network and return an NBN model.

    Parameters
    ----------
    name:
        Network name (see ``BNLEARN_NETWORKS`` keys).
    cache_dir:
        Where to cache .bif files.  Defaults to ``~/.cache/nbn/bnlearn``.
    download:
        If False, only loads from cache and errors if missing.

    Returns
    -------
    NeuralBayesianNetwork
        Pre-fitted with ``CategoricalTableMechanism`` per node.
    """
    if name not in BNLEARN_NETWORKS:
        raise ValueError(
            f"Unknown bnlearn network '{name}'. Available: {sorted(BNLEARN_NETWORKS)}"
        )

    cache_dir = cache_dir or os.path.expanduser("~/.cache/nbn/bnlearn")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    bif_path = Path(cache_dir) / f"{name}.bif"

    if not bif_path.exists():
        if not download:
            raise FileNotFoundError(f"{bif_path} not found and download=False")
        try:
            import requests
        except ImportError as e:
            raise ImportError("Downloading requires `requests`. Install with [bench] extra.") from e
        url = _bnlearn_url(BNLEARN_NETWORKS[name]["url_stem"])
        logger.info("Downloading %s from %s", name, url)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        bif_path.write_bytes(r.content)

    from nbn.core.network import NeuralBayesianNetwork
    return NeuralBayesianNetwork.from_bif(str(bif_path))
