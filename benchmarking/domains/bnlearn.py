"""bnlearn discrete-network domain.

Loads canonical bnlearn networks. We try in order:

1. ``pgmpy.utils.get_example_model(name)`` — pgmpy ships these as fitted
   models, no network needed. This is the **default** path and works offline.
2. Fall back to downloading ``.bif`` from a list of mirror URLs (the bnlearn
   site has reorganised paths repeatedly; we try several known stems).

Ground-truth marginals are computed via pgmpy's exact VE on the true CPTs.
Concurrent downloads are guarded with ``filelock``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

from benchmarking.domains.base import (
    BenchmarkDomain,
    BenchmarkProblem,
    GroundTruth,
)
from benchmarking.queries import make_query_battery

logger = logging.getLogger(__name__)


# (name → (n_nodes, n_edges))
BNLEARN_NETWORKS = {
    "asia":      (8, 8),
    "cancer":    (5, 4),
    "earthquake":(5, 4),
    "survey":    (6, 6),
    "sachs":     (11, 17),
    "child":     (20, 25),
    "alarm":     (37, 46),
    "insurance": (27, 52),
    "hailfinder":(56, 66),
    "hepar2":    (70, 123),
    "win95pts":  (76, 112),
    "barley":    (48, 84),
    "water":     (32, 66),
    "mildew":    (35, 46),
}

# Known mirror URL templates, tried in order if pgmpy.get_example_model fails.
_MIRROR_TEMPLATES = [
    "https://www.bnlearn.com/bnrepository/{name}/{name}.bif.gz",
    "https://www.bnlearn.com/bnrepository/{name}/{name}.bif",
    "https://erdogant.github.io/datasets/{name}.bif",
]


def _cache_dir() -> Path:
    p = Path(os.path.expanduser("~/.cache/nbn/bnlearn"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _bif_path(name: str) -> Path:
    return _cache_dir() / f"{name}.bif"


def _download_bif(name: str) -> Path | None:
    """Download <name>.bif into the cache; tries multiple mirrors."""
    path = _bif_path(name)
    if path.exists() and path.stat().st_size > 0:
        return path
    try:
        from filelock import FileLock
    except ImportError:
        FileLock = None  # type: ignore[assignment]
    lock_path = str(path) + ".lock"

    def _do_download() -> bool:
        import gzip
        import requests
        for tmpl in _MIRROR_TEMPLATES:
            url = tmpl.format(name=name)
            try:
                r = requests.get(url, timeout=60)
                if r.status_code != 200:
                    continue
                content = r.content
                if url.endswith(".gz"):
                    content = gzip.decompress(content)
                path.write_bytes(content)
                logger.info("Downloaded %s from %s", name, url)
                return True
            except Exception as e:
                logger.debug("Mirror %s failed: %s", url, e)
        return False

    ok = False
    if FileLock is not None:
        with FileLock(lock_path):
            if path.exists() and path.stat().st_size > 0:
                ok = True
            else:
                ok = _do_download()
    else:
        ok = _do_download()
    return path if ok else None


def _load_pgmpy_model(name: str):
    """Try every available path to get a fitted pgmpy model for ``name``."""
    # Path 1a: pgmpy >= 1.3 — `pgmpy.example_models.load_model`
    try:
        from pgmpy.example_models import load_model
        return load_model(name)
    except Exception as e:
        logger.debug("pgmpy.example_models.load_model('%s') failed: %s", name, e)

    # Path 1b: older pgmpy — `pgmpy.utils.get_example_model`
    try:
        import warnings
        from pgmpy.utils import get_example_model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return get_example_model(name)
    except Exception as e:
        logger.debug("pgmpy.utils.get_example_model('%s') failed: %s", name, e)

    # Path 2: cached / downloadable .bif
    bif = _download_bif(name)
    if bif is None:
        raise RuntimeError(
            f"Could not load bnlearn network '{name}': "
            "pgmpy built-ins failed and no .bif mirror was reachable."
        )
    from pgmpy.readwrite import BIFReader
    return BIFReader(str(bif)).get_model()


class BnlearnDomain(BenchmarkDomain):
    name = "bnlearn"

    def list_problems(self) -> list[str]:
        return list(BNLEARN_NETWORKS.keys())

    def load_problem(
        self,
        problem: str,
        *,
        n_train: int,
        n_test: int,
        seed: int,
        device: torch.device,
    ) -> BenchmarkProblem:
        try:
            from pgmpy.inference import VariableElimination
            from pgmpy.sampling import BayesianModelSampling
        except ImportError as e:
            raise ImportError("BnlearnDomain needs pgmpy: pip install pgmpy") from e

        bn = _load_pgmpy_model(problem)
        sampler = BayesianModelSampling(bn)

        edges = list(bn.edges())
        nodes = list(bn.nodes())
        torch.manual_seed(seed)

        # Card map
        cards = {n: len(bn.get_cpds(n).state_names[n]) for n in nodes}
        variables = {n: ("discrete", cards[n]) for n in nodes}

        # Train/test data — pgmpy returns string labels; map to ints via state_names.
        df_train = sampler.forward_sample(size=n_train, show_progress=False, seed=seed)
        df_test = sampler.forward_sample(size=n_test, show_progress=False, seed=seed + 1)

        def _to_long(df, n):
            states = bn.get_cpds(n).state_names[n]
            label_to_idx = {s: i for i, s in enumerate(states)}
            col = df[n]
            try:
                arr = col.astype(int).to_numpy()
            except (ValueError, TypeError):
                arr = col.map(label_to_idx).astype(int).to_numpy()
            return torch.tensor(arr, dtype=torch.long, device=device)

        train_data = {n: _to_long(df_train, n) for n in nodes}
        test_data = {n: _to_long(df_test, n) for n in nodes}

        # Queries
        queries = make_query_battery(
            nodes=nodes, discrete_nodes=nodes, continuous_nodes=[],
            cardinalities=cards,
            n_conditional_single=20, n_per_multi=10, n_map=10, n_do=10,
            seed=seed,
        )

        # Ground truth marginals via pgmpy exact VE
        infer = VariableElimination(bn)
        gt_marg: dict[str, torch.Tensor] = {}
        for n in nodes:
            res = infer.query(variables=[n], show_progress=False)
            gt_marg[n] = torch.tensor(res.values, dtype=torch.float, device=device)

        return BenchmarkProblem(
            name=problem, dag=edges, variables=variables,
            train_data=train_data, test_data=test_data,
            queries=queries, ground_truth=GroundTruth(marginals=gt_marg),
        )
