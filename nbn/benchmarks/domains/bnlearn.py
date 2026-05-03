"""bnlearn discrete-network domain.

Lazy-downloads ``.bif`` files from the bnlearn repository and caches them
locally.  Ground-truth marginals are computed via pgmpy's exact VE on the
true CPTs.  Concurrent downloads are guarded with a ``filelock``.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from nbn.benchmarks.domains.base import (
    BenchmarkDomain,
    BenchmarkProblem,
    GroundTruth,
)
from nbn.benchmarks.queries import make_query_battery


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


def _cache_dir() -> Path:
    p = Path(os.path.expanduser("~/.cache/nbn/bnlearn"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _bif_path(name: str) -> Path:
    return _cache_dir() / f"{name}.bif"


def _download_bif(name: str) -> Path:
    """Download <name>.bif into the cache; concurrent-safe."""
    path = _bif_path(name)
    if path.exists() and path.stat().st_size > 0:
        return path
    try:
        from filelock import FileLock
    except ImportError:
        FileLock = None  # type: ignore[assignment]
    lock_path = str(path) + ".lock"

    def _do_download():
        import requests
        url = f"https://www.bnlearn.com/bnrepository/{name}/{name}.bif"
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)

    if FileLock is not None:
        with FileLock(lock_path):
            if not (path.exists() and path.stat().st_size > 0):
                _do_download()
    else:
        _do_download()
    return path


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
            from pgmpy.readwrite import BIFReader
            from pgmpy.sampling import BayesianModelSampling
        except ImportError as e:
            raise ImportError("BnlearnDomain needs pgmpy: pip install pgmpy") from e

        bif = _download_bif(problem)
        bn = BIFReader(str(bif)).get_model()
        sampler = BayesianModelSampling(bn)

        edges = list(bn.edges())
        nodes = list(bn.nodes())
        torch.manual_seed(seed)

        # Card map
        cards = {n: len(bn.get_cpds(n).state_names[n]) for n in nodes}
        variables = {n: ("discrete", cards[n]) for n in nodes}

        # Train/test data
        df_train = sampler.forward_sample(size=n_train, show_progress=False, seed=seed)
        df_test = sampler.forward_sample(size=n_test, show_progress=False, seed=seed + 1)
        train_data = {
            n: torch.tensor(df_train[n].astype(int).values, dtype=torch.long, device=device)
            for n in nodes
        }
        test_data = {
            n: torch.tensor(df_test[n].astype(int).values, dtype=torch.long, device=device)
            for n in nodes
        }

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
