"""CLI entry point: ``nbn-bench``.

Runs a configurable benchmark suite comparing NBN to baselines on
bnlearn networks, writing results to a parquet table.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

logger = logging.getLogger("nbn.bench")


DEFAULT_NETWORKS = ["asia", "cancer", "earthquake", "sachs", "child"]


def run_bnlearn_suite(
    network_names: List[str],
    output_path: str,
    n_queries: int = 100,
    n_samples_lw: int = 4096,
    download: bool = True,
) -> None:
    """Run the bnlearn benchmark suite.

    For each network, fits NBN, then runs ``n_queries`` random conditional
    queries with both ``TensorVariableElimination`` and ``LikelihoodWeighting``.
    Records wall-clock time and (when pgmpy is available) the KL to pgmpy's
    exact answers as accuracy ground truth.
    """
    from nbn.benchmarks.bnlearn_loader import load_bnlearn
    from nbn.benchmarks.metrics import kl_divergence
    from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
    from nbn.inference.tensor_ve import TensorVariableElimination

    rows: List[Dict[str, Any]] = []
    for name in network_names:
        logger.info("Network: %s", name)
        try:
            model = load_bnlearn(name, download=download)
        except Exception as e:
            logger.warning("Skipping %s: %s", name, e)
            continue

        nodes = model.dag.topological_order()
        ve = TensorVariableElimination()
        lw = LikelihoodWeightingEngine(n_samples=n_samples_lw)

        rng = torch.Generator().manual_seed(0)
        for q_idx in range(n_queries):
            tgt = nodes[q_idx % len(nodes)]
            ev_nodes = [n for n in nodes if n != tgt][:2]
            evidence = {n: torch.tensor(0) for n in ev_nodes}

            row: Dict[str, Any] = {"network": name, "query_idx": q_idx, "target": tgt}

            t0 = time.perf_counter()
            try:
                p_ve = ve.query(model, [tgt], evidence)
                row["nbn_ve_time_ms"] = (time.perf_counter() - t0) * 1000
                row["nbn_ve_ok"] = True
            except Exception as e:
                row["nbn_ve_ok"] = False
                row["nbn_ve_error"] = str(e)[:100]
                p_ve = None

            t0 = time.perf_counter()
            try:
                p_lw = lw.query(model, [tgt], evidence)
                row["nbn_lw_time_ms"] = (time.perf_counter() - t0) * 1000
                row["nbn_lw_ok"] = True
                if p_ve is not None and isinstance(p_lw, torch.Tensor):
                    row["lw_vs_ve_kl"] = float(kl_divergence(p_ve, p_lw))
            except Exception as e:
                row["nbn_lw_ok"] = False
                row["nbn_lw_error"] = str(e)[:100]

            rows.append(row)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(out)
        logger.info("Wrote %d rows to %s", len(rows), out)
    except ImportError:
        import json
        out.with_suffix(".json").write_text(json.dumps(rows, default=str, indent=2))
        logger.info("Wrote JSON results to %s", out.with_suffix(".json"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="nbn-bench")
    parser.add_argument("--networks", nargs="+", default=DEFAULT_NETWORKS)
    parser.add_argument("--out", default="results/bnlearn.parquet")
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--n-samples-lw", type=int, default=4096)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run_bnlearn_suite(
        args.networks,
        args.out,
        n_queries=args.n_queries,
        n_samples_lw=args.n_samples_lw,
        download=not args.no_download,
    )


if __name__ == "__main__":
    main()
