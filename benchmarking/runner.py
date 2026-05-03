"""Plugin-based benchmark runner.

Reads a YAML config, dispatches to the requested domain + baselines, runs the
standard query battery, computes the standard metrics, and writes results to
parquet.

CLI: ``nbn-bench run <config.yaml>``.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import torch

from benchmarking import metrics as M
from benchmarking.baselines import get_adapter
from benchmarking.domains import get_domain

logger = logging.getLogger("nbn.bench")


def _load_config(config: dict | str | Path) -> dict:
    if isinstance(config, dict):
        return config
    import yaml
    with open(config) as f:
        return yaml.safe_load(f)


def _resolve_devices(devices: list[str] | None) -> list[str]:
    devs = []
    for d in devices or ["cpu"]:
        if d == "cuda" and not torch.cuda.is_available():
            logger.info("Skipping cuda — not available.")
            continue
        devs.append(d)
    return devs


def _bench_one_baseline(adapter, problem, *, kinds: set) -> list[dict]:
    """Run all of ``problem.queries`` on ``adapter``; return per-query rows."""
    rows = []
    for q in problem.queries:
        if q.kind not in kinds:
            continue
        try:
            t0 = time.perf_counter()
            res = adapter.query(q)
            dt = time.perf_counter() - t0
            ok = True
            err = None
        except (NotImplementedError, ValueError) as e:
            res = None
            dt = float("nan")
            ok = False
            err = str(e)[:80]
        rows.append({
            "baseline": adapter.name,
            "problem": problem.name,
            "kind": q.kind,
            "target": q.targets[0],
            "evidence_size": len(q.evidence),
            "time_s": dt,
            "ok": ok,
            "error": err,
            "result_shape": tuple(res.shape) if isinstance(res, torch.Tensor) else None,
        })
    return rows


def _compute_marginal_metrics(adapter, problem, gt) -> list[dict]:
    """For univariate-marginal queries, compute distance to ground truth."""
    rows = []
    if gt is None or not gt.marginals:
        return rows
    for q in problem.queries:
        if q.kind != "marginal":
            continue
        target = q.targets[0]
        if target not in gt.marginals:
            continue
        try:
            res = adapter.query(q)
        except Exception:
            continue
        ref = gt.marginals[target].cpu().float().reshape(-1)
        pred = res.cpu().float().reshape(-1)
        if pred.shape != ref.shape:
            continue
        rows.append({
            "baseline": adapter.name, "problem": problem.name,
            "target": target, "kind": "marginal_metric",
            "kl": float(M.kl_divergence(pred, ref).item()),
            "tv": float(0.5 * (pred - ref).abs().sum().item()),
            "mae": float((pred - ref).abs().mean().item()),
        })
    return rows


def run(config: dict | str | Path) -> Any:
    """Run a benchmark suite per ``config``; return a pandas DataFrame."""
    cfg = _load_config(config)
    domain = get_domain(cfg["domain"])
    problems = cfg.get("problems", [])
    n_train = int(cfg.get("n_train", 5000))
    n_test = int(cfg.get("n_test", 1000))
    seed = int(cfg.get("seed", 0))
    baselines = cfg.get("baselines", ["nbn"])
    devices = _resolve_devices(cfg.get("devices", ["cpu"]))
    kinds = set(cfg.get("query_kinds", ["marginal", "conditional", "map", "do"]))
    out = cfg.get("output")

    rows: list[dict] = []
    for device_name in devices:
        device = torch.device(device_name)
        for prob_name in problems:
            logger.info("Loading %s on %s", prob_name, device_name)
            problem = domain.load_problem(
                prob_name, n_train=n_train, n_test=n_test,
                seed=seed, device=device,
            )
            for b_name in baselines:
                kw = {"device": device_name} if b_name in {"nbn", "gpytorch"} else {}
                try:
                    adapter = get_adapter(b_name, **kw)
                    adapter.fit(problem)
                except Exception as e:
                    logger.warning("Skipping baseline %s on %s: %s", b_name, prob_name, e)
                    continue
                rows.extend([{**r, "device": device_name}
                             for r in _bench_one_baseline(adapter, problem, kinds=kinds)])
                rows.extend([{**r, "device": device_name}
                             for r in _compute_marginal_metrics(adapter, problem,
                                                                problem.ground_truth)])
                adapter.teardown()

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out)
            logger.info("Wrote %d rows to %s", len(df), out)
        return df
    except ImportError:
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(prog="nbn-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="Run a benchmark suite from YAML.")
    p_run.add_argument("config", type=str)
    p_run.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if args.cmd == "run":
        run(args.config)


if __name__ == "__main__":
    main()
