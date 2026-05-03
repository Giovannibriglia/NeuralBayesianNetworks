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


def _bench_inference_throughput(
    adapter, problem, target, batch_sizes,
) -> list[dict]:
    """Sweep batch size B; record throughput for adapters with `query_batch`."""
    from benchmarking.domains.base import Query
    rows = []
    rng = torch.Generator().manual_seed(0)
    nodes = [n for n in problem.variables if n != target]
    if len(nodes) < 2:
        return rows
    e_nodes = nodes[:2]
    cards = {n: int(problem.variables[n][1]) for n in e_nodes}
    for B in batch_sizes:
        evidence = {n: torch.randint(0, cards[n], (B,), generator=rng)
                    for n in e_nodes}
        q = Query(targets=(target,), evidence=evidence, kind="conditional")
        try:
            if hasattr(adapter, "query_batch") and adapter.name.startswith("nbn"):
                t0 = time.perf_counter()
                _ = adapter.query_batch(q)
                dt = time.perf_counter() - t0
                method = "batched"
            else:
                t0 = time.perf_counter()
                for i in range(B):
                    ev_i = {k: v[i:i + 1] for k, v in evidence.items()}
                    adapter.query(Query(targets=q.targets, evidence=ev_i,
                                         kind="conditional"))
                dt = time.perf_counter() - t0
                method = "loop"
        except Exception as e:
            logger.warning("inference_throughput skip %s B=%d: %s", adapter.name, B, e)
            continue
        rows.append({
            "baseline": adapter.name, "problem": problem.name,
            "kind": "inference_throughput", "B": B, "time_s": dt,
            "ms_per_query": dt / B * 1000, "throughput_qps": B / max(dt, 1e-9),
            "method": method,
        })
    return rows


def run(config: dict | str | Path) -> Any:
    """Run a benchmark suite per ``config``; return a pandas DataFrame.

    Three optional dimensions:

    * ``mode`` — ``"parameter_learning"`` (default; fits + serial queries +
      ground-truth metrics) or ``"inference"`` (fits + sweeps batch sizes
      via ``query_batch`` for inference-throughput measurement). When
      ``"both"``, both are recorded.
    * ``measure_fit_time`` — bool; if true, records each baseline's fit time
      in seconds (parameter-learning signal).
    * ``inference_batch_sizes`` — explicit list, e.g. ``[1, 16, 256, 1024]``.
      Implies ``mode="inference"``.
    """
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
    mode = cfg.get("mode", "parameter_learning")
    measure_fit_time = bool(cfg.get("measure_fit_time", mode != "inference"))
    inference_batch_sizes = cfg.get("inference_batch_sizes")
    if inference_batch_sizes:
        mode = "both" if mode == "parameter_learning" else mode

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
                kw = {"device": device_name} if b_name in {
                    "nbn", "nbn_lw", "nbn_hybrid", "nbn_ve", "gpytorch",
                } else {}
                try:
                    adapter = get_adapter(b_name, **kw)
                    fit_t0 = time.perf_counter()
                    adapter.fit(problem)
                    fit_dt = time.perf_counter() - fit_t0
                except Exception as e:
                    logger.warning("Skipping baseline %s on %s: %s", b_name, prob_name, e)
                    continue
                if measure_fit_time:
                    rows.append({
                        "baseline": b_name, "problem": prob_name,
                        "kind": "fit_time", "device": device_name,
                        "time_s": fit_dt,
                    })
                if mode in {"parameter_learning", "both"}:
                    rows.extend([{**r, "device": device_name}
                                 for r in _bench_one_baseline(adapter, problem,
                                                              kinds=kinds)])
                    rows.extend([{**r, "device": device_name}
                                 for r in _compute_marginal_metrics(
                                     adapter, problem, problem.ground_truth)])
                if mode in {"inference", "both"} and inference_batch_sizes:
                    target = next(iter(problem.variables))
                    rows.extend([{**r, "device": device_name}
                                 for r in _bench_inference_throughput(
                                     adapter, problem, target,
                                     inference_batch_sizes)])
                adapter.teardown()

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out)
            logger.info("Wrote %d rows to %s", len(df), out)
            # Companion LaTeX table for paper-ready copy-paste
            tex_path = Path(out).with_suffix(".tex")
            _write_summary_latex(rows, tex_path, mode=mode)
            logger.info("Wrote LaTeX table to %s", tex_path)
        return df
    except ImportError:
        return rows


def _write_summary_latex(rows: list[dict], path, *, mode: str) -> None:
    """Pick a sensible column set per mode and dump booktabs LaTeX."""
    from benchmarking.latex import write_latex_table

    if mode == "inference":
        cols = [
            ("baseline", "Baseline"), ("problem", "Problem"),
            ("device", "Device"), ("B", "B"),
            ("ms_per_query", "ms/query"),
            ("throughput_qps", "q/s"), ("method", "Method"),
        ]
        sub = [r for r in rows if r.get("kind") == "inference_throughput"]
        cap = "Inference throughput by batch size."
    else:
        cols = [
            ("baseline", "Baseline"), ("problem", "Problem"),
            ("device", "Device"), ("kind", "Query"),
            ("time_s", "Time (s)"), ("tv", "TV"), ("kl", "KL"),
        ]
        sub = [r for r in rows
               if r.get("kind") in {"marginal_metric", "fit_time"}]
        cap = "Parameter-learning + serial-inference summary."
    if not sub:
        return
    write_latex_table(
        path, sub, cols,
        caption=cap, label=f"tab:nbn-bench-{mode}",
        formats={"time_s": ".3f", "ms_per_query": ".2f",
                 "throughput_qps": ".0f", "tv": ".4f", "kl": ".4f"},
    )


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
