"""v0.6c-A Finding 1 reproducer: localise the nbn_lw continuous_lg
W₁ jump from 0.04 (smoke B=16, n_lw_samples=4000) to 0.83 (paper
B=1024, default n_lw_samples=???) at n_nodes=10.

The brief routes us toward Hypothesis A (importance resampling
operates on flattened B × n_samples weights → per-query collapse as
B grows).  Empirical check requires varying B and n_lw_samples
independently to attribute the W₁ shift to the right axis.

Variables:

* ``B``: the runner's nbn_batch_size (queries per cell).
  ``_compute_inference_accuracy`` caps the per-cell iteration at
  ``min(B, 16)`` — so even at paper config B=1024, the accuracy
  estimate is averaged over 16 (target, evidence) rows.  The
  posterior-sample path itself runs *per row*: evidence is sliced
  to a 0-D scalar and reshaped to ``[1]`` before
  ``LikelihoodWeightingEngine.query(...)`` is invoked.

* ``n_lw_samples``: the per-query posterior-sample budget passed
  through to ``LikelihoodWeightingEngine``.  Smoke YAML explicitly
  sets ``nbn_lw_n_samples: 4000``.  Paper YAML omits the field;
  ``CrashTestConfig`` defaults it to **512**.

If the W₁ shift is dominated by ``n_lw_samples`` (smoke=4000 vs
paper=512), the cause is config drift, not a B-scaling bug.

The diagnostic per-row uses ``q.evidence`` row 0 as the canonical
case (this is exactly what ``_compute_inference_accuracy`` does for
each iteration of its inner loop).
"""
from __future__ import annotations

import json
import pathlib
import sys

from benchmarking._crash_test_utils import CrashTestConfig
from benchmarking.crash_test_runner import (
    _baseline_posterior_for_query,
    _build_query_batch,
    _forward_with_clamp_posterior_samples,
)
from benchmarking.synthetic import make_synthetic_bn


def measure_single_query(
    *,
    family: str,
    n_nodes: int,
    B: int,
    n_lw_samples: int,
    seed: int = 0,
    n_train: int = 10_000,
    n_reference: int = 5_000,
    row_idx: int = 0,
) -> dict:
    """Run nbn_lw on a single (target, evidence row) on a synthetic BN
    of the given family/size.  Reports per-query effective sample
    statistics + W₁ to the forward-clamp oracle.

    Mirrors the per-row work of ``_compute_inference_accuracy``: build
    the batched query, slice ``ev_row = {k: v[row_idx] for k, v in
    q.evidence.items()}``, and pass that to
    ``_baseline_posterior_for_query`` (which reshapes each value to
    ``[1]`` before calling ``eng.query``).
    """
    bn = make_synthetic_bn(
        family=family, n_nodes=n_nodes,
        edge_density=0.20, max_in_degree=4, cardinality=4,
        seed=seed, device="cpu",
        n_train=n_train, n_test=500, n_reference=n_reference,
    )
    q = _build_query_batch(bn, B=B, seed=seed)
    target = q.targets[0]
    target_kind = bn.variable_specs[target][0]
    ev_row = {k: v[row_idx] for k, v in q.evidence.items()}

    pred = _baseline_posterior_for_query(
        bn, "nbn_lw", target, ev_row,
        target_kind=target_kind,
        n_samples=n_lw_samples, n_lw_samples=n_lw_samples,
    )
    if pred is None:
        return {
            "B": B, "n_lw_samples": n_lw_samples,
            "family": family, "n_nodes": n_nodes, "seed": seed,
            "n_returned": None, "n_unique_samples": None,
            "pred_mean": None, "pred_std": None,
            "oracle_mean": None, "oracle_std": None,
            "w1_to_oracle": None,
        }

    n_unique = (
        int(pred.unique().numel()) if pred.dtype.is_floating_point else None
    )

    oracle = _forward_with_clamp_posterior_samples(
        bn, [target], ev_row, n_samples=2000,
    )

    if pred.dim() == 1 and oracle is not None and oracle.numel() > 0:
        oracle_1d = oracle.flatten().float()
        a, _ = pred.float().sort()
        b, _ = oracle_1d.sort()
        n = min(a.numel(), b.numel())
        w1 = float((a[:n] - b[:n]).abs().mean().item())
    else:
        w1 = float("nan")

    return {
        "B": B,
        "n_lw_samples": n_lw_samples,
        "family": family,
        "n_nodes": n_nodes,
        "seed": seed,
        "n_returned": int(pred.numel()),
        "n_unique_samples": n_unique,
        "pred_mean": float(pred.float().mean().item()),
        "pred_std": float(pred.float().std().item()),
        "oracle_mean": (
            float(oracle.float().mean().item())
            if oracle is not None and oracle.numel() > 0 else None
        ),
        "oracle_std": (
            float(oracle.float().std().item())
            if oracle is not None and oracle.numel() > 0 else None
        ),
        "w1_to_oracle": w1,
    }


def main() -> None:
    print("=== v0.6c-A Finding 1: nbn_lw axis-attribution sweep ===",
          file=sys.stderr)
    print(file=sys.stderr)

    # Confirm the YAML-default disparity that motivated the test.
    smoke_cfg = CrashTestConfig.from_yaml(
        "benchmarking/configs/inference_smoke.yaml",
    )
    paper_cfg = CrashTestConfig.from_yaml(
        "benchmarking/configs/inference_paper.yaml",
    )
    config_observation = {
        "smoke_n_lw_samples": smoke_cfg.nbn_lw_n_samples,
        "paper_n_lw_samples": paper_cfg.nbn_lw_n_samples,
        "default_n_lw_samples": (
            CrashTestConfig.__dataclass_fields__["nbn_lw_n_samples"].default
        ),
    }
    print(f"smoke YAML  nbn_lw_n_samples = {config_observation['smoke_n_lw_samples']}",
          file=sys.stderr)
    print(f"paper YAML  nbn_lw_n_samples = {config_observation['paper_n_lw_samples']} "
          f"(default {config_observation['default_n_lw_samples']} — paper YAML "
          f"omits the field)",
          file=sys.stderr)
    print(file=sys.stderr)

    results: dict = {"config_observation": config_observation}

    # Sweep 1: hold n_lw_samples=4000 (smoke setting), vary B from 1 → 1024.
    # If Hypothesis A is correct, W₁ should grow with B.
    print("--- Sweep 1: vary B, fixed n_lw_samples=4000 (smoke setting) ---",
          file=sys.stderr)
    sweep_B: list[dict] = []
    for B in [1, 4, 16, 64, 256, 1024]:
        r = measure_single_query(
            family="continuous_lg", n_nodes=10, B=B,
            n_lw_samples=4000, seed=0,
        )
        nu = r["n_unique_samples"] or 0
        print(
            f"continuous_lg B={r['B']:5d}  n_unique={nu:5d}  "
            f"pred_std={r['pred_std']:.4f}  oracle_std={r['oracle_std']:.4f}  "
            f"W1={r['w1_to_oracle']:.4f}",
            file=sys.stderr,
        )
        sweep_B.append(r)
    results["sweep_1_vary_B_at_4000_samples"] = sweep_B
    print(file=sys.stderr)

    # Sweep 2: hold B=16 (smoke setting), vary n_lw_samples from 128 → 8000.
    # If config-drift is the cause, W₁ should grow as n_lw_samples shrinks.
    print("--- Sweep 2: vary n_lw_samples, fixed B=16 (smoke setting) ---",
          file=sys.stderr)
    sweep_S: list[dict] = []
    for s in [128, 256, 512, 1024, 2000, 4000, 8000]:
        r = measure_single_query(
            family="continuous_lg", n_nodes=10, B=16,
            n_lw_samples=s, seed=0,
        )
        nu = r["n_unique_samples"] or 0
        print(
            f"continuous_lg n_lw_samples={r['n_lw_samples']:5d}  "
            f"n_unique={nu:5d}  pred_std={r['pred_std']:.4f}  "
            f"W1={r['w1_to_oracle']:.4f}",
            file=sys.stderr,
        )
        sweep_S.append(r)
    results["sweep_2_vary_n_lw_samples_at_B16"] = sweep_S
    print(file=sys.stderr)

    # Sweep 3: paper-like cell (B=1024, n_lw_samples=default 512) vs
    # smoke-like cell (B=16, n_lw_samples=4000) — corner test.
    print("--- Sweep 3: paper-like vs smoke-like corner test ---",
          file=sys.stderr)
    corners: list[dict] = []
    for label, B, s in [
        ("smoke-like (B=16, n_lw_samples=4000)", 16, 4000),
        ("paper-like (B=1024, n_lw_samples=512 default)", 1024, 512),
        ("paper-B + smoke-samples (B=1024, n_lw_samples=4000)", 1024, 4000),
        ("smoke-B + paper-samples (B=16, n_lw_samples=512)", 16, 512),
    ]:
        r = measure_single_query(
            family="continuous_lg", n_nodes=10, B=B,
            n_lw_samples=s, seed=0,
        )
        r["label"] = label
        nu = r["n_unique_samples"] or 0
        print(
            f"  {label}\n"
            f"    n_unique={nu:5d}  pred_std={r['pred_std']:.4f}  "
            f"W1={r['w1_to_oracle']:.4f}",
            file=sys.stderr,
        )
        corners.append(r)
    results["sweep_3_corners"] = corners
    print(file=sys.stderr)

    # Discrete control — should be invariant in both axes since the
    # discrete posterior path doesn't use the importance-resample
    # collapse (returns probability vector, not samples).
    print("--- Sweep 4: discrete n=10 control ---", file=sys.stderr)
    disc: list[dict] = []
    for B in [16, 1024]:
        for s in [512, 4000]:
            r = measure_single_query(
                family="discrete", n_nodes=10, B=B,
                n_lw_samples=s, seed=0,
            )
            print(
                f"discrete       B={r['B']:5d}  n_lw_samples={r['n_lw_samples']:5d}  "
                f"W1={r['w1_to_oracle']}",
                file=sys.stderr,
            )
            disc.append(r)
    results["sweep_4_discrete_control"] = disc
    print(file=sys.stderr)

    # Hypothesis routing markers (machine-readable summary).
    sweep_B_w1s = [r["w1_to_oracle"] for r in sweep_B if r["w1_to_oracle"]]
    sweep_S_w1s = [r["w1_to_oracle"] for r in sweep_S if r["w1_to_oracle"]]
    results["routing_markers"] = {
        "vary_B_W1_min": min(sweep_B_w1s) if sweep_B_w1s else None,
        "vary_B_W1_max": max(sweep_B_w1s) if sweep_B_w1s else None,
        "vary_B_W1_range_ratio": (
            max(sweep_B_w1s) / min(sweep_B_w1s)
            if sweep_B_w1s and min(sweep_B_w1s) > 0 else None
        ),
        "vary_n_samples_W1_min": min(sweep_S_w1s) if sweep_S_w1s else None,
        "vary_n_samples_W1_max": max(sweep_S_w1s) if sweep_S_w1s else None,
        "vary_n_samples_W1_range_ratio": (
            max(sweep_S_w1s) / min(sweep_S_w1s)
            if sweep_S_w1s and min(sweep_S_w1s) > 0 else None
        ),
    }

    print("=== Routing markers (axis-attribution) ===", file=sys.stderr)
    rm = results["routing_markers"]
    print(f"  vary B (1 → 1024 at n_lw_samples=4000)        "
          f"W1 range: [{rm['vary_B_W1_min']}, {rm['vary_B_W1_max']}]  "
          f"ratio: {rm['vary_B_W1_range_ratio']}",
          file=sys.stderr)
    print(f"  vary n_lw_samples (128 → 8000 at B=16)         "
          f"W1 range: [{rm['vary_n_samples_W1_min']}, {rm['vary_n_samples_W1_max']}]  "
          f"ratio: {rm['vary_n_samples_W1_range_ratio']}",
          file=sys.stderr)
    print(file=sys.stderr)

    out_path = pathlib.Path("docs/v0.6c-a/lw_b_scaling.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
