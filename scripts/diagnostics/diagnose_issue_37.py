"""v0.7-#37 investigation: pgmpy-lg-predict W₁ outlier on continuous_lg.

Three diagnostic experiments:
  A — sample-size dependence of empirical W₁ for pgmpy's analytic posterior.
  B — pgmpy LG fit quality vs the true SCM parameters.
  C — scale check across pgmpy / NBN / ground-truth.

Run on cpu (cuda also fine).  No code modifications under test — pure
read-only investigation.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from benchmarking.baselines._adapter_v2 import build_adapter_v2
from benchmarking._baseline_registry import BaselineSpec
from benchmarking.crash_test_runner import (
    _baseline_posterior_for_query,
    _build_query_batch,
    _forward_with_clamp_posterior_samples,
    _w1,
)
from benchmarking.synthetic import make_synthetic_bn


def _print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    # Reproduce the smoke-config continuous_lg cell that exhibits the gap.
    bn = make_synthetic_bn(
        family="continuous_lg",
        n_nodes=10,
        edge_density=0.20,
        max_in_degree=4,
        cardinality=4,
        fraction_continuous=1.0,
        n_train=2000,
        n_test=200,
        n_reference=500,
        seed=0,
        device="cpu",
    )
    print(f"BN: {bn.name}, n_nodes={len(bn.column_order)}, "
          f"n_train={bn.train_data[bn.column_order[0]].shape[0]}")

    # Build a query battery — same shape as the inference cell.
    queries = _build_query_batch(bn, B=16, seed=0)
    target = queries.targets[0]
    print(f"Target: {target}")
    print(f"Evidence keys: {list(queries.evidence.keys())}")

    # ------------------------------------------------------------------ #
    # Fit pgmpy-lg-predict adapter once; reuse across experiments.
    # ------------------------------------------------------------------ #
    spec_pgmpy = BaselineSpec(
        library="pgmpy", mechanism="lg",
        param_method="mle", inference_method="predict",
    )
    pgmpy_adapter = build_adapter_v2(spec_pgmpy, device="cpu")
    pgmpy_adapter.fit(bn.train_data, bn.dag, bn.variable_specs)
    pgmpy_v1 = pgmpy_adapter._v1   # PgmpyAdapter

    # NBN-LG-LW for the "what should it look like" reference.
    spec_nbn = BaselineSpec(
        library="nbn", mechanism="lg",
        param_method="mle", inference_method="lw",
    )
    nbn_adapter = build_adapter_v2(spec_nbn, device="cpu")
    nbn_adapter.fit(bn.train_data, bn.dag, bn.variable_specs)

    # ------------------------------------------------------------------ #
    # EXPERIMENT A — sample-size dependence
    # ------------------------------------------------------------------ #
    _print_section("Experiment A — sample-size dependence")
    print("Computing pgmpy-lg-predict W₁ at varying n_samples per row.")
    print("Each row: averaged W₁ over 8 evidence rows; oracle uses n=2000.")
    print(f"{'n_samples':>10}  {'pgmpy_w1':>10}  {'nbn_w1 (ref)':>14}")

    for n_pgmpy in (1, 10, 100, 500, 1000, 4000):
        accs_pgmpy: List[float] = []
        accs_nbn: List[float] = []
        for i in range(8):
            ev_row = {k: v[i] for k, v in queries.evidence.items()}
            try:
                oracle = _forward_with_clamp_posterior_samples(
                    bn, [target], ev_row, n_samples=2000,
                )
            except Exception:
                continue
            if oracle is None or oracle.shape[0] < 100:
                continue
            oracle_flat = oracle.reshape(-1)

            # pgmpy at this n_samples
            pred_pgmpy = _baseline_posterior_for_query(
                bn, "pgmpy", target, ev_row, "continuous",
                n_samples=n_pgmpy, n_lw_samples=n_pgmpy,
            )
            if pred_pgmpy is not None:
                accs_pgmpy.append(_w1(pred_pgmpy, oracle_flat))

            # nbn-lg-lw at default 512 LW samples (constant; reference)
            if n_pgmpy == 4000:
                pred_nbn = _baseline_posterior_for_query(
                    bn, "nbn_lw", target, ev_row, "continuous",
                    n_samples=200, n_lw_samples=512,
                )
                if pred_nbn is not None:
                    accs_nbn.append(_w1(pred_nbn, oracle_flat))

        mean_pgmpy = sum(accs_pgmpy) / len(accs_pgmpy) if accs_pgmpy else float("nan")
        mean_nbn = (
            sum(accs_nbn) / len(accs_nbn) if accs_nbn else float("nan")
        )
        print(f"{n_pgmpy:>10d}  {mean_pgmpy:>10.4f}  "
              f"{(mean_nbn if accs_nbn else float('nan')):>14.4f}")

    # ------------------------------------------------------------------ #
    # EXPERIMENT B — fit quality (pgmpy MLE vs true LG params)
    # ------------------------------------------------------------------ #
    _print_section("Experiment B — fit quality (pgmpy MLE vs ground truth)")
    print("Per-node Frobenius / abs error of pgmpy's fitted LG vs the true SCM.")
    print(f"{'node':>8}  {'pgmpy_β':>30}  {'true_β':>30}")

    # Pull true mechanism params from bn.true_model.
    relative_beta_errs: List[float] = []
    relative_sigma_errs: List[float] = []
    abs_bias_errs: List[float] = []

    pgmpy_cpds: Dict[str, object] = {}
    for cpd in pgmpy_v1.lg_model.get_cpds():
        pgmpy_cpds[cpd.variable] = cpd

    for node in bn.column_order:
        kind, _k = bn.variable_specs[node]
        if kind != "continuous":
            continue
        true_mech = bn.true_model.mechanisms[node]
        true_w = true_mech._weight.detach().cpu().reshape(-1).numpy()  # [P]
        true_b = true_mech._bias.detach().cpu().reshape(-1).numpy()    # [1]
        true_sigma = torch.exp(
            true_mech._log_scale.detach().cpu(),
        ).reshape(-1).numpy()  # [1]
        true_parents = list(bn.dag.predecessors(node))

        cpd = pgmpy_cpds.get(node)
        if cpd is None:
            continue
        pgmpy_betas = np.asarray(cpd.beta, dtype=float).reshape(-1)
        pgmpy_evidence = list(cpd.evidence) if hasattr(cpd, "evidence") else []
        pgmpy_var = float(getattr(cpd, "std", getattr(cpd, "variance", float("nan"))))

        # pgmpy beta = [intercept, slope_for_each_evidence].
        if len(pgmpy_betas) == 0:
            continue
        pgmpy_intercept = float(pgmpy_betas[0])
        pgmpy_slopes = pgmpy_betas[1:]  # ordered by pgmpy_evidence

        # Reorder pgmpy slopes to match true_parents order.
        slope_by_parent = {p: float(pgmpy_slopes[i])
                            for i, p in enumerate(pgmpy_evidence)
                            if i < len(pgmpy_slopes)}
        pgmpy_slope_aligned = np.array(
            [slope_by_parent.get(p, 0.0) for p in true_parents],
        )

        # Errors.
        if len(true_parents) > 0 and len(true_w) > 0:
            beta_err = np.linalg.norm(pgmpy_slope_aligned - true_w)
            beta_norm = max(np.linalg.norm(true_w), 1e-9)
            relative_beta_errs.append(beta_err / beta_norm)
        bias_abs = abs(pgmpy_intercept - float(true_b[0]))
        abs_bias_errs.append(bias_abs)

        # pgmpy_var is "std" historically (sometimes variance — check below).
        # We compare both interpretations against true_sigma^2 and true_sigma.
        true_var = float(true_sigma[0]) ** 2
        sigma_err_var = abs(pgmpy_var - true_var) / max(true_var, 1e-9)
        sigma_err_std = abs(pgmpy_var - float(true_sigma[0])) / max(
            float(true_sigma[0]), 1e-9,
        )
        relative_sigma_errs.append(min(sigma_err_var, sigma_err_std))

        # Print per-node row (truncate slopes for readability).
        beta_str = (
            f"{pgmpy_intercept:+.3f} | "
            f"{', '.join(f'{s:+.3f}' for s in pgmpy_slope_aligned[:3])}"
            + (" ..." if len(pgmpy_slope_aligned) > 3 else "")
        )
        true_beta_str = (
            f"{float(true_b[0]):+.3f} | "
            f"{', '.join(f'{s:+.3f}' for s in true_w[:3])}"
            + (" ..." if len(true_w) > 3 else "")
        )
        print(f"{node:>8}  {beta_str:>30}  {true_beta_str:>30}")

    if relative_beta_errs:
        print()
        print(f"Mean relative β slope error: "
              f"{sum(relative_beta_errs) / len(relative_beta_errs):.4f}")
        print(f"Max  relative β slope error: "
              f"{max(relative_beta_errs):.4f}")
    if abs_bias_errs:
        print(f"Mean abs intercept error:    "
              f"{sum(abs_bias_errs) / len(abs_bias_errs):.4f}")
    if relative_sigma_errs:
        print(f"Mean relative σ error:       "
              f"{sum(relative_sigma_errs) / len(relative_sigma_errs):.4f}")

    # ------------------------------------------------------------------ #
    # EXPERIMENT C — scale check
    # ------------------------------------------------------------------ #
    _print_section("Experiment C — scale check")
    print("Print mean / stddev of:")
    print("  - pgmpy-lg-predict's predicted posterior (analytic).")
    print("  - ground-truth samples filtered through bn.true_model.")
    print("  - nbn-lg-lw's predicted posterior samples.")
    print("All three should be on the same scale.")

    print()
    print(f"{'row':>4}  {'pgmpy(μ,σ)':>22}  "
          f"{'oracle(μ,σ)':>22}  {'nbn(μ,σ)':>22}")
    for i in range(6):
        ev_row = {k: v[i] for k, v in queries.evidence.items()}

        try:
            oracle = _forward_with_clamp_posterior_samples(
                bn, [target], ev_row, n_samples=2000,
            )
        except Exception:
            oracle = None
        oracle_mean = oracle_std = float("nan")
        if oracle is not None and oracle.shape[0] >= 100:
            oracle_mean = float(oracle.mean().item())
            oracle_std = float(oracle.std().item())

        pred_pgmpy = _baseline_posterior_for_query(
            bn, "pgmpy", target, ev_row, "continuous",
            n_samples=2000, n_lw_samples=2000,
        )
        if pred_pgmpy is not None and pred_pgmpy.numel() > 0:
            pgmpy_mean = float(pred_pgmpy.mean().item())
            pgmpy_std = float(pred_pgmpy.std().item())
        else:
            pgmpy_mean = pgmpy_std = float("nan")

        pred_nbn = _baseline_posterior_for_query(
            bn, "nbn_lw", target, ev_row, "continuous",
            n_samples=200, n_lw_samples=512,
        )
        if pred_nbn is not None and pred_nbn.numel() > 0:
            nbn_mean = float(pred_nbn.mean().item())
            nbn_std = float(pred_nbn.std().item())
        else:
            nbn_mean = nbn_std = float("nan")

        print(f"{i:>4d}  "
              f"({pgmpy_mean:+.3f}, {pgmpy_std:.3f})    "
              f"({oracle_mean:+.3f}, {oracle_std:.3f})    "
              f"({nbn_mean:+.3f}, {nbn_std:.3f})")


if __name__ == "__main__":
    main()
