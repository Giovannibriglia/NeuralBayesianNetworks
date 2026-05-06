"""v0.6c-A Finding 2 reproducer: surface the actual non-conforming
elements in the ``Categorical(probs=pi)`` tensor that triggers the
Simplex constraint error in MDN sampling.

The PyTorch error message displays only the *first non-conforming
row* it finds and truncates with ``...`` — but the displayed row
``[0.0364, 0.7950, 0.1687]`` sums to 1.0 and is non-negative, which
means PyTorch's check found something else (NaN, Inf, negative) in
a *different* row that the truncation hid.

This script:

1. Reproduces the paper-config cell that failed
   (``continuous_nongauss n=100 seed=1, B=1024, n_train=10000``).
2. Monkey-patches ``torch.distributions.Categorical.__init__`` to
   intercept ``probs`` *before* the Simplex validate_args check
   would raise.
3. Audits the captured tensor for NaN / Inf / negative / off-simplex
   rows and prints a few samples.

Hypothesis 1: softmax overflow in ``MDNMechanism.sample`` (line 193
of ``nbn/mechanisms/mdn.py``).  Logits from
``_params_from_parents`` have ±Inf (likely from upstream parent
values that overflow the linear projection), softmax produces NaN,
and ``clamp_min(1e-7)`` + renormalise both *propagate* NaN
unchanged.

Hypothesis 2: divide-by-zero in renormalisation
(``pi = pi / pi.sum(-1, keepdim=True)``).  If softmax + clamp_min
produces an all-zero row (impossible after clamp_min(1e-7)) or a
row whose sum is itself NaN/Inf.

Hypothesis 3: another Categorical construction site (e.g., in the
LW engine's importance-weighted resampling).
"""
from __future__ import annotations

import json
import pathlib
import sys
import traceback

import torch

from benchmarking.crash_test_runner import (
    _baseline_posterior_for_query,
    _build_query_batch,
)
from benchmarking.synthetic import make_synthetic_bn


def _summarise_probs(probs: torch.Tensor) -> dict:
    """Audit a captured ``probs`` tensor for non-conformance."""
    p = probs.detach().cpu().float()
    has_nan = bool(torch.isnan(p).any())
    has_inf = bool(torch.isinf(p).any())
    has_neg = bool((p < 0).any())
    row_sums = p.sum(-1)
    off_simplex = (row_sums - 1.0).abs().gt(1e-4)
    n_off = int(off_simplex.sum().item())

    summary: dict = {
        "shape": list(p.shape),
        "dtype": str(p.dtype),
        "has_nan": has_nan, "n_nan": int(torch.isnan(p).sum()),
        "has_inf": has_inf, "n_inf": int(torch.isinf(p).sum()),
        "has_neg": has_neg, "n_neg": int((p < 0).sum()),
        "min_value": (
            float(p[~torch.isnan(p)].min().item())
            if (~torch.isnan(p)).any() else None
        ),
        "max_value": (
            float(p[~torch.isnan(p)].max().item())
            if (~torch.isnan(p)).any() else None
        ),
        "row_sums_min_finite": (
            float(row_sums[~torch.isnan(row_sums)].min().item())
            if (~torch.isnan(row_sums)).any() else None
        ),
        "row_sums_max_finite": (
            float(row_sums[~torch.isnan(row_sums)].max().item())
            if (~torch.isnan(row_sums)).any() else None
        ),
        "row_sums_n_off_simplex": n_off,
    }

    if has_nan:
        nan_rows = torch.isnan(p).any(-1).nonzero().flatten()
        summary["first_nan_row_idx"] = int(nan_rows[0].item())
        summary["nan_rows_count"] = int(nan_rows.numel())
    if has_inf:
        inf_rows = torch.isinf(p).any(-1).nonzero().flatten()
        summary["first_inf_row_idx"] = int(inf_rows[0].item())
        summary["inf_rows_count"] = int(inf_rows.numel())
    if has_neg:
        neg_rows = (p < 0).any(-1).nonzero().flatten()
        summary["first_neg_row_idx"] = int(neg_rows[0].item())
        summary["neg_rows_count"] = int(neg_rows.numel())

    # Sample 5 non-conforming rows (any pathology) for inspection.
    nan_mask = torch.isnan(p).any(-1)
    inf_mask = torch.isinf(p).any(-1)
    neg_mask = (p < 0).any(-1)
    off_mask = off_simplex
    bad_mask = nan_mask | inf_mask | neg_mask | off_mask
    bad_idx = bad_mask.nonzero().flatten()[:5]
    summary["sample_non_conforming_rows"] = []
    for i in bad_idx:
        ii = int(i.item())
        row = p[ii]
        summary["sample_non_conforming_rows"].append({
            "idx": ii,
            "row": row.tolist(),
            "row_sum": float(row.sum().item()),
            "row_has_nan": bool(torch.isnan(row).any()),
            "row_has_inf": bool(torch.isinf(row).any()),
            "row_has_neg": bool((row < 0).any()),
        })

    # Sample 3 conforming rows for contrast.
    good_idx = (~bad_mask).nonzero().flatten()[:3]
    summary["sample_conforming_rows"] = []
    for i in good_idx:
        ii = int(i.item())
        row = p[ii]
        summary["sample_conforming_rows"].append({
            "idx": ii,
            "row": row.tolist(),
            "row_sum": float(row.sum().item()),
        })

    return summary


def reproduce_mdn_simplex_error(
    *,
    family: str = "continuous_nongauss",
    n_nodes: int = 100,
    seed: int = 1,
    n_train: int = 10_000,
    n_reference: int = 5_000,
    B: int = 1024,
    n_lw_samples: int = 512,
) -> dict:
    """Build the paper-config cell that failed; capture the probs
    tensor at the moment the Simplex constraint raised.

    The patch is installed *before* ``make_synthetic_bn`` so we
    capture probs from both the data-generation ancestral sample
    chain (where MDN.sample runs to produce ``train_data`` /
    ``ground_truth_samples``) and the LW posterior path.  Either
    can fire the simplex error.
    """
    captured_probs: list[torch.Tensor] = []
    captured_logits: list[torch.Tensor] = []
    original_init = torch.distributions.Categorical.__init__

    def patched_init(self, probs=None, logits=None, validate_args=None):
        if probs is not None:
            captured_probs.append(probs.detach().cpu())
        if logits is not None:
            captured_logits.append(logits.detach().cpu())
        return original_init(
            self, probs=probs, logits=logits, validate_args=validate_args,
        )

    torch.distributions.Categorical.__init__ = patched_init
    error_info: dict | None = None
    error_phase: str | None = None
    bn = None

    try:
        bn = make_synthetic_bn(
            family=family, n_nodes=n_nodes,
            edge_density=0.20, max_in_degree=4, cardinality=4,
            seed=seed, device="cpu",
            n_train=n_train, n_test=500, n_reference=n_reference,
        )
    except Exception as exc:
        error_info = {
            "type": type(exc).__name__,
            "msg": str(exc)[:1500],
            "traceback": traceback.format_exc()[:2500],
        }
        error_phase = "make_synthetic_bn"

    if bn is not None:
        try:
            q = _build_query_batch(bn, B=B, seed=seed)
            target = q.targets[0]
            target_kind = bn.variable_specs[target][0]
            ev_row = {k: v[0] for k, v in q.evidence.items()}
            _ = _baseline_posterior_for_query(
                bn, "nbn_lw", target, ev_row,
                target_kind=target_kind,
                n_samples=n_lw_samples, n_lw_samples=n_lw_samples,
            )
        except Exception as exc:
            if error_info is None:
                error_info = {
                    "type": type(exc).__name__,
                    "msg": str(exc)[:1500],
                    "traceback": traceback.format_exc()[:2500],
                }
                error_phase = "posterior_query"

    torch.distributions.Categorical.__init__ = original_init

    findings: dict = {
        "config": {
            "family": family, "n_nodes": n_nodes, "seed": seed, "B": B,
            "n_train": n_train, "n_lw_samples": n_lw_samples,
        },
        "error": error_info,
        "error_phase": error_phase,
        "n_categorical_constructions_with_probs": len(captured_probs),
        "n_categorical_constructions_with_logits": len(captured_logits),
    }

    if captured_probs:
        # Prioritise: scan all captured probs for the first one with
        # any non-conformance.  If none non-conform, audit the last.
        target_idx = -1
        for i, p in enumerate(captured_probs):
            pf = p.float()
            if (
                torch.isnan(pf).any()
                or torch.isinf(pf).any()
                or (pf < 0).any()
                or (pf.sum(-1) - 1.0).abs().gt(1e-4).any()
            ):
                target_idx = i
                break
        if target_idx == -1:
            target_idx = len(captured_probs) - 1
        findings["audit_probs_index"] = target_idx
        findings["last_probs_audit"] = _summarise_probs(captured_probs[target_idx])

    if captured_logits:
        last_logits = captured_logits[-1].float()
        findings["last_logits_summary"] = {
            "shape": list(last_logits.shape),
            "has_nan": bool(torch.isnan(last_logits).any()),
            "has_inf": bool(torch.isinf(last_logits).any()),
            "n_nan": int(torch.isnan(last_logits).sum()),
            "n_inf": int(torch.isinf(last_logits).sum()),
        }

    return findings


def main() -> None:
    print("=== v0.6c-A Finding 2: MDN Categorical simplex violation ===",
          file=sys.stderr)
    print(file=sys.stderr)

    out_dir = pathlib.Path("docs/v0.6c-a")
    out_dir.mkdir(parents=True, exist_ok=True)

    # The paper-config-flagged cell that's most reliable to reproduce
    # is n=5000 seed=0 (the failure happens during data generation
    # ancestral sampling, not during the posterior query).  Try cells
    # in order of rising "likely to reproduce" and stop at the first
    # one that does.
    attempts = [
        (5000, 0),   # known reproducer — fires inside make_synthetic_bn's
                     # ancestral sample chain (MDN.sample at scale).
        (100, 1),    # the canonical paper-flagged cell.
        (100, 3),
        (5000, 1),
        (1000, 0),
        (500, 0),
    ]
    findings: dict = {}
    retries: list[dict] = []
    for n, seed in attempts:
        print(
            f"--- Attempt continuous_nongauss n={n} seed={seed} ---",
            file=sys.stderr,
        )
        try:
            f = reproduce_mdn_simplex_error(n_nodes=n, seed=seed)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            f = {
                "config": {"n_nodes": n, "seed": seed},
                "outer_exception": {
                    "type": type(exc).__name__, "msg": str(exc)[:500],
                },
            }
        retries.append(f)
        if f.get("error"):
            print(
                f"  REPRODUCED in {f.get('error_phase', '?')}: "
                f"{f['error']['type']}: "
                f"{f['error']['msg'][:200]}",
                file=sys.stderr,
            )
            # Build a fresh dict to avoid a circular reference when
            # the same dict is stored both as ``findings`` and inside
            # ``retries[-1]``.
            findings = {**f, "retries": retries[:-1]}
            break
        else:
            print(
                f"  did not reproduce "
                f"(n_cat_probs="
                f"{f.get('n_categorical_constructions_with_probs')})",
                file=sys.stderr,
            )
    if not findings:
        # No reproduction across any attempt; keep the last attempt's
        # data so the markdown can still be populated.  Build a fresh
        # dict (not retries[-1]) to avoid a circular reference when we
        # JSON-dump retries inside it.
        last = retries[-1] if retries else {}
        findings = {**last, "retries": retries}

    print(file=sys.stderr)
    print(
        json.dumps(findings, indent=2, default=str)[:4000],
        file=sys.stderr,
    )

    out_path = out_dir / "mdn_simplex.json"
    out_path.write_text(json.dumps(findings, indent=2, default=str))
    print(f"\nWritten to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
