"""Soft CPU throughput gate for the v0.3.1 vectorised ``query_batch``.

After §1.4, NBN's TensorVariableElimination on ``alarm`` (37 nodes) at
B=1024 should be at least as fast as pgmpy's batched VariableElimination
on the same evidence, on a single CPU. The gate is *soft*: we measure
both and only assert nbn_ve >= pgmpy (with a 10% slack to absorb timing
jitter on shared CI runners).

If this test fails, do not paper over with a larger slack — the whole
point of v0.3.1 was to make the cross-over hold honestly.
"""
from __future__ import annotations

import time

import pytest
import torch

pytest.importorskip("pgmpy")

import sys
from pathlib import Path

# Re-use the helper from the correctness test rather than duplicating it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
from test_vectorized_query_batch_correctness import (  # noqa: E402
    _build_nbn_from_pgmpy,
    _sample_targets_and_evidence,
)
from benchmarking.domains.bnlearn import _load_pgmpy_model  # noqa: E402

from nbn import TensorVariableElimination  # noqa: E402


def _bench_nbn(model, engine, targets, evidence, *, n_warmup: int = 1, n_trials: int = 3) -> float:
    for _ in range(n_warmup):
        engine.query_batch(model, targets, evidence)
    t0 = time.perf_counter()
    for _ in range(n_trials):
        engine.query_batch(model, targets, evidence)
    elapsed = (time.perf_counter() - t0) / n_trials
    return float(evidence[next(iter(evidence))].shape[0]) / max(elapsed, 1e-9)


def _bench_pgmpy(bn, target: str, evidence_dict_list: list[dict[str, int]]) -> float:
    """Throughput for pgmpy looping VE over a per-row evidence dict."""
    from pgmpy.inference import VariableElimination
    infer = VariableElimination(bn)

    # Warmup
    infer.query(variables=[target], evidence=evidence_dict_list[0], show_progress=False)

    t0 = time.perf_counter()
    for ev in evidence_dict_list:
        infer.query(variables=[target], evidence=ev, show_progress=False)
    elapsed = time.perf_counter() - t0
    return len(evidence_dict_list) / max(elapsed, 1e-9)


@pytest.mark.slow
def test_alarm_b1024_nbn_ve_beats_pgmpy() -> None:
    """nbn_ve should match or beat pgmpy on alarm B=1024 (CPU)."""
    bn = _load_pgmpy_model("alarm")
    model = _build_nbn_from_pgmpy(bn)
    engine = TensorVariableElimination()

    targets, evidence = _sample_targets_and_evidence(
        model, n_targets=1, n_evidence=3, B=1024, seed=20251101,
    )
    target = targets[0]

    # Build pgmpy evidence list. pgmpy expects state names; CategoricalTable
    # stores integer indices. Map int→state_name for each evidence node.
    cpd_states: dict[str, list] = {}
    for v in evidence:
        cpd_states[v] = list(bn.get_cpds(v).state_names[v])
    ev_list: list[dict[str, str]] = []
    for b in range(1024):
        ev_list.append({v: cpd_states[v][int(evidence[v][b].item())] for v in evidence})

    nbn_qps = _bench_nbn(model, engine, targets, evidence)
    # pgmpy is the reference loop-baseline. Cap at 64 rows to keep CPU runtime
    # reasonable on CI; throughput is rate, so the comparison is fair.
    pgmpy_qps = _bench_pgmpy(bn, target, ev_list[:64])

    print(f"nbn_ve  qps = {nbn_qps:7.1f}")
    print(f"pgmpy   qps = {pgmpy_qps:7.1f}")
    assert nbn_qps >= 0.9 * pgmpy_qps, (
        f"NBN must match-or-beat pgmpy on alarm B=1024: "
        f"got nbn_ve={nbn_qps:.1f} q/s vs pgmpy={pgmpy_qps:.1f} q/s "
        f"(threshold = 0.9 × pgmpy)."
    )
