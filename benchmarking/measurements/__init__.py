"""Measurement implementations for v0.13 benchmarks.

Currently shipped:
    AccuracyAndTiming — per-query TV, JSD, W₁ accuracy metrics plus
        individually isolated ``query_time_s``.  Used by synthetic and
        bnlearn benchmarks.
    TimingOnly — ``query_time_s`` only; no accuracy computation.  Used
        by the scalability benchmark.

Reference: docs/v0.13-benchmark-redesign.md §3, §4.1
"""
from benchmarking.measurements.accuracy_timing import AccuracyAndTiming
from benchmarking.measurements.timing_only import TimingOnly

__all__ = ["AccuracyAndTiming", "TimingOnly"]
