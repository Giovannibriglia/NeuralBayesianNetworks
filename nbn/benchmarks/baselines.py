"""Wrappers around competitor libraries (pgmpy) for benchmarking."""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch


class PgmpyBaseline:
    """Wraps pgmpy's exact ``VariableElimination`` for benchmarking."""

    def __init__(self, model_pgmpy):
        from pgmpy.inference import VariableElimination
        self._infer = VariableElimination(model_pgmpy)

    def query(self, target: str, evidence: Optional[Dict[str, int]] = None):
        return self._infer.query(variables=[target], evidence=evidence or {}, show_progress=False)

    def time_query(self, target: str, evidence: Optional[Dict[str, int]] = None) -> float:
        t0 = time.perf_counter()
        _ = self.query(target, evidence)
        return time.perf_counter() - t0
