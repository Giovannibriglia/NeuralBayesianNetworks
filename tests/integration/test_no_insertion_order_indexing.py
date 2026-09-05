"""CI-enforced grep test for the v0.6a column-order canonicalization.

Any future commit that adds ``list(*.dag.nodes()).index(...)`` (or
equivalent insertion-order indexing) to ``nbn/bench/`` will fail
this test, forcing the author to use ``bn.column_index`` instead.

This retires the entire schema-bug class, not just the specific
failing site fixed in PR #14 commit 1bd671e.
"""
from __future__ import annotations

import pathlib
import re

# Patterns that indicate insertion-order column indexing.  These are
# bug candidates — the canonical helper ``bn.column_index(name)``
# should be used instead.
FORBIDDEN_PATTERNS = [
    r"list\(.*\.dag\.nodes\(\)\)\.index\(",
    r"list\(.*\.nodes\(\)\)\.index\(",
    r"\.dag\.nodes\(\)\)\.index\(",
]

# Files allowed to use these patterns (e.g. internal to synthetic.py
# where the canonical column_order itself is computed).
ALLOWLIST = {
    "nbn/bench/synthetic.py",
}


def test_no_insertion_order_indexing_in_benchmarking() -> None:
    repo = pathlib.Path(__file__).resolve().parents[2]
    bench_dir = repo / "nbn" / "bench"
    violations: list[str] = []
    for py_file in bench_dir.rglob("*.py"):
        rel = py_file.relative_to(repo).as_posix()
        if rel in ALLOWLIST:
            continue
        text = py_file.read_text()
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            # Skip comments and obvious docstring lines so this test
            # does not flag prose that *describes* the forbidden pattern
            # (e.g. migration commentary explaining what NOT to do).
            if (
                stripped.startswith("#")
                or stripped.startswith('"""')
                or stripped.startswith("'''")
            ):
                continue
            for pattern in FORBIDDEN_PATTERNS:
                if re.search(pattern, line):
                    violations.append(f"{rel}:{line_no}: {line.strip()}")
    assert not violations, (
        "Found insertion-order column indexing in nbn/bench/.\n"
        "Use bn.column_index(name) instead of "
        "list(bn.dag.nodes()).index(name).\n"
        "Violations:\n  " + "\n  ".join(violations)
    )
