"""The oom triage must separate "too big" from "never started".

Both land as ``status='oom'``, and only one says anything about the baseline.
The negative direction is the one that matters: a run whose oom rows are
cells killed during startup measured the per-cell memory cap, not the model,
and mistaking those for genuine OOMs means reporting a capability limit that
does not exist.
"""
from __future__ import annotations

import json

import pytest

from benchmarking.diagnostics.triage_oom_rows import _classify, main, triage

_NEVER_STARTED = "subprocess produced no output rows (classification=killed, exit_code=-9)"
_VE_GUARD = (
    "TensorVariableElimination: query out of memory pre-allocation guard — "
    "plan would need ~24.00 GiB peak intermediate factor at order='min_fill'"
)


@pytest.mark.parametrize(
    ("msg", "spurious"),
    [
        (_NEVER_STARTED, True),
        ("", True),          # a diagnosed OOM always says who diagnosed it
        (None, True),
        (_VE_GUARD, False),
        ("CUDA out of memory. Tried to allocate 2.00 GiB", False),
        ("MemoryError()", False),
        ("subprocess killed (exit_code=-9)", False),
        ("skipped: config failed on an earlier seed", False),
    ],
)
def test_classification(msg, spurious):
    assert _classify(msg)[1] is spurious


def _write(tmp_path, rows, name="metrics.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def test_a_run_killed_at_startup_is_flagged(tmp_path):
    """The bnlearn-shaped case: almost every cell dies before doing any work."""
    rows = [{"status": "oom", "error_msg": _NEVER_STARTED} for _ in range(81)]
    rows += [{"status": "ok", "error_msg": None} for _ in range(19)]
    r = triage(_write(tmp_path, rows))
    assert r["oom"] == 81
    assert r["spurious_fraction"] == 1.0
    assert "never-started" in next(iter(r["breakdown"]))


def test_a_run_with_genuine_ooms_is_not_flagged(tmp_path):
    rows = [{"status": "oom", "error_msg": _VE_GUARD} for _ in range(40)]
    rows += [{"status": "ok", "error_msg": None} for _ in range(60)]
    r = triage(_write(tmp_path, rows))
    assert r["oom"] == 40
    assert r["spurious_fraction"] == 0.0


def test_mixed_run_reports_the_proportion(tmp_path):
    rows = [{"status": "oom", "error_msg": _NEVER_STARTED} for _ in range(3)]
    rows += [{"status": "oom", "error_msg": _VE_GUARD} for _ in range(1)]
    r = triage(_write(tmp_path, rows))
    assert r["spurious_fraction"] == pytest.approx(0.75)


def test_inherited_skips_do_not_dilute_the_fraction(tmp_path):
    """20 propagated skips must not make 3-of-3 startup kills look like 3-of-23."""
    rows = [{"status": "oom", "error_msg": _NEVER_STARTED} for _ in range(3)]
    rows += [
        {"status": "oom", "error_msg": "skipped: config failed on an earlier seed"}
        for _ in range(20)
    ]
    r = triage(_write(tmp_path, rows))
    assert r["oom"] == 23
    assert r["attributable"] == 3
    assert r["spurious_fraction"] == 1.0


def test_rows_without_oom_are_ignored(tmp_path):
    rows = [{"status": s, "error_msg": None} for s in ("ok", "timeout", "not_supported")]
    r = triage(_write(tmp_path, rows))
    assert r["oom"] == 0 and r["spurious_fraction"] == 0.0


def test_parquet_and_jsonl_agree(tmp_path):
    pd = pytest.importorskip("pandas")
    rows = [{"status": "oom", "error_msg": _NEVER_STARTED} for _ in range(5)]
    rows += [{"status": "ok", "error_msg": None} for _ in range(5)]
    jsonl = _write(tmp_path, rows)
    parquet = str(tmp_path / "m.parquet")
    pd.DataFrame(rows).to_parquet(parquet)
    a, b = triage(jsonl), triage(parquet)
    assert (a["rows"], a["oom"], a["spurious"]) == (b["rows"], b["oom"], b["spurious"])


def test_cli_runs_and_reports(tmp_path, capsys):
    rows = [{"status": "oom", "error_msg": _NEVER_STARTED} for _ in range(4)]
    _write(tmp_path, rows)
    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "never-started" in out and "re-run" in out


def test_cli_on_an_empty_directory_reports_rather_than_crashing(tmp_path):
    assert main([str(tmp_path)]) == 1
