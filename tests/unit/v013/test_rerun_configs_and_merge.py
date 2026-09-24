"""v0.19.0 partial-rerun configs + ``nbn-bench merge`` (#286, #288).

The rerun configs must reproduce the original runs' problems cell for cell —
only the family / network list and the baseline subset may differ — or the
merged parquet would mix incomparable cells.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from nbn.bench._merge import CELL_KEY, merge_runs
from nbn.bench.cli import main
from nbn.bench.core.yaml_config import load_runner_config

_C = Path("nbn/bench/configs")
_RERUNS = {
    # rerun config → (original, kept families/networks, expected baseline filter)
    "synthetic/reruns/complete_continuous.yaml":
        ("synthetic/complete/inference_complete.yaml", "continuous"),
    "synthetic/reruns/complete_discrete_avi.yaml":
        ("synthetic/complete/inference_complete.yaml", "cat_avi"),
    "synthetic/reruns/scalability_continuous.yaml":
        ("synthetic/complete/inference_scalability_complete.yaml", "continuous"),
    "synthetic/reruns/scalability_discrete_avi.yaml":
        ("synthetic/complete/inference_scalability_complete.yaml", "cat_avi"),
    "synthetic/reruns/batch_speed_avi.yaml":
        ("synthetic/speed/inference_speed.yaml", "avi"),
    "bnlearn/reruns/bnlearn_continuous.yaml":
        ("bnlearn/complete/inference_complete.yaml", "continuous"),
    "bnlearn/reruns/bnlearn_discrete_avi.yaml":
        ("bnlearn/complete/inference_complete.yaml", "cat_avi"),
}
_DISCRETE_ONLY = {("nbn", "cat"), ("pomegranate", "discrete"), ("pgmpy", "discrete")}


def _keep(kind: str, b: dict) -> bool:
    if kind == "continuous":
        return (b["library"], b["mechanism"]) not in _DISCRETE_ONLY
    if kind == "cat_avi":
        return b["mechanism"] == "cat" and b.get("inference_method") == "avi"
    return b.get("inference_method") == "avi"


@pytest.mark.parametrize("rerun", sorted(_RERUNS))
def test_rerun_config_matches_original(rerun: str, tmp_path: Path) -> None:
    orig_name, kind = _RERUNS[rerun]
    new, orig = (yaml.safe_load((_C / p).read_text()) for p in (rerun, orig_name))
    # Everything except the name, the family/network list and the baselines
    # is identical, so problems, seeds, queries and budgets match.
    for d in (new, orig):
        d.pop("config_name")
        (d["source"].pop("families", None), d["source"].pop("networks", None))
    new_b, orig_b = new.pop("baselines"), orig.pop("baselines")
    assert new == orig
    assert new_b == [b for b in orig_b if _keep(kind, b)] and new_b
    load_runner_config(_C / rerun, jsonl_path=tmp_path / "o.jsonl")


def test_rerun_families_cover_the_fixes() -> None:
    fam = lambda p: yaml.safe_load((_C / p).read_text())["source"]  # noqa: E731
    assert fam("synthetic/reruns/complete_continuous.yaml")["families"] == [
        "continuous_lg", "continuous_nongauss"]
    assert fam("synthetic/reruns/complete_discrete_avi.yaml")["families"] == ["discrete"]
    orig = fam("bnlearn/complete/inference_complete.yaml")["networks"]
    cont = fam("bnlearn/reruns/bnlearn_continuous.yaml")["networks"]
    disc = fam("bnlearn/reruns/bnlearn_discrete_avi.yaml")["networks"]
    assert sorted(cont + disc) == sorted(orig) and not set(cont) & set(disc)
    assert set(cont) == {"ecoli70", "arth150", "healthcare", "sangiovese", "mehra"}


def _rows(cells, value, extra=None):
    out = []
    for fam, pid, seed, bl in cells:
        for metric in ("w1_per_node", "query_time_s"):
            out.append({"benchmark": "synthetic", "family": fam, "problem_id": pid,
                        "seed": seed, "baseline": bl, "metric": metric,
                        "value": value, "status": "ok", **(extra or {})})
    return pd.DataFrame(out)


def test_merge_replaces_only_rerun_cells(tmp_path: Path) -> None:
    base_cells = [("discrete", "10", 0, "nbn-cat-ve"), ("discrete", "10", 0, "nbn-cat-avi"),
                  ("continuous_lg", "10", 0, "nbn-lg-lw"), ("continuous_lg", "10", 1, "nbn-lg-lw")]
    _rows(base_cells, 1.0).to_parquet(tmp_path / "base_metrics.parquet")
    upd1 = [("continuous_lg", "10", 0, "nbn-lg-lw"), ("continuous_lg", "10", 1, "nbn-lg-lw")]
    upd2 = [("discrete", "10", 0, "nbn-cat-avi")]
    (tmp_path / "r1").mkdir()
    _rows(upd1, 2.0, {"newcol": 1}).to_parquet(tmp_path / "r1" / "x_metrics.parquet")
    _rows(upd2, 3.0).to_parquet(tmp_path / "upd2.parquet")

    out = tmp_path / "merged_metrics.parquet"
    stats = merge_runs(tmp_path / "base_metrics.parquet",
                       [tmp_path / "r1", tmp_path / "upd2.parquet"], out)
    m = pd.read_parquet(out)
    assert len(m) == 8 and stats["dropped"] == 6 and stats["added"] == 6
    assert stats["new_cells"] == 0
    v = m.groupby(list(CELL_KEY)).value.agg(set)
    assert v[("synthetic", "discrete", "10", 0, "nbn-cat-ve")] == {1.0}
    assert v[("synthetic", "discrete", "10", 0, "nbn-cat-avi")] == {3.0}
    assert v[("synthetic", "continuous_lg", "10", 1, "nbn-lg-lw")] == {2.0}


def test_merge_cli(tmp_path: Path) -> None:
    _rows([("discrete", "10", 0, "a")], 1.0).to_parquet(tmp_path / "b.parquet")
    _rows([("discrete", "10", 0, "a")], 5.0).to_parquet(tmp_path / "u.parquet")
    out = tmp_path / "m_metrics.parquet"
    assert main(["merge", str(tmp_path / "b.parquet"), str(tmp_path / "u.parquet"),
                 "-o", str(out)]) == 0
    assert set(pd.read_parquet(out).value) == {5.0}
