"""Batch-speed rendering under the all/common bar layout
(docs/v0.18-bar-reporting-all-common.md).

A ``batch_sizes`` sweep parquet is detected per family (``sweep_axis ==
"batch_size"``) and renders ``query_time_vs_batch_size.{pdf,tex}`` in both
views under ``<out>/<bench>/<family>/{all,common}/``. Pinned (non-batchable)
baselines only have B=1 cells and read ``--`` beyond; a partially failed
config keeps its solved seeds with a ``(k/n)`` mark instead of failing the
whole cell (the rule that blanked pgmpy in the 2026-09-11 run).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nbn.bench._paper_figures import run_plot


def _row(*, family="discrete", baseline="nbn-cat-ve", batch_size=1, seed=0,
         value=0.01, status="ok", metric="query_time_s"):
    return {
        "benchmark": "synthetic", "family": family, "problem_id": "100",
        "seed": seed, "baseline": baseline, "query_role": "hub",
        "query_kind": "prediction", "metric": metric, "value": value,
        "status": status, "fit_time_s": 1.0, "query_time_s": value,
        "metrics_time_s": 0.0, "error_msg": None, "batch_size": batch_size,
    }


def _make_sweep_df(families=("discrete", "hybrid")) -> pd.DataFrame:
    """Swept nbn-cat-ve at B in {1, 4, 16}; pinned pgmpy at B=1 only."""
    rows = []
    for fam in families:
        for bs in (1, 4, 16):
            for seed in (0, 1):
                rows.append(_row(family=fam, baseline="nbn-cat-ve", batch_size=bs,
                                 seed=seed, value=0.01 / bs))
        for seed in (0, 1):
            rows.append(_row(family=fam, baseline="pgmpy-mle-ve", batch_size=1,
                             seed=seed, value=0.005))
    return pd.DataFrame(rows)


def _render(df: pd.DataFrame, tmp_path: Path, aggregation="iqm_iqr", **kw) -> Path:
    pq = tmp_path / "speed_metrics.parquet"
    df.to_parquet(pq)
    out = tmp_path / "figs"
    assert run_plot(parquet=pq, output_dir=out, aggregation=aggregation, **kw) == 0
    return out


def _table(out: Path, family: str, view: str) -> str:
    return (out / "synthetic" / family / view / "tables"
            / "query_time_vs_batch_size.tex").read_text()


class TestLayout:
    def test_sweep_renders_per_family_both_views(self, tmp_path):
        out = _render(_make_sweep_df(), tmp_path)
        for fam in ("discrete", "hybrid"):
            for view in ("all", "common"):
                base = out / "synthetic" / fam / view
                assert (base / "plots" / "query_time_vs_batch_size.pdf").exists()
                assert (base / "tables" / "query_time_vs_batch_size.tex").exists()
        # no benchmark-level batch_speed.* leftovers from the old layout
        assert not (out / "synthetic" / "batch_speed.pdf").exists()
        assert not list((out / "synthetic").glob("batch_speed_table_*.tex"))

    def test_unbatched_parquet_uses_n_nodes_axis(self, tmp_path):
        df = _make_sweep_df()
        df = df[df["batch_size"] == 1]
        out = _render(df, tmp_path)
        plots = out / "synthetic" / "discrete" / "all" / "plots"
        assert not (plots / "query_time_vs_batch_size.pdf").exists()
        assert (plots / "total_query_time_vs_n_nodes.pdf").exists()


class TestTables:
    def test_pinned_baseline_dashes_beyond_b1_and_bold_best(self, tmp_path):
        out = _render(_make_sweep_df(), tmp_path)
        tex = _table(out, "discrete", "all")
        header = [ln for ln in tex.splitlines() if ln.startswith("Method")][0]
        assert header == "Method & $B=1$ & $B=4$ & $B=16$ \\\\"
        pg = [ln for ln in tex.splitlines() if ln.startswith("pgmpy-mle-ve")][0]
        assert pg.count("--") == 2                     # B=4, B=16 not run
        assert "\\textbf{0.005" in pg                   # pgmpy fastest at B=1
        ve = [ln for ln in tex.splitlines() if ln.startswith("nbn-cat-ve")][0]
        assert "$^\\dagger$" in ve                      # nbn rows are marked
        assert "\\textbf{0.000625" in ve                # best at B=16
        assert "\\label{tab:synthetic_discrete_all_query_time_vs_batch_size}" in tex
        assert "\\begin{table}[t]" in tex and "INFERENCE SPEED" in tex

    def test_partial_failure_keeps_solved_seeds_with_k_over_n(self, tmp_path):
        """The 2026-09-11 rule change: one failed seed no longer blanks the
        cell. pgmpy solved 1 of 2 seeds -> value with (1/2) in all, and the
        common column shrinks to that one seed."""
        df = _make_sweep_df(families=("discrete",))
        df.loc[(df["baseline"] == "pgmpy-mle-ve") & (df["seed"] == 1),
               ["status", "value"]] = ["timeout", float("nan")]
        out = _render(df, tmp_path)
        pg_all = [ln for ln in _table(out, "discrete", "all").splitlines()
                  if ln.startswith("pgmpy-mle-ve")][0]
        assert "0.005" in pg_all and "(1/2)" in pg_all
        common = _table(out, "discrete", "common")
        pg_common = [ln for ln in common.splitlines() if ln.startswith("pgmpy-mle-ve")][0]
        assert "0.005" in pg_common and "(1/2)" not in pg_common
        footer = [ln for ln in common.splitlines() if ln.startswith("$|C|$")][0]
        assert footer == "$|C|$ (common seeds) & 1 & 2 & 2 \\\\"

    def test_all_seeds_failed_shows_code(self, tmp_path):
        df = _make_sweep_df(families=("discrete",))
        mask = df["baseline"] == "pgmpy-mle-ve"
        df.loc[mask, ["status", "value"]] = ["timeout", float("nan")]
        out = _render(df, tmp_path)
        for view in ("all", "common"):
            pg = [ln for ln in _table(out, "discrete", view).splitlines()
                  if ln.startswith("pgmpy-mle-ve")][0]
            assert pg.split("&")[1].strip() == "timeout"
        # a DNF method does not empty the common column
        footer = [ln for ln in _table(out, "discrete", "common").splitlines()
                  if ln.startswith("$|C|$")][0]
        assert footer == "$|C|$ (common seeds) & 2 & 2 & 2 \\\\"

    def test_batchable_oom_above_b1_keeps_smaller_batches(self, tmp_path):
        df = _make_sweep_df(families=("discrete",))
        mask = (df["baseline"] == "nbn-cat-ve") & (df["batch_size"] == 16)
        df.loc[mask, ["status", "value"]] = ["oom", float("nan")]
        out = _render(df, tmp_path)
        ve = [ln for ln in _table(out, "discrete", "all").splitlines()
              if ln.startswith("nbn-cat-ve")][0]
        cells = [c.strip() for c in ve.split("&")[1:]]
        assert "0.01" in cells[0] and "0.0025" in cells[1]
        assert cells[2].rstrip("\\").strip() == "oom"

    @pytest.mark.parametrize("aggregation", ["iqm_iqr", "mean_std"])
    def test_aggregation_flag_changes_band(self, tmp_path, aggregation):
        df = _make_sweep_df(families=("discrete",))
        df.loc[(df["baseline"] == "nbn-cat-ve") & (df["seed"] == 1), "value"] *= 3
        out = _render(df, tmp_path, aggregation=aggregation)
        ve = [ln for ln in _table(out, "discrete", "all").splitlines()
              if ln.startswith("nbn-cat-ve")][0]
        b1 = ve.split("&")[1]
        # seeds 0.01 and 0.03: mean_std -> 0.02±0.01; iqm_iqr -> 0.02±0.005
        assert ("0.02$\\pm$0.01" in b1) == (aggregation == "mean_std")
        assert ("0.02$\\pm$0.005" in b1) == (aggregation == "iqm_iqr")


class TestSelection:
    def test_top_nbn_limits_nbn_rows(self, tmp_path):
        df = _make_sweep_df(families=("discrete",))
        extra = []
        for name, v in (("nbn-cat-lw", 0.05), ("nbn-cat-ais", 0.02)):
            for bs in (1, 4, 16):
                for seed in (0, 1):
                    extra.append(_row(baseline=name, batch_size=bs, seed=seed, value=v / bs))
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
        out = _render(df, tmp_path, top_nbn=1)
        tex = _table(out, "discrete", "all")
        assert "nbn-cat-ve" in tex and "nbn-cat-ais" not in tex and "nbn-cat-lw" not in tex
        sel = (out / "synthetic" / "discrete" / "selection.txt").read_text()
        assert "all\tquery_time\tbatch_size=16\tnbn-cat-ve" in sel
        (tmp_path / "two").mkdir()
        out2 = _render(df, tmp_path / "two", top_nbn=2)
        tex2 = _table(out2, "discrete", "all")
        assert "nbn-cat-ais" in tex2 and "nbn-cat-lw" not in tex2

    def test_common_seeds_sidecar(self, tmp_path):
        out = _render(_make_sweep_df(families=("discrete",)), tmp_path)
        txt = (out / "synthetic" / "discrete" / "common" / "common_seeds.txt").read_text()
        assert "query_time\tbatch_size=1\t|C|=2\t100/s0, 100/s1" in txt


class TestSpeedSmokeRoundTrip:
    def test_render_from_speed_smoke_run(self, tmp_path):
        """The committed inference_speed_smoke run (if present) renders."""
        root = Path(__file__).resolve().parents[2]
        runs = sorted(root.glob("results/benchmark_synthetic_batch_speed_*"))
        runs = [r for r in runs
                if any(p.stat().st_size < 5_000_000 for p in r.glob("*_metrics.parquet"))]
        if not runs:
            pytest.skip("no small (smoke-sized) batch_speed run directory present")
        out = tmp_path / "figs"
        assert run_plot(parquet=runs[-1], output_dir=out, aggregation="iqm_iqr") == 0
        assert list(out.rglob("query_time_vs_batch_size.pdf"))


class TestEvidenceModeSplit:
    """Runs whose queries mix ``full`` / ``empty`` evidence render every
    query-derived metric once per mode next to the combined figure: nbn's
    batched path only covers full evidence (empty batches fall back to
    sequential), so the combined per-query time hides the batching gain."""

    @staticmethod
    def _mixed_df() -> pd.DataFrame:
        rows = []
        for bs in (1, 16):
            for seed in (0, 1):
                # full evidence batches (16x at B=16); empty evidence does not
                rows.append({**_row(batch_size=bs, seed=seed, value=0.016 / bs),
                             "evidence_mode": "full"})
                rows.append({**_row(batch_size=bs, seed=seed, value=0.016),
                             "evidence_mode": "empty"})
                # a cell-level fit row: no query_role, default "full" mode
                rows.append({**_row(batch_size=bs, seed=seed, value=1.0,
                                    metric="fit_time_s"),
                             "query_role": "", "evidence_mode": "full"})
        return pd.DataFrame(rows)

    @staticmethod
    def _b16(tex: str) -> str:
        line = [ln for ln in tex.splitlines() if ln.startswith("nbn-cat-ve")][0]
        cell = line.split("&")[2].strip().rstrip("\\").strip()
        return cell.removeprefix("\\textbf{").removesuffix("}")

    def test_per_mode_figures_and_tables(self, tmp_path):
        out = _render(self._mixed_df(), tmp_path)
        base = out / "synthetic" / "discrete"
        for view in ("all", "common"):
            for stem in ("query_time_vs_batch_size",
                         "query_time_vs_batch_size_evidence_full",
                         "query_time_vs_batch_size_evidence_empty"):
                assert (base / view / "plots" / f"{stem}.pdf").exists(), stem
                assert (base / view / "tables" / f"{stem}.tex").exists(), stem
        tables = base / "all" / "tables"
        assert self._b16((tables / "query_time_vs_batch_size_evidence_full.tex").read_text()) \
            .startswith("0.001")
        assert self._b16((tables / "query_time_vs_batch_size_evidence_empty.tex").read_text()) \
            .startswith("0.016")
        # combined = mean of the two modes, the midpoint that motivated the split
        assert self._b16((tables / "query_time_vs_batch_size.tex").read_text()) \
            .startswith("0.0085")
        full_tex = (tables / "query_time_vs_batch_size_evidence_full.tex").read_text()
        assert "(evidence=full)" in full_tex
        assert "_evidence_full}" in full_tex          # distinct \label per mode
        sel = (base / "selection.txt").read_text()
        assert "all\tquery_time [evidence=full]\tbatch_size=16\tnbn-cat-ve" in sel

    def test_single_mode_run_has_no_per_mode_files(self, tmp_path):
        df = _make_sweep_df(families=("discrete",))
        df["evidence_mode"] = "full"
        out = _render(df, tmp_path)
        assert not list(out.rglob("*_evidence_*"))

    def test_cell_sentinel_fails_every_mode(self, tmp_path):
        """A whole-cell failure sentinel (query_role "") marks the seed
        unsolved in both per-mode views, not only the default 'full' one."""
        df = self._mixed_df()
        df = pd.concat([df, pd.DataFrame([{
            **_row(batch_size=16, seed=1, value=float("nan"), metric="status",
                   status="timeout"),
            "query_role": "", "evidence_mode": "full"}])], ignore_index=True)
        out = _render(df, tmp_path)
        tables = out / "synthetic" / "discrete" / "all" / "tables"
        for mode in ("full", "empty"):
            tex = (tables / f"query_time_vs_batch_size_evidence_{mode}.tex").read_text()
            assert "(1/2)" in self._b16(tex), mode
