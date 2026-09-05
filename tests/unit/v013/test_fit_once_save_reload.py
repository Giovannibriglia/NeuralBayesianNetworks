"""Tests for fit-once-save-reload (#191 Path 2).

Baselines sharing a fit-identity (same library/mechanism/epochs/fit-data,
differing ONLY in inference_method) fit the base model ONCE: the first writes it
with torch.save, the rest reload it. nbn-only; pgmpy/pomegranate/pyro run
standalone unchanged.

Layers, fastest first:
  * the SINGLE-SOURCE key + filename helpers (_fit_identity_key,
    _fit_cache_filename) and the grouping pre-pass (_assign_fit_roles) — pure,
    no fit;
  * cell_worker fit/reload/standalone branch in-process (real fit, no
    subprocess) — reload-skips-fit, cache-miss fallback, save-on-fit, flow
    pickle round-trip (exercises the #192 fix through the save/load path);
  * the HEADLINE grouped==ungrouped guarantee + cache cleanup through the REAL
    runner and REAL subprocesses.

Reference: docs/v0.14-fit-once-query-many-design.md; issue #191.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
import torch

from nbn.bench.core.config import BaselineSpec, RunnerConfig
from nbn.bench.core.runner import (
    Runner,
    _assign_fit_roles,
    _fit_cache_filename,
    _fit_identity_key,
)
from nbn.bench.domains.base import BenchmarkProblem, GroundTruth
from nbn.bench.measurements import AccuracyAndTiming, TimingOnly
from nbn.bench.selectors.uniform import UniformRandomSelector


# --- fixtures ----------------------------------------------------------------

def _problem(seed: int, *, family: str = "discrete", n_nodes: int = 4,
             pid: str | None = None) -> BenchmarkProblem:
    nodes = [f"X{i}" for i in range(n_nodes)]
    return BenchmarkProblem(
        name=f"{family}_n{n_nodes}_seed{seed}",
        dag=[(nodes[i], nodes[i + 1]) for i in range(n_nodes - 1)],
        variables=dict.fromkeys(nodes, ("discrete", 4)),
        train_data={n: torch.randint(0, 4, (64,)).float() for n in nodes},
        test_data={n: torch.randint(0, 4, (16,)).float() for n in nodes},
        queries=[],
        ground_truth=GroundTruth(),
        family=family,
        problem_id=pid or str(n_nodes),
        seed=seed,
    )


def _spec(mechanism="cat", engine="ve", **extra):
    return BaselineSpec(
        library="nbn", mechanism=mechanism, param_method="mle",
        inference_method=engine,
        extra_kwargs=extra or {},
    )


class _ListSource:
    def __init__(self, problems):
        self._problems = problems

    def iter_problems(self, _cfg) -> Iterator[BenchmarkProblem]:
        yield from self._problems


# =====================================================================
# 1. Single source of truth: the fit-identity key + cache filename
# =====================================================================

class TestFitIdentityKey:
    def test_ve_and_lw_share_a_key(self):
        p = _problem(0)
        assert _fit_identity_key(_spec("cat", "ve"), p) == \
               _fit_identity_key(_spec("cat", "lw"), p)

    def test_n_samples_and_device_excluded(self):
        p = _problem(0)
        a = BaselineSpec(library="nbn", mechanism="cat", param_method="mle",
                         inference_method="ve", device="cpu")
        b = BaselineSpec(library="nbn", mechanism="cat", param_method="mle",
                         inference_method="lw", device="cuda",
                         extra_kwargs={"n_samples": 4096})
        assert _fit_identity_key(a, p) == _fit_identity_key(b, p)

    def test_epochs_splits_the_key(self):
        p = _problem(0)
        assert _fit_identity_key(_spec("cat", "ve"), p) != \
               _fit_identity_key(_spec("cat", "ais", epochs=2), p)

    def test_absent_budget_is_distinct_from_explicit(self):
        # PR A: absent epochs/batch_size/lr = "mechanism-designed budget",
        # a DISTINCT fit identity from any explicit value (the old key
        # collapsed absent epochs to 20).
        p = _problem(0)
        base = _fit_identity_key(_spec("cat", "ve"), p)
        assert base != _fit_identity_key(_spec("cat", "ve", epochs=20), p)
        assert base != _fit_identity_key(_spec("cat", "ve", batch_size=256), p)
        assert base != _fit_identity_key(_spec("cat", "ve", lr=1e-3), p)
        # filename formatting tolerates the None fields
        name = _fit_cache_filename(base)
        assert name.startswith("fit_") and name.endswith(".pt")

    def test_mechanism_and_problem_split_the_key(self):
        p0, p1 = _problem(0), _problem(1)
        base = _fit_identity_key(_spec("cat", "ve"), p0)
        assert base != _fit_identity_key(_spec("neuralcat", "ve"), p0)  # mech
        assert base != _fit_identity_key(_spec("cat", "ve"), p1)        # seed

    def test_cache_filename_derives_from_key_and_is_stable(self):
        p = _problem(0)
        kv = _fit_identity_key(_spec("cat", "ve"), p)
        kl = _fit_identity_key(_spec("cat", "lw"), p)
        # ve and lw share the key -> the fitter's save path and the reloader's
        # load path are the SAME file. This is the no-drift guarantee.
        assert _fit_cache_filename(kv) == _fit_cache_filename(kl)
        assert _fit_cache_filename(kv).startswith("fit_")
        assert _fit_cache_filename(kv).endswith(".pt")


# =====================================================================
# 2. Grouping pre-pass: _assign_fit_roles
# =====================================================================

def _cfg(baselines, *, batch_sizes=None, tmp_path=None):
    return RunnerConfig(
        benchmark="synthetic", config_name="t",
        problem_source=_ListSource([]), source_config=None,
        selector=UniformRandomSelector(), measurement=TimingOnly(),
        baselines=baselines, n_queries_per_cell=1, per_cell_timeout_s=60.0,
        batch_sizes=batch_sizes,
        jsonl_path=(tmp_path or Path("/tmp")) / "out.jsonl",
    )


class TestAssignFitRoles:
    def _dir(self, tmp_path):
        return lambda: tmp_path

    def test_group_of_two_fit_then_reload(self, tmp_path):
        cfg = _cfg([_spec("cat", "ve"), _spec("cat", "lw")])
        roles, delete_after = _assign_fit_roles(
            cfg, _problem(0), {}, False, self._dir(tmp_path))
        assert roles[0][0] == "fit"
        assert roles[1][0] == "reload"
        assert roles[0][1] == roles[1][1]          # same cache path
        assert delete_after == {1: roles[0][1]}    # deleted after last member

    def test_singleton_is_standalone(self, tmp_path):
        cfg = _cfg([_spec("lg", "lw")])
        roles, delete_after = _assign_fit_roles(
            cfg, _problem(0, family="continuous_lg"), {}, False,
            self._dir(tmp_path))
        assert roles[0] == ("standalone", None)
        assert delete_after == {}

    def test_non_nbn_never_grouped(self, tmp_path):
        # Two pgmpy baselines that WOULD share a key if grouped — must not be.
        cfg = _cfg([
            BaselineSpec(library="pgmpy", mechanism="discrete",
                         param_method="mle", inference_method="ve"),
            BaselineSpec(library="pgmpy", mechanism="discrete",
                         param_method="bayes", inference_method="ve"),
            _spec("cat", "ve"), _spec("cat", "lw"),
        ])
        roles, delete_after = _assign_fit_roles(
            cfg, _problem(0), {}, False, self._dir(tmp_path))
        assert roles[0] == ("standalone", None)    # pgmpy
        assert roles[1] == ("standalone", None)    # pgmpy
        assert roles[2][0] == "fit"                # nbn group still forms
        assert roles[3][0] == "reload"

    def test_three_member_group_one_fit_two_reload(self, tmp_path):
        cfg = _cfg([_spec("cat", "ve"), _spec("cat", "lw"),
                    _spec("cat", "ais")])
        roles, delete_after = _assign_fit_roles(
            cfg, _problem(0), {}, False, self._dir(tmp_path))
        assert [r[0] for r in roles] == ["fit", "reload", "reload"]
        assert delete_after == {2: roles[0][1]}    # after the LAST reloader

    def test_seed_skipped_fitter_promotes_next_live_member(self, tmp_path):
        # The would-be fitter (cat-ve) is fully seed-skipped: its only batch
        # size already failed on an earlier seed. The next live member (cat-lw)
        # must become the fitter, NOT be left a reloader with no cache.
        cfg = _cfg([_spec("cat", "ve"), _spec("cat", "lw"), _spec("cat", "ais")],
                   batch_sizes=[1])
        failed = {("discrete", "4", "nbn-cat-ve", 1): "oom"}
        roles, delete_after = _assign_fit_roles(
            cfg, _problem(0), failed, True, self._dir(tmp_path))
        assert roles[0] == ("standalone", None)    # cat-ve skipped -> not live
        assert roles[1][0] == "fit"                # cat-lw promoted to fitter
        assert roles[2][0] == "reload"
        assert delete_after == {2: roles[1][1]}

    def test_lazy_dir_not_made_when_no_group(self, tmp_path):
        made = {"n": 0}

        def _dir():
            made["n"] += 1
            return tmp_path

        cfg = _cfg([_spec("lg", "lw")])   # singleton only
        _assign_fit_roles(cfg, _problem(0, family="continuous_lg"), {}, False,
                          _dir)
        assert made["n"] == 0              # never created the cache dir


# =====================================================================
# 3. cell_worker fit / reload / standalone branch (in-process, real fit)
# =====================================================================

def _worker_ctx(problem, spec, *, fit_role, cache_path, batch_sizes=(1,)):
    return {
        "problem": problem,
        "baseline_spec": spec,
        "seed": problem.seed,
        "selector": UniformRandomSelector(),
        "measurement": TimingOnly(),
        "benchmark": "synthetic",
        "n_queries_per_cell": 2,
        "per_cell_timeout_s": 60.0,
        "fit_budget_s": 1000.0,
        "default_role": "random",
        "batch_sizes": list(batch_sizes),
        "fit_role": fit_role,
        "cache_path": str(cache_path) if cache_path is not None else None,
    }


class TestCellWorkerBranch:
    def test_fit_role_writes_cache(self, tmp_path):
        from nbn.bench.core import cell_worker
        cache = tmp_path / "fit_x.pt"
        rows = cell_worker._run_cell(
            _worker_ctx(_problem(0), _spec("cat", "ve"),
                        fit_role="fit", cache_path=cache))
        assert cache.exists()                      # fitter saved the base
        assert all(r["status"] != "error" for r in rows)

    def test_reload_skips_fit_and_uses_base(self, tmp_path, monkeypatch):
        from nbn.bench.core import cell_worker
        from nbn.bench.adapters.nbn_adapter import NBNAdapter

        # First, produce a real cache via a fit-role cell.
        cache = tmp_path / "fit_y.pt"
        cell_worker._run_cell(_worker_ctx(_problem(0), _spec("cat", "ve"),
                                          fit_role="fit", cache_path=cache))
        assert cache.exists()

        # Now a reload-role cell must NOT call fit(), only load_base_and_attach.
        calls = {"fit": 0, "load": 0}
        real_load = NBNAdapter.load_base_and_attach

        def spy_fit(self, *a, **k):
            calls["fit"] += 1
            raise AssertionError("reload path must not call fit()")

        def spy_load(self, *a, **k):
            calls["load"] += 1
            return real_load(self, *a, **k)

        monkeypatch.setattr(NBNAdapter, "fit", spy_fit)
        monkeypatch.setattr(NBNAdapter, "load_base_and_attach", spy_load)
        rows = cell_worker._run_cell(
            _worker_ctx(_problem(0), _spec("cat", "lw"),
                        fit_role="reload", cache_path=cache))
        assert calls == {"fit": 0, "load": 1}
        assert all(r["status"] != "error" for r in rows)

    def test_reload_cache_miss_falls_back_to_fit(self, tmp_path, monkeypatch):
        from nbn.bench.core import cell_worker
        from nbn.bench.adapters.nbn_adapter import NBNAdapter

        missing = tmp_path / "does_not_exist.pt"
        calls = {"fit": 0, "load": 0}
        real_fit = NBNAdapter.fit

        def spy_fit(self, *a, **k):
            calls["fit"] += 1
            return real_fit(self, *a, **k)

        def spy_load(self, *a, **k):
            calls["load"] += 1
            raise AssertionError("must not load a missing cache")

        monkeypatch.setattr(NBNAdapter, "fit", spy_fit)
        monkeypatch.setattr(NBNAdapter, "load_base_and_attach", spy_load)
        rows = cell_worker._run_cell(
            _worker_ctx(_problem(0), _spec("cat", "lw"),
                        fit_role="reload", cache_path=missing))
        # cache miss -> fell back to a standalone fit, no crash, valid rows.
        assert calls == {"fit": 1, "load": 0}
        assert all(r["status"] != "error" for r in rows)

    def test_flow_group_pickle_round_trip(self, tmp_path):
        # Exercises the #192 pickle fix through the worker's save/load path: a
        # fitted flow base model is saved by the fitter and reloaded by a
        # sibling. continuous_lg family so flow is applicable.
        from nbn.bench.core import cell_worker
        prob = _problem(0, family="continuous_lg")
        prob = BenchmarkProblem(
            name=prob.name, dag=prob.dag,
            variables=dict.fromkeys(prob.variables, ("continuous", 1)),
            train_data={n: torch.randn(64) for n in prob.variables},
            test_data={n: torch.randn(16) for n in prob.variables},
            queries=[], ground_truth=GroundTruth(),
            family="continuous_lg", problem_id="4", seed=0,
        )
        cache = tmp_path / "fit_flow.pt"
        rows_fit = cell_worker._run_cell(
            _worker_ctx(prob, _spec("flow", "lw", epochs=2),
                        fit_role="fit", cache_path=cache))
        assert cache.exists()
        assert all(r["status"] != "error" for r in rows_fit)
        rows_reload = cell_worker._run_cell(
            _worker_ctx(prob, _spec("flow", "lw", epochs=2),
                        fit_role="reload", cache_path=cache))
        assert all(r["status"] != "error" for r in rows_reload)


# =====================================================================
# 4. VALUE faithfulness: reload-path inference == fit-path inference
# =====================================================================
# The deterministic half of the grouped==ungrouped guarantee. Grouping replaces
# a baseline's own fit() with load_base_and_attach() of a sibling's saved base;
# this asserts the two paths produce BITWISE-identical inference. Kept in ONE
# process with cat (closed-form MLE fit, exact VE inference) so there is no RNG
# to confound the comparison — the runner-level accuracy metric, by contrast,
# samples its oracle per-subprocess and is nondeterministic run-to-run (so it
# cannot carry a bitwise check; see TestGroupedEqualsUngrouped).

class TestReloadValueFaithfulness:
    def test_reload_ve_equals_fresh_ve_bitwise(self, tmp_path):
        from nbn.bench.adapters.nbn_adapter import NBNAdapter
        from nbn.bench.domains.base import Query

        problem = _problem(0, n_nodes=5)
        q = Query(targets=("X2",), evidence={"X0": 0})

        # Fresh fit (what an UNGROUPED cat-ve cell does).
        fresh = NBNAdapter(mechanism="cat", engine="ve", device="cpu")
        fresh.fit(problem)
        r_fresh = fresh.query(q).probs

        # Save the base, then reload into a new ve adapter (what a GROUPED
        # cat-ve reloader does: reuse a sibling's saved base).
        cache = tmp_path / "base.pt"
        torch.save(fresh.model, cache)
        reloaded = NBNAdapter(mechanism="cat", engine="ve", device="cpu")
        reloaded.load_base_and_attach(str(cache), problem)
        r_reload = reloaded.query(q).probs

        assert torch.equal(r_fresh, r_reload)      # bitwise-identical posterior

    def test_reloaded_adapter_state_matches_fresh(self, tmp_path):
        # load_base_and_attach must leave the adapter indistinguishable from a
        # freshly-fit one: model + engine + problem all populated.
        from nbn.bench.adapters.nbn_adapter import NBNAdapter
        problem = _problem(0)
        fresh = NBNAdapter(mechanism="cat", engine="ve", device="cpu")
        fresh.fit(problem)
        cache = tmp_path / "base.pt"
        torch.save(fresh.model, cache)

        reloaded = NBNAdapter(mechanism="cat", engine="ve", device="cpu")
        reloaded.load_base_and_attach(str(cache), problem)
        assert reloaded.model is not None
        assert reloaded._engine_obj is not None
        assert reloaded.problem is problem
        assert type(reloaded._engine_obj) is type(fresh._engine_obj)


# =====================================================================
# 5. HEADLINE: grouped == ungrouped, through the real runner + subprocesses
# =====================================================================

def _synth_cfg(tmp_path, *, baselines, metrics):
    from nbn.bench.problems.synthetic import (
        SyntheticConfig, SyntheticProblemSource,
    )
    measurement = AccuracyAndTiming() if metrics == "all" else TimingOnly()
    return RunnerConfig(
        benchmark="synthetic", config_name="fos",
        problem_source=SyntheticProblemSource(),
        source_config=SyntheticConfig(
            families=["discrete"], n_nodes_list=[5], seeds=[0],
            n_train=128, n_test=32, n_reference=128, cardinality=2,
        ),
        selector=UniformRandomSelector(),
        measurement=measurement,
        baselines=baselines,
        n_queries_per_cell=3,
        per_cell_timeout_s=120.0,
        jsonl_path=tmp_path / "out.jsonl",
    )


def _key_tuple(r):
    return (r.baseline, r.family, r.problem_id, r.seed, r.batch_size,
            r.query_role, r.metric, r.status)


@pytest.mark.slow
class TestGroupedEqualsUngrouped:
    """The runner-level correctness anchor: a grouped run yields the SAME cells
    and the SAME statuses as a forced-ungrouped run on the same config — proving
    grouping changes only RUNTIME, not WHICH cells run or whether they succeed.

    The row VALUES are NOT bitwise-compared here: the accuracy metric samples its
    oracle with per-subprocess RNG and so is nondeterministic run-to-run EVEN
    with grouping held constant (verified empirically — two grouped runs differ
    identically). The deterministic value guarantee — reload-path inference ==
    fit-path inference — is proven without that confound by
    TestReloadValueFaithfulness (in-process, cat closed-form, exact VE) and by
    the Stage-1 bitwise save/load verification. Ordering [cat-lw, cat-ve] makes
    the deterministic VE baseline the RELOADER, so the grouped run genuinely
    exercises reload-of-a-saved-base for a baseline whose cell would otherwise
    fit."""

    def _run(self, tmp_path, monkeypatch, *, grouped):
        baselines = [_spec("cat", "lw"), _spec("cat", "ve")]
        cfg = _synth_cfg(tmp_path, baselines=baselines, metrics="all")
        if not grouped:
            import nbn.bench.core.runner as rmod
            # Force ungrouped: every baseline fits its own model.
            monkeypatch.setattr(
                rmod, "_assign_fit_roles",
                lambda cfg, *a, **k: (
                    [("standalone", None)] * len(cfg.baselines), {}),
            )
        return list(Runner().run(cfg))

    def test_grouped_matches_ungrouped_structure(self, tmp_path, monkeypatch):
        grouped = self._run(tmp_path / "g", monkeypatch, grouped=True)
        with pytest.MonkeyPatch.context() as mp:
            ungrouped = self._run(tmp_path / "u", mp, grouped=False)

        # Identical cell structure + statuses across both runs: same
        # (baseline, family, problem_id, seed, batch_size, query_role, metric,
        # status) multiset. Grouping changes runtime, not the cells.
        assert sorted(map(_key_tuple, grouped)) == \
               sorted(map(_key_tuple, ungrouped))
        # Both the fitter and the reloader produced ok query rows (no cell was
        # silently dropped or failed by grouping).
        ok = {(r.baseline, r.status) for r in grouped
              if r.metric == "query_time_s"}
        assert ("nbn-cat-lw", "ok") in ok and ("nbn-cat-ve", "ok") in ok

    def test_fitcache_cleaned_after_run(self, tmp_path, monkeypatch):
        self._run(tmp_path, monkeypatch, grouped=True)
        leaked = list(Path(tmp_path).rglob("*.pt"))
        assert leaked == [], f"leaked cache files: {leaked}"
        assert not (Path(tmp_path) / "_fitcache").exists()


@pytest.mark.slow
class TestNonNbnUnaffected:
    """A config mixing nbn + pomegranate: the non-nbn baseline is never grouped
    or cached, and the run completes cleanly through real subprocesses."""

    def test_mixed_run_completes_and_no_leak(self, tmp_path):
        baselines = [
            _spec("cat", "ve"), _spec("cat", "lw"),
            BaselineSpec(library="pomegranate", mechanism="discrete",
                         param_method="mle", inference_method="ve"),
        ]
        cfg = _synth_cfg(tmp_path, baselines=baselines, metrics="timing")
        rows = list(Runner().run(cfg))
        names = {r.baseline for r in rows}
        assert "nbn-cat-ve" in names and "nbn-cat-lw" in names
        assert any("pomegranate" in n for n in names)
        assert list(Path(tmp_path).rglob("*.pt")) == []
