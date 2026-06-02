"""Tests for BnlearnProblemSource — Stage 2 (discrete kind only).

Network-touching tests are marked ``slow`` (they fetch from bnlearn.com on
first run, then use the ~/.cache/nbn/bnlearn/ cache). Offline tests (unknown
network, not-implemented kind) run on the fast gate.

Reference: docs/phase4-design-draft.md §2.1, §4, §7, §8.
"""
from __future__ import annotations

import pytest
import torch

from benchmarking.problems.bnlearn import (
    _NETWORKS,
    BnlearnConfig,
    BnlearnProblemSource,
    _decode_config_index,
)


# --- Offline tests (no network) ---------------------------------------------

class TestBnlearnValidation:
    """Config validation that doesn't touch the network."""

    def test_unknown_network_raises(self):
        source = BnlearnProblemSource()
        cfg = BnlearnConfig(networks=["totally_made_up_network"], seeds=[0])
        with pytest.raises(ValueError, match="Unknown bnlearn network"):
            list(source.iter_problems(cfg))

    def test_registry_kinds_recorded(self):
        """Gaussian/CLG kinds are registered (loading is covered in Stage 3)."""
        assert _NETWORKS["ecoli70"]["kind"] == "gaussian"
        assert _NETWORKS["sangiovese"]["kind"] == "clg"

    def test_registry_kind_counts(self):
        """24 discrete + 4 gaussian + 3 clg = 31 networks.

        Note: design doc §3 *summary* says "22 discrete … 29 networks", but
        its own §3.1 *table* lists 24 discrete networks (the count used here
        and in the relay's registry). The summary line is a doc typo.
        """
        kinds = {}
        for meta in _NETWORKS.values():
            kinds[meta["kind"]] = kinds.get(meta["kind"], 0) + 1
        assert kinds == {"discrete": 24, "gaussian": 4, "clg": 3}
        assert len(_NETWORKS) == 31


# --- Network-touching tests (slow) ------------------------------------------

class TestBnlearnDiscreteLoading:
    """Discrete network loading via on-demand .bif download."""

    @pytest.mark.slow
    def test_asia_loads(self):
        source = BnlearnProblemSource()
        cfg = BnlearnConfig(networks=["asia"], seeds=[0], n_train=100, n_test=20,
                            n_reference=200)
        problems = list(source.iter_problems(cfg))
        assert len(problems) == 1

        p = problems[0]
        assert p.name == "asia"
        assert p.family == "discrete"
        assert p.problem_id == "asia"
        assert p.seed == 0
        assert p.true_model is not None

        # ASIA: 8 nodes, all discrete.
        assert len(p.variables) == 8
        assert all(v[0] == "discrete" for v in p.variables.values())

        # Data shapes: one tensor per node.
        assert len(p.train_data) == 8
        assert all(t.shape == (100,) for t in p.train_data.values())
        assert all(t.dtype == torch.long for t in p.train_data.values())
        assert len(p.test_data) == 8
        assert all(t.shape == (20,) for t in p.test_data.values())

        # Ground-truth reference pool populated for the discrete oracle.
        assert p.ground_truth is not None
        assert p.ground_truth.samples is not None
        assert p.ground_truth.samples.shape == (200, 8)

    @pytest.mark.slow
    def test_state_indices_in_range(self):
        """Sampled values are valid state indices in [0, n_states)."""
        source = BnlearnProblemSource()
        cfg = BnlearnConfig(networks=["asia"], seeds=[0], n_train=200, n_test=20,
                            n_reference=100)
        p = next(source.iter_problems(cfg))
        for node, (_, n_states) in p.variables.items():
            col = p.train_data[node]
            assert col.min() >= 0
            assert col.max() < n_states


class TestBnlearnDeterminism:
    """Forward sampling is deterministic given seed."""

    @pytest.mark.slow
    def test_same_seed_same_data(self):
        cfg = BnlearnConfig(networks=["asia"], seeds=[42], n_train=100, n_test=20,
                            n_reference=100)
        p1 = next(BnlearnProblemSource().iter_problems(cfg))
        p2 = next(BnlearnProblemSource().iter_problems(cfg))
        for node in p1.train_data:
            assert torch.equal(p1.train_data[node], p2.train_data[node])
            assert torch.equal(p1.test_data[node], p2.test_data[node])

    @pytest.mark.slow
    def test_different_seeds_different_data(self):
        cfg = BnlearnConfig(networks=["asia"], seeds=[0, 1], n_train=200, n_test=50,
                            n_reference=100)
        problems = list(BnlearnProblemSource().iter_problems(cfg))
        assert len(problems) == 2
        assert problems[0].seed != problems[1].seed
        all_equal = all(
            torch.equal(problems[0].train_data[n], problems[1].train_data[n])
            for n in problems[0].train_data
        )
        assert not all_equal, "different seeds produced identical data"


class TestBnlearnOracleCompatibility:
    """Stage 2 verification: the existing oracle scores a bnlearn problem."""

    @pytest.mark.slow
    def test_oracle_accepts_bnlearn_problem(self):
        """filter_ground_truth returns reference-pool target samples for a
        bnlearn discrete problem — no oracle changes needed."""
        from benchmarking.core.oracle import filter_ground_truth

        cfg = BnlearnConfig(networks=["asia"], seeds=[0], n_train=100, n_test=20,
                            n_reference=500)
        problem = next(BnlearnProblemSource().iter_problems(cfg))
        target = list(problem.variables.keys())[0]

        # Empty (marginal) evidence: no filtering → whole target column.
        gt = filter_ground_truth(problem, {}, target)
        assert gt is not None, "oracle returned None — ground-truth pool missing?"
        assert gt.ndim == 1
        assert gt.shape[0] == 500  # all reference rows survive empty evidence
        n_states = problem.variables[target][1]
        assert gt.min() >= 0 and gt.max() < n_states


# --- Stage 3: Gaussian + CLG -----------------------------------------------

class TestBnlearnGaussian:
    """Pure Gaussian network loading + sampling."""

    def test_config_decoder_roundtrip(self):
        """_decode_config_index is the inverse of the R-side flat encoding."""
        import json
        from pathlib import Path

        path = Path("benchmarking/data/bnlearn/mehra.json")
        if not path.exists():
            pytest.skip("mehra.json not bundled")
        data = json.loads(path.read_text())
        for cpd in data["cpds"]:
            if cpd["type"] == "clg_continuous" and len(cpd.get("discrete_parents", [])) >= 2:
                dps, dlevels = cpd["discrete_parents"], cpd["dlevels"]
                K = len(cpd["intercepts"])
                for i in [0, 1, 17, 100, K - 1]:
                    cfg = _decode_config_index(i, dps, dlevels)
                    re_i, mult = 0, 1
                    for dp in dps:
                        re_i += dlevels[dp].index(cfg[dp]) * mult
                        mult *= len(dlevels[dp])
                    assert re_i == i, f"round-trip failed: {i} -> {cfg} -> {re_i}"
                return
        pytest.skip("no multi-discrete-parent CLG node found")

    @pytest.mark.slow
    def test_ecoli70_loads_and_samples(self):
        cfg = BnlearnConfig(networks=["ecoli70"], seeds=[0],
                            n_train=200, n_test=50, n_reference=100)
        p = next(BnlearnProblemSource().iter_problems(cfg))
        assert p.family == "continuous_gauss"
        assert p.problem_id == "ecoli70"
        assert len(p.variables) == 46
        assert len(p.train_data) == 46
        assert all(t.shape == (200,) for t in p.train_data.values())
        assert all(t.dtype == torch.float32 for t in p.train_data.values())
        for t in p.train_data.values():
            assert torch.isfinite(t).all()
        assert any(t.std() > 0 for t in p.train_data.values()), \
            "all nodes zero-variance — sampler broken"
        assert p.true_model is not None
        assert p.ground_truth is not None and p.ground_truth.samples.shape == (100, 46)

    @pytest.mark.slow
    def test_ecoli70_determinism(self):
        cfg = BnlearnConfig(networks=["ecoli70"], seeds=[42],
                            n_train=100, n_test=50, n_reference=50)
        p1 = next(BnlearnProblemSource().iter_problems(cfg))
        p2 = next(BnlearnProblemSource().iter_problems(cfg))
        for node in p1.train_data:
            assert torch.equal(p1.train_data[node], p2.train_data[node])


class TestBnlearnCLG:
    """CLG network loading + sampling (mixed discrete/continuous)."""

    @pytest.mark.slow
    def test_sangiovese_loads_and_samples(self):
        cfg = BnlearnConfig(networks=["sangiovese"], seeds=[0],
                            n_train=200, n_test=50, n_reference=100)
        p = next(BnlearnProblemSource().iter_problems(cfg))
        assert p.family == "clg"
        assert len(p.variables) == 15

        discrete = [n for n, v in p.variables.items() if v[0] == "discrete"]
        continuous = [n for n, v in p.variables.items() if v[0] != "discrete"]
        assert discrete and continuous, "sangiovese should have both kinds"

        for node in discrete:
            t = p.train_data[node]
            assert t.dtype == torch.long
            assert (t >= 0).all() and (t < p.variables[node][1]).all()
        for node in continuous:
            t = p.train_data[node]
            assert t.dtype == torch.float32
            assert torch.isfinite(t).all()

    @pytest.mark.slow
    def test_mehra_loads_finite(self):
        """Largest network (multi-discrete config decoding) samples finite."""
        cfg = BnlearnConfig(networks=["mehra"], seeds=[0],
                            n_train=50, n_test=20, n_reference=20)
        p = next(BnlearnProblemSource().iter_problems(cfg))
        assert p.family == "clg"
        for node, t in p.train_data.items():
            assert torch.isfinite(t.float()).all(), f"non-finite in {node}"


class TestBnlearnContinuousModel:
    """_BnlearnContinuousModel acts correctly as problem.true_model."""

    @pytest.mark.slow
    def test_ecoli70_model_clamps_continuous_evidence(self):
        cfg = BnlearnConfig(networks=["ecoli70"], seeds=[0],
                            n_train=10, n_test=10, n_reference=10)
        p = next(BnlearnProblemSource().iter_problems(cfg))
        node = list(p.variables.keys())[0]
        ev = {node: torch.tensor([1.5])}
        samples = p.true_model.sample(n=100, evidence=ev)
        assert torch.allclose(samples[node], torch.full((100,), 1.5))
        # Non-clamped nodes still finite.
        for other, t in samples.items():
            assert torch.isfinite(t.float()).all()

    @pytest.mark.slow
    def test_sangiovese_model_clamps_discrete_evidence(self):
        cfg = BnlearnConfig(networks=["sangiovese"], seeds=[0],
                            n_train=10, n_test=10, n_reference=10)
        p = next(BnlearnProblemSource().iter_problems(cfg))
        discrete = [n for n, v in p.variables.items() if v[0] == "discrete"]
        if not discrete:
            pytest.skip("sangiovese has no discrete nodes (unexpected)")
        node = discrete[0]
        samples = p.true_model.sample(n=100, evidence={node: torch.tensor([0])})
        assert torch.all(samples[node] == 0)
