"""Tests for SyntheticProblemSource and SyntheticConfig.

Three test classes:
  - TestSyntheticConfig: default values, validation
  - TestSyntheticProblemSourceStructural: iter_problems() yields BenchmarkProblems
    matching config, correct field population (no model training — fast).
  - TestSyntheticProblemSourceBehavioral: end-to-end generation with real
    make_synthetic_bn (marked @pytest.mark.slow).

Reference: docs/v0.13-benchmark-redesign.md §2.1, §4.1
"""
from __future__ import annotations

import pytest
import torch

from nbn.bench.problems.synthetic import SyntheticConfig, SyntheticProblemSource
from nbn.bench.domains.base import BenchmarkProblem, GroundTruth
from nbn.bench.core.interfaces import ProblemSource


class TestSyntheticConfig:
    """SyntheticConfig defaults and field types."""

    def test_default_families(self):
        cfg = SyntheticConfig()
        assert set(cfg.families) == {
            "discrete", "continuous_lg", "continuous_nongauss", "hybrid",
        }

    def test_default_n_nodes_list(self):
        cfg = SyntheticConfig()
        assert len(cfg.n_nodes_list) > 0
        assert all(isinstance(n, int) for n in cfg.n_nodes_list)

    def test_default_seeds(self):
        cfg = SyntheticConfig()
        assert len(cfg.seeds) > 0

    def test_custom_config(self):
        cfg = SyntheticConfig(
            families=["discrete"],
            n_nodes_list=[5, 10],
            seeds=[0, 1],
            n_train=500,
            n_test=100,
        )
        assert cfg.families == ["discrete"]
        assert cfg.n_nodes_list == [5, 10]
        assert cfg.n_train == 500

    def test_config_iteration_count(self):
        cfg = SyntheticConfig(
            families=["discrete", "continuous_lg"],
            n_nodes_list=[5, 10, 20],
            seeds=[0, 1],
        )
        expected = len(cfg.families) * len(cfg.n_nodes_list) * len(cfg.seeds)
        assert expected == 12


class TestSyntheticProblemSourceProtocol:
    """SyntheticProblemSource satisfies the ProblemSource protocol."""

    def test_satisfies_problem_source_protocol(self):
        src = SyntheticProblemSource()
        assert isinstance(src, ProblemSource)

    def test_has_iter_problems(self):
        src = SyntheticProblemSource()
        assert callable(getattr(src, "iter_problems", None))


@pytest.mark.slow
class TestSyntheticProblemSourceBehavioral:
    """End-to-end problem generation — requires make_synthetic_bn.

    Uses very small n_nodes and n_train to keep wall-clock < 10 s.
    """

    @pytest.fixture(scope="class")
    def discrete_problem(self):
        """One small discrete problem."""
        src = SyntheticProblemSource()
        cfg = SyntheticConfig(
            families=["discrete"],
            n_nodes_list=[5],
            seeds=[0],
            n_train=200,
            n_test=100,
            n_reference=500,
        )
        return next(iter(src.iter_problems(cfg)))

    @pytest.fixture(scope="class")
    def continuous_problem(self):
        """One small continuous_lg problem."""
        src = SyntheticProblemSource()
        cfg = SyntheticConfig(
            families=["continuous_lg"],
            n_nodes_list=[5],
            seeds=[0],
            n_train=200,
            n_test=100,
            n_reference=500,
        )
        return next(iter(src.iter_problems(cfg)))

    def test_yields_benchmark_problem(self, discrete_problem):
        assert isinstance(discrete_problem, BenchmarkProblem)

    def test_family_populated(self, discrete_problem):
        assert discrete_problem.family == "discrete"

    def test_problem_id_is_n_nodes_string(self, discrete_problem):
        assert discrete_problem.problem_id == "5"

    def test_true_model_populated(self, discrete_problem):
        assert discrete_problem.true_model is not None

    def test_train_data_shape(self, discrete_problem):
        for name, tensor in discrete_problem.train_data.items():
            assert tensor.shape[0] == 200, f"node {name}: expected 200 rows"

    def test_test_data_shape(self, discrete_problem):
        for name, tensor in discrete_problem.test_data.items():
            assert tensor.shape[0] == 100, f"node {name}: expected 100 rows"

    def test_variables_consistent_with_dag(self, discrete_problem):
        # All nodes referenced in dag edges must be in variables.
        import networkx as nx
        dag_nx = nx.DiGraph()
        dag_nx.add_nodes_from(discrete_problem.variables)
        dag_nx.add_edges_from(discrete_problem.dag)
        for node in dag_nx.nodes():
            assert node in discrete_problem.variables

    def test_ground_truth_samples_present_discrete(self, discrete_problem):
        """Discrete families get a synthesised ground-truth pool."""
        assert discrete_problem.ground_truth is not None
        assert discrete_problem.ground_truth.samples is not None
        samples = discrete_problem.ground_truth.samples
        assert samples.ndim == 2
        assert samples.shape[1] == len(discrete_problem.variables)

    def test_ground_truth_samples_present_continuous(self, continuous_problem):
        """Continuous families have ground_truth_samples from make_synthetic_bn."""
        assert continuous_problem.ground_truth is not None
        assert continuous_problem.ground_truth.samples is not None
        samples = continuous_problem.ground_truth.samples
        assert samples.ndim == 2
        assert samples.shape[1] == len(continuous_problem.variables)

    def test_iteration_count_matches_config(self):
        src = SyntheticProblemSource()
        cfg = SyntheticConfig(
            families=["discrete", "continuous_lg"],
            n_nodes_list=[4, 6],
            seeds=[0, 1],
            n_train=100,
            n_test=50,
            n_reference=200,
        )
        problems = list(src.iter_problems(cfg))
        assert len(problems) == 2 * 2 * 2  # 2 families × 2 n_nodes × 2 seeds

    def test_deterministic_given_same_seed(self):
        src = SyntheticProblemSource()
        cfg = SyntheticConfig(
            families=["discrete"],
            n_nodes_list=[4],
            seeds=[42],
            n_train=100,
            n_test=50,
            n_reference=200,
        )
        p1 = next(iter(src.iter_problems(cfg)))
        p2 = next(iter(src.iter_problems(cfg)))
        # Same seed → same train data.
        for node in p1.train_data:
            assert torch.allclose(p1.train_data[node].float(), p2.train_data[node].float())

    def test_different_seeds_give_different_data(self):
        src = SyntheticProblemSource()
        cfg_0 = SyntheticConfig(families=["discrete"], n_nodes_list=[4], seeds=[0],
                                 n_train=100, n_test=50, n_reference=200)
        cfg_1 = SyntheticConfig(families=["discrete"], n_nodes_list=[4], seeds=[1],
                                 n_train=100, n_test=50, n_reference=200)
        p0 = next(iter(src.iter_problems(cfg_0)))
        p1 = next(iter(src.iter_problems(cfg_1)))
        first_node = next(iter(p0.train_data))
        assert not torch.allclose(p0.train_data[first_node].float(),
                                   p1.train_data[first_node].float())

    def test_name_encodes_family_and_size(self, discrete_problem):
        assert "discrete" in discrete_problem.name
        assert "n5" in discrete_problem.name

    def test_seed_populated_from_config(self, discrete_problem):
        """problem.seed must match the seed passed in SyntheticConfig.seeds."""
        assert discrete_problem.seed == 0

    def test_seed_propagated_correctly_for_different_seeds(self):
        src = SyntheticProblemSource()
        cfg = SyntheticConfig(
            families=["discrete"],
            n_nodes_list=[4],
            seeds=[0, 1, 2],
            n_train=100,
            n_test=50,
            n_reference=200,
        )
        problems = list(src.iter_problems(cfg))
        seeds_seen = [p.seed for p in problems]
        assert seeds_seen == [0, 1, 2]
