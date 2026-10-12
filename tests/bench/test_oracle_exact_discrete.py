"""Discrete oracle tiers (D1): exact VE on the true model, LW, rejection.

The discrete oracle used to be a 5,000-row rejection filter whose TV noise
(0.05-0.07 per query) hid real method differences. These tests pin:

* the exact tier against brute-force enumeration (synthetic NBN true model
  and a pgmpy true model, as used by bnlearn ``.bif`` networks);
* that a method returning the exact posterior now scores TV ~ 0 (it scored
  the oracle's Monte-Carlo error before);
* the LW tier (hybrid networks / intractable VE): deterministic per seed and
  close to exact;
* the synthetic discrete reference pool honours ``cfg.n_reference`` and is
  reproducible.
"""
from __future__ import annotations

import itertools

import pytest
import torch

from nbn.bench.core.exact_discrete import exact_posterior, tabular_source
from nbn.bench.core.oracle import (
    ORACLE_EXACT,
    ORACLE_LW,
    discrete_oracle,
)
from nbn.bench.domains.base import BenchmarkProblem, GroundTruth, Query
from nbn.bench.domains.posterior import Posterior
from nbn.bench.measurements.accuracy_timing import AccuracyAndTiming
from nbn.bench.problems import SyntheticConfig, SyntheticProblemSource


def _problem(family="discrete", n_nodes=7, seed=3, card=3, n_reference=200, **kw):
    cfg = SyntheticConfig(
        families=[family], n_nodes_list=[n_nodes], seeds=[seed], n_train=200,
        n_test=40, n_reference=n_reference, edge_density=kw.pop("density", 0.5),
        max_in_degree=3, cardinality=card, **kw,
    )
    return next(iter(SyntheticProblemSource().iter_problems(cfg)))


def _brute_force(src, nodes, card, target, ev):
    out = torch.zeros(card, dtype=torch.float64)
    for assign in itertools.product(range(card), repeat=len(nodes)):
        a = dict(zip(nodes, assign))
        if any(a[k] != v for k, v in ev.items()):
            continue
        lp = sum(
            float(src.log_table(nd)[tuple(a[p] for p in src.parents(nd)) + (a[nd],)])
            for nd in nodes
        )
        out[a[target]] += torch.tensor(lp, dtype=torch.float64).exp()
    return out / out.sum()


def test_exact_posterior_matches_brute_force_enumeration():
    problem = _problem()
    src = tabular_source(problem)
    nodes = list(problem.variables)
    g = torch.Generator().manual_seed(0)
    for trial in range(25):
        perm = torch.randperm(len(nodes), generator=g).tolist()
        target = nodes[perm[0]]
        ev = {nodes[i]: int(torch.randint(0, 3, (1,), generator=g))
              for i in perm[1:1 + trial % 4]}
        got = exact_posterior(src, target, ev)
        want = _brute_force(src, nodes, 3, target, ev)
        assert torch.allclose(got, want, atol=1e-10), (target, ev)


def test_exact_posterior_target_in_evidence_is_delta():
    problem = _problem()
    src = tabular_source(problem)
    node = next(iter(problem.variables))
    out = exact_posterior(src, node, {node: 2})
    assert out.tolist() == [0.0, 0.0, 1.0]


class _ExactAdapter:
    """Answers every query with the exact posterior (a perfect method)."""

    name = "exact-oracle-twin"

    def __init__(self, problem):
        self._src = tabular_source(problem)

    def query(self, q):
        ev = {k: int(v.item()) for k, v in q.evidence.items() if v is not None}
        return Posterior(probs=exact_posterior(self._src, q.targets[0], ev).float())


def test_perfect_method_scores_zero_tv_against_exact_oracle():
    """Before D1 the reference was a 5,000-row rejection sample, so even the
    exact posterior scored TV ~ 0.05; the oracle noise floor is now gone."""
    problem = _problem(n_nodes=8, n_reference=50_000)
    nodes = list(problem.variables)
    queries = [
        Query(targets=(nodes[i % 8],),
              evidence={nodes[(i + 3) % 8]: problem.test_data[nodes[(i + 3) % 8]][i]
                        .reshape(()).float(),
                        nodes[(i + 5) % 8]: problem.test_data[nodes[(i + 5) % 8]][i]
                        .reshape(()).float()},
              kind="marginal")
        for i in range(12)
    ]
    rows = AccuracyAndTiming().measure(problem, _ExactAdapter(problem), queries)
    tv = [r for r in rows if r.metric == "tv_per_node"]
    assert len(tv) == 12
    assert all(r.status == "ok" for r in tv)
    assert all(r.oracle == ORACLE_EXACT for r in tv)
    assert max(r.value for r in tv) < 1e-5


def _pgmpy_problem():
    pytest.importorskip("pgmpy")
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.models import DiscreteBayesianNetwork

    model = DiscreteBayesianNetwork([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    model.add_cpds(
        TabularCPD("A", 2, [[0.3], [0.7]], state_names={"A": ["a0", "a1"]}),
        TabularCPD("B", 3, [[0.2, 0.6], [0.5, 0.3], [0.3, 0.1]], evidence=["A"],
                   evidence_card=[2], state_names={"B": ["x", "y", "z"], "A": ["a0", "a1"]}),
        TabularCPD("C", 2, [[0.9, 0.25], [0.1, 0.75]], evidence=["A"], evidence_card=[2],
                   state_names={"C": ["c0", "c1"], "A": ["a0", "a1"]}),
        TabularCPD("D", 2,
                   [[0.1, 0.4, 0.8, 0.5, 0.35, 0.95], [0.9, 0.6, 0.2, 0.5, 0.65, 0.05]],
                   evidence=["B", "C"], evidence_card=[3, 2],
                   state_names={"D": ["d0", "d1"], "B": ["x", "y", "z"], "C": ["c0", "c1"]}),
    )
    variables = {"A": ("discrete", 2), "B": ("discrete", 3), "C": ("discrete", 2),
                 "D": ("discrete", 2)}
    problem = BenchmarkProblem(
        name="toy", dag=list(model.edges()), variables=variables,
        train_data={}, test_data={}, queries=[],
        ground_truth=GroundTruth(samples=torch.zeros(1000, 4)), true_model=model,
        family="discrete", problem_id="toy", seed=0,
    )
    return problem, model


def test_exact_oracle_on_pgmpy_true_model_matches_pgmpy_ve():
    problem, model = _pgmpy_problem()
    from pgmpy.inference import VariableElimination

    ve = VariableElimination(model)
    probs, tier = discrete_oracle(problem, "A", {"D": torch.tensor(1.0),
                                                 "B": torch.tensor(2.0)})
    ref = ve.query(["A"], evidence={"D": "d1", "B": "z"}, show_progress=False)
    assert tier == ORACLE_EXACT
    assert torch.allclose(probs, torch.tensor(ref.values, dtype=torch.float64), atol=1e-9)


def test_lw_tier_on_pgmpy_model_is_seeded_and_close_to_exact():
    problem, _ = _pgmpy_problem()
    ev = {"D": torch.tensor(0.0)}
    exact, _ = discrete_oracle(problem, "C", ev)
    # max_entries=1 makes exact VE intractable -> LW tier.
    a, tier = discrete_oracle(problem, "C", ev, max_entries=1, n_particles=20_000, seed=5)
    b, _ = discrete_oracle(problem, "C", ev, max_entries=1, n_particles=20_000, seed=5)
    assert tier == ORACLE_LW
    assert torch.equal(a, b)
    assert float(0.5 * (a - exact).abs().sum()) < 0.02


def test_hybrid_discrete_target_uses_lw_not_rejection_window():
    problem = _problem(family="hybrid", n_nodes=8, seed=1, density=0.6,
                       fraction_continuous=0.5)
    cont = [n for n, (k, _) in problem.variables.items() if k == "continuous"]
    disc = [n for n, (k, _) in problem.variables.items() if k == "discrete"]
    assert cont and disc
    ev = {cont[0]: problem.test_data[cont[0]][0].reshape(())}
    target = next(d for d in disc if d not in ev)
    a, tier = discrete_oracle(problem, target, ev, n_particles=2000, seed=11)
    b, _ = discrete_oracle(problem, target, ev, n_particles=2000, seed=11)
    assert tier == ORACLE_LW
    assert a is not None and torch.equal(a, b)
    assert abs(float(a.sum()) - 1.0) < 1e-9


def test_synthetic_discrete_pool_uses_n_reference_and_is_reproducible():
    p1 = _problem(n_reference=1234)
    p2 = _problem(n_reference=1234)
    assert p1.ground_truth.samples.shape[0] == 1234  # was hard-coded 5_000
    assert torch.equal(p1.ground_truth.samples, p2.ground_truth.samples)


def test_impossible_evidence_in_a_factor_detached_from_the_target():
    """R13: a -inf scalar factor (evidence impossible in a part of the network
    that does not touch the target) was skipped, returning a confident
    posterior for P(e) = 0. It is now 'zero-probability evidence'."""
    from nbn.bench.core.exact_discrete import TabularSource, ZeroProbabilityEvidence

    tables = {"A": torch.tensor([0.5, 0.5]),
              "B": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),   # B is always 0
              "T": torch.tensor([0.3, 0.7])}
    parents = {"A": (), "B": ("A",), "T": ()}
    src = TabularSource(lambda n: parents[n], lambda n: 2, lambda n: tables[n])
    with pytest.raises(ZeroProbabilityEvidence):
        exact_posterior(src, "T", {"B": 1})
    assert torch.allclose(exact_posterior(src, "T", {"B": 0}),
                          torch.tensor([0.3, 0.7], dtype=torch.float64))


def test_out_of_range_evidence_state_is_a_clear_error():
    from nbn.bench.core.exact_discrete import InvalidEvidence

    problem = _problem()
    src = tabular_source(problem)
    nodes = list(problem.variables)
    with pytest.raises(InvalidEvidence, match="outside the node's states"):
        exact_posterior(src, nodes[0], {nodes[1]: 7})
