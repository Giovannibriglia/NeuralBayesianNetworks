"""Sanity tests for the bnlearn benchmark (pre-relaunch validation).

Three layers across a six-network representative subset:

- **Layer 1 (structural)**: networks load, node counts match the registry,
  DAGs are acyclic, discrete CPDs are finite.
- **Layer 2 (data generation)**: sampled data has valid cardinalities and
  per-node marginals consistent with a large reference sample.
- **Layer 3 (oracle correctness)**: the benchmark's *sample-based* oracle
  agrees with an independent reference. The oracle is Monte-Carlo (rejection
  sampling on a forward-sampled reference pool for discrete targets; ancestral
  sampling with clamped evidence for continuous/CLG), so the comparison is
  within a combined noise floor, not exact equality.

  Layer 3 uses a **tiered reference** for discrete networks:
    - Tier A: pgmpy ``VariableElimination`` on the true model (exact),
      SIGALRM-bounded at 60s; tolerance ``atol=0.03`` (oracle MC noise
      dominates; the exact reference is precise).
    - Tier B: if Tier A times out / OOMs / is unsupported, fall back to pgmpy
      ``likelihood_weighted_sample`` (N=10000); tolerance ``atol=0.05``
      (combined oracle-MC + LW-MC noise). This mirrors what the paper will do
      for networks too large for exact inference.
    - Skip only if Tier B also fails (should not happen on this subset).
  For continuous/CLG networks pgmpy exact does not apply; Layer 3 instead does
  a summary-statistics sanity check on the clamped-sample oracle.

The oracle is sample-based at ``n_reference`` samples; this suite bumps the
fixture to ``n_reference=20000`` (vs the 5000 default) to tighten the oracle's
own MC noise so the agreement bands are meaningful. The honest framing — the
"ground truth" is a Monte-Carlo reference with a ~1/sqrt(n_reference) noise
floor, verified against exact where feasible and against LW otherwise — is
documented for the paper methodology section (issue #138).

All tests are ``@pytest.mark.slow``:
    pytest -m slow tests/integration/test_bnlearn_sanity.py -v

Reference: docs/v0.13-paper-figures.md §7, issue #47.
"""
from __future__ import annotations

import logging
import signal

import networkx as nx
import numpy as np
import pytest
import torch

logger = logging.getLogger(__name__)

# Six-network representative subset (scoping conversation).
SUBSET = ["asia", "barley", "hailfinder", "link", "sangiovese", "ecoli70"]

LAYER3_TIMEOUT_S = 60
LW_SIZE = 10000
SANITY_N_REFERENCE = 20000          # tighter oracle MC noise for this suite
ATOL_EXACT = 0.03                   # Tier A: oracle MC vs precise exact
ATOL_LW = 0.05                      # Tier B: oracle MC + LW MC combined
DISCRETE_FAMILIES = {"discrete"}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module", params=SUBSET)
def network_name(request):
    return request.param


@pytest.fixture(scope="module")
def loaded_problem(network_name):
    from benchmarking.problems.bnlearn import BnlearnConfig, BnlearnProblemSource

    cfg = BnlearnConfig(
        networks=[network_name], seeds=[0],
        n_train=200, n_test=50, n_reference=SANITY_N_REFERENCE,
    )
    return next(BnlearnProblemSource().iter_problems(cfg))


def _dag_graph(problem) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(problem.variables)
    g.add_edges_from(problem.dag)
    return g


# --------------------------------------------------------------------------- #
# Layer 1: structural
# --------------------------------------------------------------------------- #

@pytest.mark.slow
class TestStructural:
    def test_network_loads(self, loaded_problem, network_name):
        assert loaded_problem is not None
        assert loaded_problem.true_model is not None
        assert loaded_problem.variables, f"{network_name} has no variables"

    def test_dag_is_acyclic(self, loaded_problem, network_name):
        assert nx.is_directed_acyclic_graph(_dag_graph(loaded_problem)), \
            f"{network_name} DAG has a cycle"

    def test_node_count_matches_registry(self, loaded_problem, network_name):
        from benchmarking.problems.bnlearn import _NETWORKS
        expected = _NETWORKS[network_name].get("n_nodes")
        if expected is not None:
            assert len(loaded_problem.variables) == expected, (
                f"{network_name}: registry says {expected} nodes, "
                f"loaded {len(loaded_problem.variables)}"
            )

    def test_discrete_cpds_finite(self, loaded_problem, network_name):
        """pgmpy discrete CPDs have no NaN/inf. (Continuous models have no
        get_cpds; their parameter finiteness is covered by Layer 2 stats.)"""
        tm = loaded_problem.true_model
        if not hasattr(tm, "get_cpds"):
            pytest.skip(f"{network_name}: continuous model has no discrete CPDs")
        for node in tm.nodes():
            cpd = tm.get_cpds(node)
            assert np.all(np.isfinite(cpd.values)), \
                f"{network_name}: CPD for {node} non-finite"


# --------------------------------------------------------------------------- #
# Layer 2: data generation
# --------------------------------------------------------------------------- #

@pytest.mark.slow
class TestDataGeneration:
    def test_training_data_valid_cardinalities(self, loaded_problem, network_name):
        data = loaded_problem.train_data
        for node, (kind, k) in loaded_problem.variables.items():
            if kind != "discrete":
                continue
            col = data[node]
            assert int(col.min()) >= 0, f"{network_name}.{node}: value < 0"
            assert int(col.max()) < k, \
                f"{network_name}.{node}: value {int(col.max())} >= cardinality {k}"

    def test_node_marginals_match_reference(self, loaded_problem, network_name):
        """Train-data per-node statistics are consistent with the large
        reference pool (both sampled from the true model)."""
        from benchmarking.core.oracle import _column_order

        col_order = _column_order(loaded_problem)
        col_idx = {n: i for i, n in enumerate(col_order)}
        ref = loaded_problem.ground_truth.samples
        assert ref is not None and ref.numel() > 0, f"{network_name}: empty ref pool"

        # Check a handful of nodes (cheap; full set is redundant).
        for node in sorted(loaded_problem.variables)[:5]:
            kind, k = loaded_problem.variables[node]
            train_col = loaded_problem.train_data[node].float()
            ref_col = ref[:, col_idx[node]].float()
            if kind == "discrete":
                tr = np.bincount(train_col.long().numpy(), minlength=k).astype(float)
                tr /= tr.sum()
                rf = np.bincount(ref_col.long().numpy(), minlength=k).astype(float)
                rf /= rf.sum()
                # n_train=200 sampling noise ~0.035; 0.12 is a safe loose bound.
                assert np.max(np.abs(tr - rf)) < 0.12, (
                    f"{network_name}.{node}: train marginal {tr} vs ref {rf}"
                )
            else:
                # continuous: compare mean within a scale-relative tolerance
                mu_t, mu_r = float(train_col.mean()), float(ref_col.mean())
                sd_r = float(ref_col.std()) + 1e-9
                assert abs(mu_t - mu_r) < 0.25 * sd_r + 0.1, (
                    f"{network_name}.{node}: train mean {mu_t:.3f} vs ref {mu_r:.3f} "
                    f"(ref std {sd_r:.3f})"
                )


# --------------------------------------------------------------------------- #
# Layer 3 helpers
# --------------------------------------------------------------------------- #

class _Timeout(Exception):
    pass


def _state_name(model, node, i: int):
    return model.get_cpds(node).state_names[node][i]


def _aligned(model, target, result_state_names, values) -> np.ndarray:
    """Reorder a posterior given over ``result_state_names`` into the model's
    canonical state_names order (== the integer encoding used by the oracle)."""
    sn = list(model.get_cpds(target).state_names[target])
    out = np.zeros(len(sn), dtype=float)
    for j, name in enumerate(result_state_names):
        out[sn.index(name)] = float(values[j])
    return out


def _pgmpy_exact(model, target, evidence_int, timeout_s):
    """Tier A: exact posterior over target via VE, SIGALRM-bounded.
    Returns a length-K array in model state_names order, or None on
    timeout/OOM/unsupported."""
    from pgmpy.inference import VariableElimination

    ev = {n: _state_name(model, n, v) for n, v in evidence_int.items()}

    def _handler(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(timeout_s))
    try:
        res = VariableElimination(model).query(
            variables=[target], evidence=ev or None, show_progress=False,
        )
        return _aligned(model, target, res.state_names[target], res.values)
    except (_Timeout, MemoryError):
        return None
    except Exception as exc:  # unsupported / pgmpy internal — fall back
        logger.warning("pgmpy exact failed (%s): %s", type(exc).__name__, exc)
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _pgmpy_lw(model, target, evidence_int, size):
    """Tier B: LW-sampled posterior over target. Returns length-K array in
    model state_names order, or None on failure."""
    from pgmpy.factors.discrete import State
    from pgmpy.sampling import BayesianModelSampling

    ev = [State(n, _state_name(model, n, v)) for n, v in evidence_int.items()]
    try:
        df = BayesianModelSampling(model).likelihood_weighted_sample(
            evidence=ev, size=size, show_progress=False,
        )
    except Exception as exc:
        logger.warning("pgmpy LW failed (%s): %s", type(exc).__name__, exc)
        return None
    sn = list(model.get_cpds(target).state_names[target])
    out = np.zeros(len(sn), dtype=float)
    weights = df["_weight"].to_numpy(dtype=float)
    for name, w in zip(df[target].to_numpy(), weights):
        out[sn.index(str(name))] += w
    s = out.sum()
    return out / s if s > 0 else None


def _oracle_discrete(problem, target, evidence_int):
    """The benchmark's discrete oracle posterior (rejection on the reference
    pool), as a length-K array, or None if too few rows survive."""
    from benchmarking.core.oracle import filter_ground_truth

    ev_row = {n: torch.tensor(v) for n, v in evidence_int.items()}
    samp = filter_ground_truth(problem, ev_row, target)
    if samp is None:
        return None
    k = problem.variables[target][1]
    counts = np.bincount(samp.long().numpy(), minlength=k).astype(float)
    s = counts.sum()
    return counts / s if s > 0 else None


def _tiered_reference(model, target, evidence_int):
    """Return (reference_posterior, atol, tier_label). Tier A (exact) first,
    Tier B (LW) on infeasibility."""
    ref = _pgmpy_exact(model, target, evidence_int, LAYER3_TIMEOUT_S)
    if ref is not None:
        return ref, ATOL_EXACT, "exact"
    ref = _pgmpy_lw(model, target, evidence_int, LW_SIZE)
    if ref is not None:
        return ref, ATOL_LW, "lw"
    return None, None, "none"


def _pick_conditional(problem):
    """A (target, {parent: 0}) pair: first node (sorted) that has a parent."""
    g = _dag_graph(problem)
    for node in sorted(problem.variables):
        parents = sorted(g.predecessors(node))
        if parents:
            return node, {parents[0]: 0}
    return None, None


# --------------------------------------------------------------------------- #
# Layer 3: oracle correctness — discrete (tiered exact/LW reference)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
class TestOracleDiscrete:
    def test_marginal_matches_reference(self, loaded_problem, network_name):
        if loaded_problem.family not in DISCRETE_FAMILIES:
            pytest.skip(f"{network_name}: not discrete (Layer 3 discrete path)")
        model = loaded_problem.true_model
        target = sorted(loaded_problem.variables)[0]

        oracle = _oracle_discrete(loaded_problem, target, {})
        assert oracle is not None, \
            f"{network_name}: discrete marginal oracle returned None (unexpected)"

        ref, atol, tier = _tiered_reference(model, target, {})
        if ref is None:
            pytest.skip(f"{network_name}: both exact and LW references failed")
        logger.info("%s marginal P(%s) verified via tier=%s", network_name, target, tier)
        np.testing.assert_allclose(
            oracle, ref, atol=atol,
            err_msg=(f"{network_name} P({target}) oracle vs {tier} ref: "
                     f"oracle={oracle}, ref={ref}"),
        )

    def test_conditional_matches_reference(self, loaded_problem, network_name):
        if loaded_problem.family not in DISCRETE_FAMILIES:
            pytest.skip(f"{network_name}: not discrete (Layer 3 discrete path)")
        model = loaded_problem.true_model
        target, evidence = _pick_conditional(loaded_problem)
        if target is None:
            pytest.skip(f"{network_name}: no node with a parent for a conditional")

        oracle = _oracle_discrete(loaded_problem, target, evidence)
        if oracle is None:
            pytest.skip(
                f"{network_name}: rejection left too few rows for "
                f"P({target}|{evidence}); oracle returns None by design"
            )
        ref, atol, tier = _tiered_reference(model, target, evidence)
        if ref is None:
            pytest.skip(f"{network_name}: both references failed for conditional")
        logger.info("%s conditional P(%s|%s) verified via tier=%s",
                    network_name, target, evidence, tier)
        np.testing.assert_allclose(
            oracle, ref, atol=atol,
            err_msg=(f"{network_name} P({target}|{evidence}) oracle vs {tier}: "
                     f"oracle={oracle}, ref={ref}"),
        )


# --------------------------------------------------------------------------- #
# Layer 3: oracle correctness — continuous/CLG (summary-stats sanity)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
class TestOracleContinuous:
    def test_clamped_marginal_stats(self, loaded_problem, network_name):
        """pgmpy exact does not apply to continuous/CLG. Instead verify the
        clamped-sample oracle's marginal (no evidence == ancestral) has
        mean/std consistent with a direct large ancestral sample. Scale-free
        tolerances (10% mean, 20% std) avoid flakiness on large-scale nodes."""
        if loaded_problem.family in DISCRETE_FAMILIES:
            pytest.skip(f"{network_name}: discrete (handled by TestOracleDiscrete)")
        from benchmarking.core.oracle import (
            _column_order, forward_with_clamp_posterior_samples,
        )

        cont = [n for n, (k, _) in loaded_problem.variables.items() if k == "continuous"]
        if not cont:
            pytest.skip(f"{network_name}: no continuous targets")
        target = sorted(cont)[0]

        oracle = forward_with_clamp_posterior_samples(
            loaded_problem, [target], {}, n_samples=2000,
        )
        if oracle is None:
            pytest.skip(f"{network_name}: clamped-sample oracle unavailable")
        o = oracle.reshape(-1).numpy()

        col_idx = {n: i for i, n in enumerate(_column_order(loaded_problem))}
        ref = loaded_problem.ground_truth.samples[:, col_idx[target]].float().numpy()

        mu_o, mu_r = float(o.mean()), float(ref.mean())
        sd_o, sd_r = float(o.std()), float(ref.std())
        scale = abs(mu_r) + sd_r + 1e-9
        assert abs(mu_o - mu_r) < 0.1 * scale, (
            f"{network_name}.{target}: clamped mean {mu_o:.3f} vs ancestral "
            f"{mu_r:.3f} (scale {scale:.3f})"
        )
        assert abs(sd_o - sd_r) < 0.2 * (sd_r + 1e-9) + 0.05 * abs(mu_r), (
            f"{network_name}.{target}: clamped std {sd_o:.3f} vs ancestral {sd_r:.3f}"
        )
