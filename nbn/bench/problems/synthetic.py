"""SyntheticProblemSource — thin wrapper over ``make_synthetic_bn``.

Adapts the existing ``SyntheticBN`` dataclass into the v0.13
``BenchmarkProblem`` format, populating all fields including the
three v0.13 additions (``true_model``, ``family``, ``problem_id``).

Key responsibilities beyond the thin-wrap contract:
  1. Synthesise the ground-truth reference pool for discrete families
     (``SyntheticBN.ground_truth_samples`` is ``None`` for discrete).
     This eliminates the ``object.__setattr__`` hack on a frozen dataclass
     that the v0.12 runner used for lazy synthesis.
  2. Store the pool in ``GroundTruth.samples`` (``problem.ground_truth``).
  3. Store the true model on ``problem.true_model`` for oracle sampling.
  4. Populate ``problem.family`` and ``problem.problem_id`` for the v3 schema.

Reference: docs/v0.13-benchmark-redesign.md §2.1, §4.1, §5.2
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterator

import torch

from nbn.bench.domains.base import BenchmarkProblem, GroundTruth
from nbn.bench.synthetic import make_synthetic_bn


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class SyntheticConfig:
    """Configuration for ``SyntheticProblemSource.iter_problems()``.

    Mirrors the parameters of ``make_synthetic_bn`` plus the iteration
    dimensions (families, n_nodes_list, seeds).

    Attributes
    ----------
    families:
        List of family strings to iterate over.  Valid values:
        ``"discrete"``, ``"continuous_lg"``, ``"continuous_nongauss"``,
        ``"hybrid"``.
    n_nodes_list:
        List of node-count values.  One problem is generated per element.
    seeds:
        List of RNG seeds.  One problem is generated per seed per
        (family, n_nodes) combination.
    edge_density:
        Expected fraction of possible directed edges (topological-order
        pairs) that are realised.
    max_in_degree:
        Maximum number of parents per node; enforced during edge sampling.
    cardinality:
        Number of categories per discrete node.
    fraction_continuous:
        For ``family="hybrid"``: probability that each node is continuous.
    n_train, n_test:
        Training / held-out sample counts.
    n_reference:
        Number of joint samples drawn for the ground-truth reference pool
        (continuous and hybrid families) or the discrete oracle pool.
        See ``make_synthetic_bn`` for the automatic halving rule when
        ``n_nodes >= 1000``.
    device:
        PyTorch device string passed to ``make_synthetic_bn``.
    """

    families: list[str] = field(
        default_factory=lambda: [
            "discrete", "continuous_lg", "continuous_nongauss", "hybrid",
        ],
    )
    n_nodes_list: list[int] = field(default_factory=lambda: [10, 50, 100])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    edge_density: float = 0.20
    max_in_degree: int = 4
    cardinality: int = 4
    fraction_continuous: float = 0.5
    n_train: int = 10_000
    # n_train sensitivity sweep (#109 PR 6, learning curves): when set, the
    # source iterates these training-set sizes as an extra problem axis (one
    # fit per value), mutually exclusive with the scalar n_train. None = no
    # sweep (every existing config). yaml_config validates exactly one is set.
    n_train_sweep: list[int] | None = None
    n_test: int = 2_000
    n_reference: int = 50_000
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Problem source
# ---------------------------------------------------------------------------

class SyntheticProblemSource:
    """Thin wrapper over ``make_synthetic_bn`` implementing ``ProblemSource``.

    Iterates over the Cartesian product ``families × n_nodes_list × seeds``
    in that order, yielding one ``BenchmarkProblem`` per combination.

    v0.13 vs v0.12 differences
    --------------------------
    * **Discrete oracle pool synthesised at generation time.**  The v0.12
      runner synthesised it lazily inside ``_inference_cell`` using
      ``object.__setattr__`` on the frozen ``SyntheticBN``.  Here it is
      created once per problem and stored in ``problem.ground_truth.samples``.
    * **``true_model`` on problem.**  ``BenchmarkProblem.true_model`` is
      populated so ``AccuracyAndTiming`` can call
      ``problem.true_model.sample(n=N, evidence=ev)`` for continuous oracle
      sampling without needing ``SyntheticBN``.
    * **Schema fields populated.** ``problem.family`` and
      ``problem.problem_id`` are set for the v3 parquet schema.

    Usage::

        src = SyntheticProblemSource()
        cfg = SyntheticConfig(
            families=["discrete"],
            n_nodes_list=[10, 50],
            seeds=[0, 1],
        )
        for problem in src.iter_problems(cfg):
            print(problem.name, problem.family, problem.problem_id)
    """

    def iter_problems(self, cfg: SyntheticConfig) -> Iterator[BenchmarkProblem]:
        """Yield ``BenchmarkProblem`` instances matching the config.

        Iteration order: outer → families, middle → n_nodes_list,
        inner → seeds.  This matches the v0.12 runner's loop nesting.

        n_train sensitivity sweep (#109 PR 6): when ``cfg.n_train_sweep`` is set,
        an INNERMOST axis iterates the training-set sizes, yielding one problem
        per ``(family, n_nodes, seed, n_train)`` — each its own fit downstream
        (the runner fits once per problem; there is no fit-cache reuse across
        n_train, unlike the cell-internal ``batch_sizes`` sweep). The INVARIANT
        that makes a learning curve well-posed: ``(seed, n_nodes, family)``
        alone determine the ``true_model`` — ``make_synthetic_bn`` builds the
        DAG + CPTs from the seeded RNG *before* drawing samples, so n_train only
        varies how many training rows are sampled, never the ground truth.
        (Verified empirically: same seed, n_train=50 vs 5000 → bit-identical true
        CPTs and DAG.) So across a sweep the recovery target is fixed and only
        the data grows — "how does method M's recovery improve with more data on
        THIS model?".

        Parameters
        ----------
        cfg:
            A ``SyntheticConfig`` instance.
        """
        n_train_values = cfg.n_train_sweep or [cfg.n_train]
        for family in cfg.families:
            for n_nodes in cfg.n_nodes_list:
                for seed in cfg.seeds:
                    for n_train in n_train_values:
                        yield self._make_problem(
                            cfg, family, n_nodes, seed, n_train=n_train)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_problem(
        cfg: SyntheticConfig,
        family: str,
        n_nodes: int,
        seed: int,
        n_train: int | None = None,
    ) -> BenchmarkProblem:
        """Generate and adapt one synthetic BN into a ``BenchmarkProblem``.

        ``n_train`` overrides ``cfg.n_train`` for the learning-curve sweep
        (#109 PR 6); the seed/n_nodes/family still fix the true_model, so only
        the sampled training-set size changes.
        """
        sbn = make_synthetic_bn(
            family=family,
            n_nodes=n_nodes,
            edge_density=cfg.edge_density,
            max_in_degree=cfg.max_in_degree,
            cardinality=cfg.cardinality,
            fraction_continuous=cfg.fraction_continuous,
            n_train=cfg.n_train if n_train is None else int(n_train),
            n_test=cfg.n_test,
            n_reference=cfg.n_reference,
            seed=seed,
            device=cfg.device,
        )

        gt_samples = _ensure_gt_samples(sbn)

        return BenchmarkProblem(
            name=sbn.name,
            dag=list(sbn.dag.edges()),
            variables=sbn.variable_specs,
            train_data=sbn.train_data,
            test_data=sbn.test_data,
            queries=[],
            ground_truth=GroundTruth(samples=gt_samples),
            # v0.13 additions:
            true_model=sbn.true_model,
            family=family,
            problem_id=str(n_nodes),
            seed=seed,
        )


# ---------------------------------------------------------------------------
# Ground-truth pool synthesis
# ---------------------------------------------------------------------------

def _ensure_gt_samples(sbn) -> torch.Tensor | None:
    """Return the joint reference pool in topological-sort column order.

    For non-discrete families, ``sbn.ground_truth_samples`` is already
    populated by ``make_synthetic_bn``.

    For discrete families, ``sbn.ground_truth_samples`` is ``None`` (by
    design — discrete families use exact-match rejection on sampled
    marginals, not a pre-computed pool, in the v0.12 runner).  We
    synthesise a pool here so the problem carries its own oracle data.

    The pool size is ``min(5_000, n_reference // 2)`` capped at the
    v0.12 runner's convention (line 473: ``n_ref = min(5000, max(1000,
    cfg.n_reference // 2))``).  We use 5_000 as a reasonable default
    matching the v0.12 ``_compute_inference_metrics`` fallback.

    Columns are written in ``sbn.column_order`` (topological sort), which
    matches what ``nbn.bench.core.oracle.filter_ground_truth`` will
    reconstruct via ``nx.topological_sort``.
    """
    if sbn.ground_truth_samples is not None:
        return sbn.ground_truth_samples

    # Discrete family: synthesise now.
    n_ref = 5_000
    try:
        with torch.no_grad():
            ref = sbn.true_model.sample(n=n_ref)
        pool = torch.cat(
            [ref[nm].reshape(n_ref, -1).float().cpu() for nm in sbn.column_order],
            dim=-1,
        )
        return pool
    except Exception as exc:  # pragma: no cover — best-effort
        warnings.warn(
            f"SyntheticProblemSource: failed to synthesise discrete ground-truth "
            f"pool for {sbn.name!r}: {exc}",
            UserWarning,
            stacklevel=4,
        )
        return None
