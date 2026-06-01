# Phase 3 — HeaviestQueryByRole + Scalability Benchmark

**Status:** Design draft (pre-implementation)
**Depends on:** Phase 2 (TopologicalAllocator, merged in PR #124)
**Related:** issue #109 (v0.13 roadmap), issue #74 (selectors)

## 1. Background

Phase 2 introduced topology-aware query selection: a stochastic allocator
that samples queries proportionally across hub/cut/terminal/random buckets.
The output is a *distribution* of queries that lets us measure how
baselines perform across query types in aggregate.

Phase 3 introduces a complementary selector for the **scalability
benchmark** of paper §5. Where Phase 2's selector samples a *population*
of queries, Phase 3's selector picks the **single hardest query of each
role**, and exercises it under two evidence modes. The deliverable is a
focused stress-test: how do baselines scale as DAG size grows, when the
queries are deliberately the structurally hardest ones?

## 2. The selector

### 2.1 Specification

`HeaviestQueryByRole` is a deterministic selector (given a seed, the
returned query list is fully determined). Per DAG, it produces up to 12
queries organized along three orthogonal axes:

- **Role** (3 values): hub, cut, terminal
- **Direction** (2 values): prediction, diagnosis
- **Evidence mode** (2 values): full, empty

The target for each role is fixed:
- hub target = `NodeRoles.hubs[0]` (highest |MB|)
- cut target = `NodeRoles.cuts[0]` (highest degree centrality among APs)
- terminal target = `NodeRoles.terminals[0]` (highest depth − |descendants|)

The evidence strategy is fixed per role (matching the structural
"hardness" of that role):
- hub → longest_path (ancestors/descendants by depth distance)
- cut → mb_neighbors (parents/children + co-parents)
- terminal → longest_path

If `NodeRoles.cuts` is empty (no articulation points), the 4 cut queries
are silently dropped. Final query count per DAG: 12 if APs exist, 8 if
not.

### 2.2 Query table

| # | Target     | Strategy     | Direction | Mode  |
|---|------------|--------------|-----------|-------|
| 1 | hubs[0]    | longest_path | prediction | full  |
| 2 | hubs[0]    | longest_path | prediction | empty |
| 3 | hubs[0]    | longest_path | diagnosis  | full  |
| 4 | hubs[0]    | longest_path | diagnosis  | empty |
| 5 | cuts[0]    | mb_neighbors | prediction | full  |
| 6 | cuts[0]    | mb_neighbors | prediction | empty |
| 7 | cuts[0]    | mb_neighbors | diagnosis  | full  |
| 8 | cuts[0]    | mb_neighbors | diagnosis  | empty |
| 9 | terminals[0] | longest_path | prediction | full  |
| 10 | terminals[0] | longest_path | prediction | empty |
| 11 | terminals[0] | longest_path | diagnosis  | full  |
| 12 | terminals[0] | longest_path | diagnosis  | empty |

### 2.3 Evidence semantics

The two evidence modes differ only in whether evidence variables carry
concrete values:

**Full mode** (V1):
- `Query.evidence = {Y: 0.2, Z: 0.3, ...}` — concrete values sampled from `problem.test_data`
- Adapter computes the conditional posterior `P(X | Y=0.2, Z=0.3)`
- Single 1D distribution over target X

**Empty mode** (V2):
- `Query.evidence = {Y: None, Z: None, ...}` — keys present, values are `None`
- Adapter marginalizes over Y and Z, computing `P(X)`:
  $$P(X) = \sum_{y, z} P(X | Y=y, Z=z) \cdot P(Y=y, Z=z)$$
- Output is also a 1D distribution over X (same shape as V1)

**Both modes use the same evidence node set** for a given (role, direction)
pair. The two queries differ only in whether the values are sampled or
`None`. This lets the paper compare V1 vs V2 *on the same structural
query*, isolating the effect of evidence specification.

V1 values come from `problem.test_data` via the same `_ValueAllocator`
infrastructure introduced in Phase 2.

## 3. Data model changes

### 3.1 `Query.evidence` type

Currently typed as `Mapping[str, int | float | torch.Tensor]`. Extended
to `Mapping[str, int | float | torch.Tensor | None]` to permit empty-mode
evidence.

`Query.kind` stays at `"marginal"` for both V1 and V2 — `kind` enumerates
inference *operations* (marginal/conditional/do/sample/map), an orthogonal
dimension from evidence shape.

### 3.2 `CellResult.evidence_mode`

New field on `CellResult`:
```python
evidence_mode: str = "full"   # "full" | "empty"
```
Backward-compat default `"full"` keeps all pre-Phase-3 selectors working
unchanged. UniformRandomSelector and TopologicalAllocator both produce
full-mode evidence by construction; their rows get `evidence_mode="full"`
implicitly.

### 3.3 Parquet schema

The parquet gains one categorical column: `evidence_mode`. Combined with
the existing `query_role`, `query_kind`, and `evidence_strategy` columns
from Phase 2, the parquet now supports four-way decomposition of cells
by query structure.

## 4. Adapter changes

Each baseline adapter that reads `Query.evidence` must handle `None`
values. The fix is uniform across adapters: **split evidence into
observed values and marginalized variables**.

```python
# Before (Phase 2): all values are concrete
observed = {k: prep(v) for k, v in q.evidence.items()}

# After (Phase 3): split by None
observed = {k: prep(v) for k, v in q.evidence.items() if v is not None}
marginalized = [k for k, v in q.evidence.items() if v is None]
# `marginalized` is informational; engines auto-marginalize unobserved variables
```

Inference engines all natively marginalize over variables not in the
evidence dict. The fix is mechanical: skip None entries instead of trying
to coerce them with `.item()` or `.float()`.

**Per-adapter scope estimate** (from investigation report):

| Adapter | Scope | Notes |
|---|---|---|
| pgmpy (mle-ve, bayes-ve) | NEEDS_TRIVIAL | Filter Nones from evidence dict |
| pgmpy-lg-predict | NEEDS_TRIVIAL | Extend empty-evidence branch to handle Nones |
| pomegranate | NEEDS_TRIVIAL | Filter Nones |
| NBN (8 variants) | NEEDS_TRIVIAL | Filter Nones in `_prep_evidence` |
| pyro | NEEDS_TRIVIAL | Filter Nones (pyro marginalizes unconditioned variables natively) |

The investigation's original NEEDS_MAJOR verdict for pyro assumed V2
required joint marginals over (target ∪ evidence). The corrected V2
semantics (marginalize evidence, return single-target posterior) is
pyro's *default* behavior — no special multi-site logic needed.

## 5. Oracle changes

`oracle.py` and `filter_ground_truth` currently assume numeric evidence
and would crash on `None`. Same fix: skip None entries when computing
ground truth. The ground-truth computation for V2 is the marginal P(X)
with Y, Z marginalized out — mathematically equivalent to feeding the
adapter `evidence={}` and asking for the same target, but applied to the
true model rather than the fitted one.

Metrics (`tv_per_node`, `jsd_per_node`, `w1_per_node`) are unchanged.
Both V1 and V2 produce 1D distributions over a single target; the existing
metric implementations work without modification.

## 6. YAML surface

```yaml
selector:
  type: heaviest_by_role
```

No allocation knobs (the selector is deterministic). No `n_queries_per_cell`
— the selector produces up to 12 queries per problem regardless of
budget. The YAML validation should reject `n_queries_per_cell` when
selector is `heaviest_by_role`, or at minimum log a warning.

The runner's `seeds` list controls how many different seeds get evaluated
per problem. Each seed re-samples the V1 values (the V2 queries are
deterministic per seed).

## 7. Scalability benchmark

### 7.1 Configurations

Two configs:

**scalability_smoke.yaml** (CI / smoke verification):
- 2 families: discrete, continuous_lg
- 2 seeds: [0, 1]
- 3 n values: [10, 50, 100]
- Per-cell budget: 30s
- Expected runtime: ~5-10 min

**scalability_paper.yaml** (paper §5 figure source):
- 4 families: discrete, continuous_lg, continuous_non_lg, hybrid
- 5 seeds: [0, 1, 2, 3, 4]
- 11 n values: [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
- 4 baselines: pgmpy, pomegranate, pyro, NBN
- Per-cell budget: 60s
- Measurement: AccuracyAndTiming
- Expected runtime: ~14-30 hours wall-clock worst case

### 7.2 Expected status distribution

The scalability benchmark explicitly expects baselines to hit limits at
large n. The existing status taxonomy (`ok / timeout / oom / error /
not_supported`) records these as data rather than failures:

- Small n (5-100): all baselines `ok` for both V1 and V2
- Medium n (200-1000): some baselines start hitting `timeout`
- Large n (2000-10000): expected mix of `ok / timeout / oom` per baseline
- pyro V2 specifically: works as well as pyro V1 (corrected verdict from investigation)

The paper §5 figure plots query-time vs n by (baseline, family,
evidence_mode), with a separate panel showing status counts. Timeouts
and OOMs are part of the deliverable, not bugs.

## 8. Implementation plan (4 stages)

Following the Phase 2 pattern: one PR, reviewer checkpoint between
stages.

### Stage 1: HeaviestQueryByRole selector
- New file: `benchmarking/selectors/heaviest.py` (~150 LOC)
- Reuses `compute_node_roles` from `topological.py`
- Reuses `_ValueAllocator` for V1 value sampling
- Unit tests on ASIA fixture verifying:
  - 12 queries emitted (or 8 if cuts empty)
  - Pairwise V1/V2 share the same evidence nodes
  - Determinism with seed
  - Target selection matches NodeRoles top-of-bucket

### Stage 2: Data model + oracle None-handling
- `Query.evidence` type extended to allow None values
- `CellResult.evidence_mode` field added with `"full"` default
- `oracle.py` + `filter_ground_truth` handle None-valued evidence
- Tests verify oracle returns correct marginal for None-valued evidence

### Stage 3: Adapter None-handling + end-to-end smoke
- One commit per adapter family (pgmpy, pomegranate, pyro, NBN)
- Each adapter filters Nones from observed evidence
- Per-adapter tests verify V2 query path produces output
- End-to-end smoke: run scalability_smoke.yaml, verify parquet shows
  both `evidence_mode="full"` and `evidence_mode="empty"` rows with
  populated metrics for all baselines

### Stage 4: YAML dispatch + scalability configs
- `yaml_config.py` dispatch for `selector: {type: heaviest_by_role}`
- `scalability_smoke.yaml` and `scalability_paper.yaml` committed
- Final smoke verification: parquet schema, expected query counts,
  baseline coverage
- PR marked ready for review

## 9. Acceptance criteria

- [ ] `HeaviestQueryByRole` selector produces up to 12 queries per DAG (4 if no APs)
- [ ] Each (role, direction) pair has matching V1 and V2 queries with identical evidence nodes
- [ ] `CellResult.evidence_mode` populated in parquet for all baselines
- [ ] V1 queries reproduce existing Phase 2 metric values (within noise) when given the same evidence
- [ ] V2 queries produce valid 1D distributions on all four baselines
- [ ] Oracle computes correct ground truth for both modes
- [ ] Scalability smoke runs end-to-end with the new selector
- [ ] Backward-compat: all existing 312 v0.13 tests remain green
- [ ] Backward-compat: existing parquets (TopologicalAllocator, UniformRandomSelector) get `evidence_mode="full"` by default

## 10. Out of scope

- Phase 4 (BnlearnProblemSource): scalability benchmark stays on synthetic DAGs only
- Phase 5 (per-role figure aggregation): paper figures are a separate task
- Multi-variable joint marginals (P(X, Y) where Y is a "joint output"): not needed under the corrected V2 semantics
- HeaviestQueryByRole as a configurable allocator (different ranking criteria, etc.): keep the selector deterministic and minimal
