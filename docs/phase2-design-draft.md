# Phase 2 — TopologicalAllocator: Implementation Spec

> **Ready for implementation.** §3 contains the locked-in design.
> §5 lists four remaining implementation-level decisions that must be
> answered before the Phase 0 hard-gate relay. §4 open questions from
> the prior draft are fully resolved.

---

## §1. Background

The current `UniformRandomSelector` picks one `(target, evidence_nodes)` pair
per call and repeats it across all `n_queries` rows, varying only the evidence
*values* from `problem.test_data`. With `n_nodes=1000` and
`n_queries_per_cell=1024`, almost all queries land on peripheral nodes with
small Markov blankets — easy for every method. A paper that reports only
per-query averages over this distribution will hide NBN's advantage on the
hard cases (hub nodes, articulation-point nodes) and will be weak against
reviewers who ask "how do you perform on worst-case queries?"

`TopologicalAllocator` replaces this with a pipeline that assigns every query
a **target bucket**, a **query kind**, and an **evidence strategy** — three
orthogonal structural axes that together determine where in the DAG the query
lands and in which inferential direction evidence flows.

Four target buckets determine which topological region of the DAG the target
node is drawn from:

| Bucket | Node definition | Why it stresses inference |
|---|---|---|
| **hub** | High Markov-blanket cardinality: `parents ∪ children ∪ co-parents` | Factor sizes in VE explode; LW memory scales with \|MB\| |
| **cut** | Articulation points of the moralised undirected graph | Treewidth bottlenecks; exact inference's worst-case subgraph |
| **terminal** | High `depth(v) − \|descendants(v)\|` score (deep, few children) | Long propagation paths for LW; sparse evidence reach |
| **random** | Uniform random | PAC-style coverage; prevents structural buckets from biasing the mean |

Default target budget split: **30 % hub / 25 % cut / 25 % terminal / 20 % random**.
All fractions are YAML-configurable. Validated against ASIA (n=8), SACHS (n=11),
and synthetic n=1000 DAGs (NodeRoles computation <55 ms, well inside the <1 s
budget).

---

## §2. Current state (post-PR #122)

### 2.1 QuerySelector protocol

Defined in `benchmarking/core/interfaces.py:41`:

```python
class QuerySelector(Protocol):
    def select(
        self,
        problem: BenchmarkProblem,
        n_queries: int,
        seed: int,
    ) -> list[Query]:
        """Return a list of queries, ordered. Deterministic given seed."""
        ...
```

The protocol does **not** currently include per-query metadata fields — only a
single `selector.query_role: str` attribute is read by the runner (see §2.3).
This is resolved by adding metadata fields to the `Query` dataclass (see §3.6).

### 2.2 Current implementation

`benchmarking/selectors/uniform.py` — `UniformRandomSelector`:

- Constructor: `__init__(self, n_evidence: int = 3)`
- Picks **one** `(target, evidence_nodes)` pair using a seeded permutation
  (`seed * 1000 + 7`), then emits `min(n_queries, n_test)` `Query` objects
  sharing that pair, iterating over `test_data` rows for evidence values.
- All returned queries have `kind="marginal"`.
- Class attribute `query_role: str = "random"` — a single string, not
  per-query.

**Key behavioral difference for the TopologicalAllocator:** it produces queries
with *different* `(target, evidence_nodes)` per role group, not a single
repeated pair. Every query has its own target, kind, and evidence assignment.

`benchmarking/selectors/topological.py` — **does not exist yet**. Phase 2
creates it from scratch. The redesign doc expected Phase 1 to create a
placeholder; it was not. This is the authoritative target location.

### 2.3 Runner integration (`benchmarking/core/runner.py`)

Relevant lines:

```python
# Line 236 — reads a single role string from the selector
default_role = getattr(cfg.selector, "query_role", "random")

# Line 267 — selector is called once per cell
queries = cfg.selector.select(problem, cfg.n_queries_per_cell, problem.seed)

# Line 268 — all queries get the same role (the bottleneck this PR resolves)
query_roles = [default_role] * len(queries)
```

`query_roles` is then passed to `_measure_queries()` which emits one
`CellResult` row per query with `query_role=role`. After Phase 2, the runner
reads per-query metadata from the `Query` objects themselves (§3.6).

### 2.4 Inputs available inside `select()`

From `BenchmarkProblem` (`benchmarking/domains/base.py`):

| Field | Type | Notes |
|---|---|---|
| `dag` | `list[tuple[str, str]]` | Edge list — must build nx.DiGraph from this |
| `variables` | `dict[str, tuple[str, int]]` | `('discrete', K)` or `('continuous', D)` |
| `train_data` | `dict[str, torch.Tensor]` | Not needed for topology |
| `test_data` | `dict[str, torch.Tensor]` | Source of evidence values (Step 4) |
| `seed` | `int` | Passed in as the `seed` argument to `select()` |
| `true_model` | `Any \| None` | Not needed for topology |
| `family` | `str` | Not needed for topology |
| `problem_id` | `str` | Cache key for NodeRoles; use `(family, problem_id)` to avoid collisions |

No networkx DAG is stored on `BenchmarkProblem`; `dag` is an edge list.
The selector calls `nx.DiGraph(problem.dag)` itself.

### 2.5 Graph utilities available

NetworkX is already a dependency imported in multiple places:

- `benchmarking/core/oracle.py:50` — `nx.topological_sort(dag_nx)`
- `benchmarking/synthetic.py:31` — `import networkx as nx`; full DAG generation
- `benchmarking/adapters/nbn_adapter.py:176`, `pyro_adapter.py:147`,
  `pomegranate_adapter.py:92` — local nx imports for topo sort

Relevant nx functions for NodeRoles (none currently called in selectors/):

| Function | Purpose | Measured cost at n=1000 |
|---|---|---|
| `nx.moral_graph(G)` | Add co-parent edges | ~3 ms |
| `nx.articulation_points(G_undirected)` | Cut nodes | ~21 ms |
| Custom BFS depth + descendant count | Terminal score | ~30 ms |
| MB cardinality from adjacency | Hub ranking | ~3 ms |
| `nx.eccentricity()` | (rejected) all-pairs BFS, O(V(V+E)) | ~3.2 s at n=1000 — too slow |

No `markov_blanket`, `articulation_points`, or topology wrapper exists in the
codebase today.

### 2.6 Where TopologicalAllocator plugs in

Single file: `benchmarking/selectors/topological.py`. Contains `NodeRoles`
(dataclass + computation) and `TopologicalAllocator` (selector class). No
separate `benchmarking/query_selection/` package — that was the issue #74
draft; the redesign doc (newer, authoritative) consolidates into `selectors/`.

Phase 3 = `HeaviestQueryByRole` in `benchmarking/selectors/heaviest.py`, which
imports `NodeRoles` from `topological.py`.

---

## §3. The locked-in design

Phase 2 introduces a four-step query-construction pipeline.

### 3.1 Pipeline overview

```
Step 1         Step 2          Step 3            Step 4
─────────      ─────────       ───────────       ─────────────────
Target         Kind            Evidence-set      Value assignment
selection  →   assignment  →   selection     →   (inference only)
               (K1: per-
               bucket)
```

For each query slot in the budget `Q = n_queries_per_cell`:

1. **Target selection** — draw a target node from one of the four topological
   buckets (hub / cut / terminal / random) per the `target_allocation` fractions.
2. **Kind assignment** — assign `prediction` or `diagnosis` to the query,
   independently per bucket (K1), per the `kind_allocation` fractions.
3. **Evidence-set selection** — pick evidence nodes from the pool defined by
   `(target, kind)` via a sub-bucket strategy per `evidence_allocation`.
4. **Value assignment** (inference only) — sample evidence values from
   `test_data` rows; dedupe by full `(target, evidence_nodes, evidence_values)`
   triple (B1). Skipped for parameter-learning measurement.

---

### 3.2 Step 1: Target selection

**YAML block:**

```yaml
selector:
  type: topological
  target_allocation:
    hub:      0.30    # top-k by |MB(v)|
    cut:      0.25    # articulation points of moralised graph
    terminal: 0.25    # top-k by depth(v) − |descendants(v)|
    random:   0.20    # uniform random
```

**Behavior:**

- **Hub** — sort all nodes by `|MB(v)|` = `|parents ∪ children ∪ co-parents|`
  descending; take the top `ceil(target_allocation.hub × Q)` nodes.
- **Cut** — `nx.articulation_points(nx.moral_graph(G).to_undirected())`;
  sub-rank by articulation-point importance when more cuts exist than quota
  (see §5, Q1 for the sub-ranking choice).
- **Terminal** — sort by `depth(v) − |descendants(v)|` descending; depth via
  BFS from roots; descendants via reverse BFS.
- **Random** — uniform random from all nodes.

**Shortfall rule:** if a bucket has fewer available nodes than its quota, cap
at `min(quota, available)` and redistribute the surplus to the random bucket.
Example: ASIA (n=8) has 1 articulation point; a cut quota of 4 becomes
1 cut + 3 random.

**Repetition:** the same target node may appear in multiple query slots with
different kind and evidence assignments. Dedup is enforced at Step 4 by the
full triple, not here.

---

### 3.3 Step 2: Kind assignment (K1 — independent per bucket)

**YAML block:**

```yaml
  kind_allocation:
    hub:      {prediction: 0.5, diagnosis: 0.5}
    cut:      {prediction: 0.5, diagnosis: 0.5}
    terminal: {prediction: 1.0, diagnosis: 0.0}
    random:   {prediction: 0.5, diagnosis: 0.5}
```

**Semantics:**

- `prediction` — evidence conditions on **ancestor** nodes of the target.
  Exercises the forward-propagation path; canonical use case is "given upstream
  observations, what is the target's marginal?"
- `diagnosis` — evidence conditions on **descendant** nodes of the target.
  Exercises backward / diagnostic inference; canonical use case is "given
  downstream observations, explain the target."

**K1:** each target bucket has its own kind fractions, independently. Terminal
bucket defaults to prediction-only because deep nodes with few descendants
rarely have useful descendants as evidence. Hub and cut default to 50/50 to
stress both inference directions.

**Mechanism:** for each target drawn in Step 1, draw a uniform random value in
`[0, 1)` seeded by `(problem.seed × 31 + query_index)`; assign `prediction` if
the value < `kind_allocation[bucket].prediction`, else `diagnosis`.

---

### 3.4 Step 3: Evidence-set selection

**YAML block:**

```yaml
  evidence_allocation:
    prediction:
      longest_path: 0.50    # ancestors on longest path to target
      mb_neighbors: 0.50    # parents + co-parents of target
    diagnosis:
      mb_neighbors: 0.50    # children + co-children of target
      random:       0.50    # uniform random non-target non-descendant nodes
  n_evidence_prediction: 3
  n_evidence_diagnosis:  3
  n_evidence_random:     3
```

**Evidence pools per kind:**

| Kind | Sub-bucket | Evidence candidate pool | Hardness ordering |
|---|---|---|---|
| `prediction` | `longest_path` | Ancestor nodes of target, sub-ranked by path-distance to target (see §5, Q2) | Descending distance |
| `prediction` | `mb_neighbors` | `parents(target) ∪ co-parents(target)` | Descending MB cardinality |
| `diagnosis` | `mb_neighbors` | `children(target) ∪ co-children(target)` | Descending MB cardinality |
| `diagnosis` | `random` | All non-target, non-descendant nodes | Uniform random — no hardness sort |

**Count knobs:** `n_evidence_prediction` / `n_evidence_diagnosis` / `n_evidence_random`
give the number of evidence nodes to sample for prediction, diagnosis, and
random-bucket queries respectively. These are the three count knobs.

**Sub-bucket assignment:** within each kind, assign the evidence sub-bucket for
each query using `evidence_allocation[kind]` fractions (same seeded mechanism
as Step 2). Then draw exactly `n_evidence_{kind}` nodes from the pool, taking
the top-k by hardness ordering (descending) for structured sub-buckets and
uniform random for the `random` sub-bucket.

**Pool shortfall:** if the pool has fewer candidates than `n_evidence_{kind}`,
take all available candidates (do not pad with random nodes). This produces
smaller evidence sets on star-topology or root-node targets.

---

### 3.5 Step 4: Value assignment (inference only)

For each `(target, evidence_nodes)` pair produced by Step 3:

1. Draw a row index uniformly at random from `range(len(test_data))` using
   `(problem.seed × 997 + query_index)` as seed.
2. Slice the test data row: `{node: test_data[node][row_idx] for node in evidence_nodes}`.
3. **No hardness sorting on values** (Q-A: A3). Values are assigned by uniform
   random row sampling only; the structural difficulty is already encoded in
   Steps 1–3.
4. **Dedup by full triple (B1):** the dedup key is
   `(target, frozenset(evidence_nodes), row_idx)`. Using the row index rather
   than the value tensor avoids hashability issues with continuous tensors (see
   §5, Q4). If the triple is already in the seen-set for this cell, skip and
   draw the next row index (up to a max-retry limit, then discard the slot).
5. For **parameter-learning measurement**: Step 4 is skipped entirely. The
   selector returns `(target, evidence_nodes)` pairs with no value tensors;
   the measurement class handles value sampling internally.

---

### 3.6 Per-query metadata

Three new fields are added to the `Query` dataclass
(`benchmarking/core/interfaces.py`):

```python
@dataclass(frozen=True)
class Query:
    targets:           tuple[str, ...]
    evidence:          Mapping[str, int | float | torch.Tensor]
    kind:              str = "marginal"       # existing — Bayesian query type
    target_bucket:     str = "random"         # NEW — hub | cut | terminal | random
    query_kind:        str = "prediction"     # NEW — prediction | diagnosis
    evidence_strategy: str = "random"         # NEW — longest_path | mb_neighbors | random
```

`query_kind` is named to avoid collision with the existing `kind` field
("marginal" / "conditional" / "do" etc.), which is an orthogonal dimension.
All three new fields default to their "least structured" values so that
`UniformRandomSelector` — which never sets them — remains unchanged.

The runner propagates all three to `CellResult` (replacing the current single
`query_role` field). The parquet gains three categorical columns:
`target_bucket`, `query_kind`, `evidence_strategy`. Plotter and aggregator
can group by any combination.

---

### 3.7 Total query count

```
Q_actual ≤ Q_requested = n_queries_per_cell
```

The returned count is strictly ≤ `Q_requested` due to:

- **Step 1 shortfall** — bucket smaller than quota; surplus goes to random,
  but if random also exhausts all nodes, the total shrinks.
- **Step 3 pool shortfall** — evidence pool has zero candidates (e.g., a root
  node with `prediction` kind has no ancestors). That slot is discarded.
- **Step 4 dedup exhaustion** — all distinct triples for a given
  `(target, evidence_nodes)` pair are consumed before filling the quota; slot
  discarded after max-retry.

For ASIA (n=8) at `Q=1024`, the actual count will be far below 1024 due to all
three shortfall sources. This is expected and correct; `CellResult.n_queries`
records the actual count.

---

### 3.8 Hardness metrics summary

| Pipeline level | Metric | Sort order |
|---|---|---|
| Step 1: hub targets | `\|MB(v)\|` = `\|parents ∪ children ∪ co-parents\|` | Descending |
| Step 1: cut targets | Articulation-point importance (see §5, Q1) | Descending |
| Step 1: terminal targets | `depth(v) − \|descendants(v)\|` | Descending |
| Step 3: `longest_path` evidence | Path-distance to target (see §5, Q2) | Descending |
| Step 3: `mb_neighbors` evidence | MB cardinality of candidate node | Descending |
| Step 3: `random` evidence sub-bucket | Uniform random | None |
| Step 4: value assignment | Uniform random row index | None (Q-A: A3) |

---

## §4. Implementation notes

### 4.1 File layout

All new code goes in one file: `benchmarking/selectors/topological.py`
(~600–800 LOC). Three small wiring changes in existing files.

```
benchmarking/
  selectors/
    __init__.py          (existing)
    uniform.py           (existing, unchanged)
    topological.py       (NEW — all Phase 2 code)
  core/
    interfaces.py        (3 new fields on Query)
    runner.py            (read per-query metadata, 4 lines)
    config.py            (selector factory, ~25 LOC)
tests/
  unit/v013/
    test_selector_topological.py  (NEW, ~200–250 LOC)
benchmarking/configs/
  synthetic_paper.yaml   (add selector: block, ~8 lines)
  synthetic_smoke.yaml   (add selector: block, ~8 lines)
```

### 4.2 Internal class structure

```python
# benchmarking/selectors/topological.py

@dataclass
class NodeRoles:
    hubs:      list[str]          # sorted by |MB(v)| desc
    cuts:      list[str]          # sorted by importance desc
    terminals: list[str]          # sorted by depth-descendants desc
    mb_size:   dict[str, int]
    depth:     dict[str, int]
    descendant_count: dict[str, int]
    ancestors: dict[str, frozenset[str]]
    descendants: dict[str, frozenset[str]]

def compute_node_roles(G: nx.DiGraph) -> NodeRoles: ...   # <1s at n=1000

class _TargetAllocator:   # Step 1
    def allocate(self, roles, budget, allocation, rng) -> list[tuple[str, str]]:
        """Returns list of (target_node, bucket_name)."""

class _KindEvidenceAllocator:   # Steps 2 + 3
    def assign(self, target, bucket, roles, kind_alloc, ev_alloc, n_ev, rng) \
        -> tuple[str, str, list[str]]:
        """Returns (query_kind, evidence_strategy, evidence_nodes)."""

class _ValueAllocator:   # Step 4 (inference only)
    def assign_values(self, target, ev_nodes, test_data, seen, seed, idx) \
        -> dict[str, torch.Tensor] | None:
        """Returns evidence dict or None if dedup exhausted."""

class TopologicalAllocator:   # public QuerySelector
    def __init__(self, target_allocation, kind_allocation,
                 evidence_allocation, n_evidence_prediction,
                 n_evidence_diagnosis, n_evidence_random): ...
    def select(self, problem, n_queries, seed) -> list[Query]: ...

    _cache: dict[tuple[str, str], NodeRoles]   # keyed (family, problem_id)
```

`_KindEvidenceAllocator` handles Steps 2 and 3 jointly because the evidence
pool is determined by the kind assignment — they cannot be cleanly separated
into independent passes without redundant graph traversal.

### 4.3 NodeRoles caching

Cache on the `TopologicalAllocator` instance, keyed by `(problem.family, problem.problem_id)`.
Avoids recomputing the same DAG's topology for every (problem × baseline) cell.
The cache key uses `family` in addition to `problem_id` because synthetic
problems use `str(n_nodes)` as `problem_id`, and two different families at
n=100 have different DAGs.

### 4.4 Test surface

| Test | Coverage |
|---|---|
| `test_node_roles_asia` | ASIA (n=8): hub/cut/terminal lists, MB sizes, ancestor/descendant sets |
| `test_node_roles_n1000` | Synthetic n=1000: computation time <1s, disconnected-component handling |
| `test_target_allocator_shortfall` | ASIA cut bucket: 1 articulation point → surplus to random |
| `test_kind_assignment_k1` | terminal bucket: prediction-only; hub: 50/50 verified statistically |
| `test_evidence_pool_prediction` | longest_path: correct ancestor sub-pool; mb_neighbors: correct parent+co-parent pool |
| `test_evidence_pool_diagnosis` | mb_neighbors: correct children+co-children pool |
| `test_value_dedup_b1` | Step 4 B1: same triple not returned twice |
| `test_select_returns_lte_budget` | ASIA at Q=1024: actual count < 1024, no crash |
| `test_select_deterministic` | Same seed → same query list |
| `test_backward_compat` | `target_allocation={random:1.0}`: all queries have `target_bucket="random"` |

---

## §5. Open implementation questions

Answer these before writing any code. They are implementation details, not
design re-opens.

**Q1: Cut-bucket sub-ranking.** When more articulation points exist than
the cut quota, they must be sub-ranked. Options:
- **(a) Degree centrality in the moralised undirected graph** — O(E), correlates
  with treewidth contribution, consistent with the moral-graph pass already done
  for articulation-point detection.
- **(b) Min-fill ordering** — directly related to treewidth; more expensive (~O(V²)).
- **(c) Induced-width heuristic** — intermediate cost, deferred to Phase 3 in
  the redesign doc for `HeaviestQueryByRole`.

**Recommendation:** (a) for Phase 2 — cheap, already using the moral graph,
good enough for target selection. Phase 3 upgrades to (b) or (c) for the
"heaviest cut" scalability benchmark.

**Q2: Longest-path distance definition.** For the `longest_path` evidence
sub-bucket, "distance" from a candidate ancestor node to the target:
- **(a) Topological depth delta** — `depth(target) − depth(candidate)`. O(V),
  reuses the depth dict already computed for terminal scores. May not reflect
  true path length in sparse DAGs.
- **(b) Longest directed path from candidate to target** — exact but O(V × E)
  per query in worst case; feasible only at small n.
- **(c) Undirected shortest path** — `nx.shortest_path_length(G_undirected)`;
  ignores direction.

**Recommendation:** (a) for Phase 2 — reuses NodeRoles.depth, no extra
computation, consistent with the terminal-score approach.

**Q3: `mb_neighbors` for diagnosis kind.** The pool is described as
"children + co-children of target." Co-children are nodes that share at least
one child with the target in the original DAG. Implementation:
```python
children = set(G.successors(target))
co_children = {
    n for child in children
    for n in G.predecessors(child)
    if n != target
}
pool = children | co_children
```
Verify this is the intended semantics before committing to it; it mirrors
the co-parent construction used for MB and the moral graph.

**Q4: Value-tuple dedup key.** The B1 dedup key must be hashable. For
continuous evidence values, `torch.Tensor` is unhashable. Two safe alternatives:
- **(a) Row index as key** — `(target, frozenset(evidence_nodes), row_idx)`.
  Integers are hashable; collision-free since each row is a unique sample.
  Simple, O(1) per query.
- **(b) Rounded float values** — round each float to 4 decimal places and
  hash as a tuple. More precise dedup but fragile for values near rounding
  boundaries.

**Recommendation:** (a) — clean, unambiguous, and the row index is already
computed in Step 4.

---

## §6. Dependencies

- **Issue #74** — umbrella for this work; acceptance criteria 1–5 are the
  gate for the Phase 2 PR.
- **Requires:** Phase 1 complete (PRs #111–#122, all merged). No blocking PRs.
- **Blocks:** Phase 3 (`HeaviestQueryByRole` in `selectors/heaviest.py` imports
  `NodeRoles` from `topological.py`). Phase 4 (`BnlearnProblemSource` uses
  `TopologicalAllocator` for query selection on real-world DAGs).
- **Cross-references from #74:** PR #65 (runner infra, pre-Phase 1), Issue #73
  (bnlearn — independent, can land before or after Phase 2).
- **Related:** Issue #123 (Phase D plotter findings — per-query metadata fields
  `target_bucket` / `query_kind` / `evidence_strategy` will need plotter support
  in Phase 5 for per-role aggregation plots).

---

## §7. Recommendation for next session

1. **Read this spec fresh.** Answer the four questions in §5 (one line each
   is enough). The recommended answers are given; override only if a concrete
   reason exists.

2. **Send a Phase 0 hard-gate relay** that:
   - Reads `benchmarking/core/interfaces.py` (current `Query` fields) and
     `benchmarking/core/runner.py` (lines 236–268) to confirm the exact wiring
     changes needed.
   - Runs a quick disconnected-DAG empirical check:
     `nx.is_connected(nx.moral_graph(G).to_undirected())` on a synthetic n=1000
     DAG to confirm moralisation is safe before the articulation-point call.
   - Produces concrete method signatures for `_TargetAllocator`,
     `_KindEvidenceAllocator`, and `_ValueAllocator`.
   - Estimates actual LOC split across the four internal classes.

3. **Implement Step 1 and `NodeRoles` first.** Get `compute_node_roles()` and
   `_TargetAllocator` passing on ASIA, SACHS, and synthetic n=1000 before
   touching Steps 2–4. These are independently testable and establish the
   cache infrastructure.

4. **Wire per-query metadata into `CellResult` in the same PR.** The three
   new parquet columns (`target_bucket`, `query_kind`, `evidence_strategy`) are
   the visible deliverable of Phase 2; don't defer them to a follow-up.

5. **Write the backward-compat smoke test** (acceptance criterion 5 from #74):
   run with `target_allocation: {random: 1.0}` and confirm parquet values
   match the v0.12 archive within noise. This is the regression gate.
