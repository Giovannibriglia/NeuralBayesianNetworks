# What was done, in nine plots

Each plot answers one question.  Words used below:

* **loose graph**: each node's parents are among the 5 nodes just before it,
  so the graph never gets "thick" (tree-width ≤ 5) however many nodes it has;
* **dense graph**: parents drawn from *all* earlier nodes (Erdős–Rényi), so
  the graph gets thicker as it grows;
* **budget**: 1 CPU thread, 4 GB of memory, 30 s per run (the GPU is the
  Apple M5's);
* **NBN before**: NBN `master`, or the old one-node-at-a-time training path.

Regenerate with `python benchmarks/scaling/explain.py` (needs the saved
results; plot 03 also needs `measure_ordering.py` run on both checkouts).

| plot | question | what to look at |
|---|---|---|
| `01_overview_frontier` | How far does each method get within the budget? | Dark circles (this branch) at the right edge = reaches the largest size tested (5 000 nodes). Exact queries on dense graphs stop at ~100 nodes for *everyone*; the guaranteed sampling answers do not stop. |
| `02_what_each_change_bought` | What did each change buy? | Before/after factors with the same answers: 51× (exact queries), 12–32× (sampling queries), 6× (S4 sets), 3.4× (non-Gaussian fit), 8× less memory (sparse table), 2× narrower guaranteed intervals. |
| `03_where_exact_query_time_went` | Why were exact queries slow? | Before, 95% of the time went into *choosing the elimination order* (re-scoring every node and copying the graph at each step). Now that step is incremental and gives the identical order. |
| `04_treewidth_wall` | Why can't exact inference scale on dense graphs? | Each dot is a query. Its "width" sets the size of the biggest table, K^(width+1). On loose graphs the width stays ≤ 6; on dense graphs it grows with size and crosses the memory line: those queries are refused. No ordering trick removes this (finding the best order is NP-hard). |
| `05_certified_answers` | What does a "certified" answer look like? | Black dot = exact answer; bars = intervals guaranteed to contain it with ≥ 95% probability; the pink (KL ∩ Bernstein) bars are half as wide as the grey (KL only). The green triangle is plain sampling: close, but with no guarantee. |
| `06_stacked_fit_cost_per_node` | Why is stacked fitting faster, and why does it make the GPU useful? | Cost per node: one-node-at-a-time training (and Pyro) pays a fixed overhead per node and is *slowest* on the GPU; stacked training batches all nodes, so the cost per node keeps falling, most of all on the GPU. Same objective per node. |
| `07_sparse_table_memory` | How does the sparse table save memory? | A full table needs K^(parents+1) numbers (10 billion at 20 parents); with 10 000 data rows at most 10 000 parent combinations are ever seen, so the sparse table stores at most K × 10 000. Same answers. |
| `08_data_fragmentation` | Then is "many parents" solved? | No. Computing is solved, statistics is not: with 20 parents almost every parent combination is seen once, and the fitted table's error grows from 0.01 to 0.25. That needs a different model type, not faster code. |
| `09_barren_pruning` | Why are sampling queries 12–32× faster? | Before, every query sampled every node. Only the ancestors of the target and the evidence can change the answer; on a 5 000-node dense graph that is about 5% of the nodes. |
