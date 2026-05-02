"""NBN crash test: show speed on batched conditional queries.

Headline demo:
1. Build the asia BN (8 nodes, fully discrete).
2. Run 10_000 batched conditional queries via TensorVariableElimination.
3. Compare against pgmpy's exact VariableElimination (single-query loop).
4. Save a bar chart `crash_test_results.png`.

Run with:
    python examples/crash_test.py
"""
from __future__ import annotations

import time

import torch
from torch import nn

from nbn import NeuralBayesianNetwork, TensorVariableElimination, LikelihoodWeightingEngine, seed_all
from nbn.mechanisms import CategoricalTableMechanism


seed_all(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_QUERIES = 1_000


# ── 1. Build the asia network ────────────────────────────────────────────────
edges = [
    ("asia", "tub"), ("smoke", "lung"), ("smoke", "bronc"),
    ("tub", "either"), ("lung", "either"),
    ("either", "xray"), ("either", "dysp"), ("bronc", "dysp"),
]
nodes = ["asia", "smoke", "tub", "lung", "bronc", "either", "xray", "dysp"]
cpts = {
    "asia":   {(): [0.99, 0.01]}, "smoke":  {(): [0.5, 0.5]},
    "tub":    {(0,): [0.99, 0.01], (1,): [0.95, 0.05]},
    "lung":   {(0,): [0.99, 0.01], (1,): [0.9, 0.1]},
    "bronc":  {(0,): [0.7, 0.3],  (1,): [0.4, 0.6]},
    "either": {(0,0): [1.0,0.0], (0,1): [0.0,1.0],
               (1,0): [0.0,1.0], (1,1): [0.0,1.0]},
    "xray":   {(0,): [0.95, 0.05], (1,): [0.02, 0.98]},
    "dysp":   {(0,0): [0.9,0.1], (0,1): [0.2,0.8],
               (1,0): [0.3,0.7], (1,1): [0.1,0.9]},
}
parent_order = {
    "asia": [], "smoke": [], "tub": ["asia"], "lung": ["smoke"],
    "bronc": ["smoke"], "either": ["tub", "lung"],
    "xray": ["either"], "dysp": ["either", "bronc"],
}

model = NeuralBayesianNetwork(edges, variables={n: ("discrete", 2) for n in nodes})
for node in model.dag.topological_order():
    parents = parent_order[node]
    log_cpt = torch.zeros(max(1, 2 ** len(parents)), 2)
    for pa, probs in cpts[node].items():
        row = sum(v * (2 ** (len(pa) - 1 - i)) for i, v in enumerate(pa))
        log_cpt[row] = torch.log(torch.tensor(probs).clamp_min(1e-9))
    mech = CategoricalTableMechanism()
    mech._logits = nn.Parameter(log_cpt)
    mech._n_classes = 2
    mech._parent_cards = [2] * len(parents)
    mech._parent_strides = [2 ** (len(parents) - 1 - i) for i in range(len(parents))]
    mech._class_values = torch.tensor([0.0, 1.0])
    mech.output_dim = 1
    model.set_mechanism(node, mech)
model.to(DEVICE)


# ── 2. NBN: TensorVE timing ──────────────────────────────────────────────────
print(f"Running {N_QUERIES} queries on device={DEVICE}…")
ve = TensorVariableElimination()
rng = torch.Generator().manual_seed(0)
asia_vals = torch.randint(0, 2, (N_QUERIES,), generator=rng)
smoke_vals = torch.randint(0, 2, (N_QUERIES,), generator=rng)

t0 = time.perf_counter()
nbn_results = []
for i in range(N_QUERIES):
    p = ve.query(model, ["dysp"], evidence={"asia": asia_vals[i], "smoke": smoke_vals[i]})
    nbn_results.append(p)
nbn_time = time.perf_counter() - t0
print(f"  NBN TensorVE: {nbn_time:.2f} s ({nbn_time / N_QUERIES * 1e3:.2f} ms/query)")


# ── 3. NBN: LikelihoodWeighting batched timing ──────────────────────────────
lw = LikelihoodWeightingEngine(n_samples=512)
t0 = time.perf_counter()
batch_p = lw.query_batch(
    model, ["dysp"],
    evidence={"asia": asia_vals.float().unsqueeze(-1),
              "smoke": smoke_vals.float().unsqueeze(-1)},
)
lw_time = time.perf_counter() - t0
print(f"  NBN LW batch: {lw_time:.2f} s ({lw_time / N_QUERIES * 1e3:.2f} ms/query)")


# ── 4. pgmpy comparison ─────────────────────────────────────────────────────
pgmpy_time = None
try:
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination

    pmodel = DiscreteBayesianNetwork(edges)
    for node in nodes:
        parents = parent_order[node]
        n_par = max(1, 2 ** len(parents))
        # pgmpy expects values of shape [n_classes, n_par_states]
        values = torch.zeros(2, n_par)
        for pa, probs in cpts[node].items():
            row = sum(v * (2 ** (len(pa) - 1 - i)) for i, v in enumerate(pa))
            values[:, row] = torch.tensor(probs)
        pmodel.add_cpds(TabularCPD(
            variable=node, variable_card=2,
            values=values.tolist(),
            evidence=parents if parents else None,
            evidence_card=[2] * len(parents) if parents else None,
        ))
    pinf = VariableElimination(pmodel)
    t0 = time.perf_counter()
    for i in range(N_QUERIES):
        pinf.query(["dysp"],
                   evidence={"asia": int(asia_vals[i]), "smoke": int(smoke_vals[i])},
                   show_progress=False)
    pgmpy_time = time.perf_counter() - t0
    print(f"  pgmpy VE:     {pgmpy_time:.2f} s ({pgmpy_time / N_QUERIES * 1e3:.2f} ms/query)")
except Exception as e:
    print(f"  pgmpy: SKIPPED ({e})")


# ── 5. Plot ─────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    labels, values = ["NBN VE\n(loop)", "NBN LW\n(batched)"], [nbn_time, lw_time]
    if pgmpy_time is not None:
        labels.append("pgmpy VE\n(loop)")
        values.append(pgmpy_time)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=["#1f77b4", "#2ca02c", "#d62728"][: len(labels)])
    ax.set_ylabel(f"Time for {N_QUERIES:,} queries (s)")
    ax.set_title("NBN crash test on asia BN: throughput comparison")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}s",
                ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig("crash_test_results.png", dpi=120)
    print("Saved crash_test_results.png")
except ImportError:
    print("matplotlib not installed — skipping figure.")


print()
print(f"NBN: {N_QUERIES} queries in {nbn_time:.2f}s "
      f"({nbn_time / N_QUERIES * 1e3:.2f} ms/query) on {DEVICE}; "
      f"NBN LW batched: {lw_time:.2f}s; "
      f"pgmpy: {pgmpy_time:.2f}s" if pgmpy_time is not None else "pgmpy: SKIPPED")
