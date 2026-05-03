"""Continuous + discrete do-interventions on hybrid NBNs.

Two demonstrations:
    1. Continuous chain (A → B → C): show do(B = v) is implemented as a
       Dirac-tight-Gaussian and breaks the A → C path through B.
    2. Discrete chain (S → E → I): show do(E = k) is implemented as a
       delta-Categorical.
"""
from __future__ import annotations

from pathlib import Path

import torch

from nbn import NeuralBayesianNetwork, seed_all
from nbn.mechanisms import CategoricalTableMechanism, LinearGaussianMechanism

seed_all(0)


# --- 1. Continuous: A → B → C ; A → C  ---------------------------------------
print("=== Continuous intervention ===")
edges = [("A", "B"), ("A", "C"), ("B", "C")]
model = NeuralBayesianNetwork(
    edges, variables=dict.fromkeys("ABC", ("continuous", 1)), device="cpu",
)
n = 4000
a = torch.randn(n, 1)
b = 2 * a + 0.5 * torch.randn(n, 1)
c = a + b + 0.3 * torch.randn(n, 1)
data = {"A": a, "B": b, "C": c}
for nm in model.dag.topological_order():
    parents = model.dag.parents(nm)
    pa_t = torch.cat([data[p] for p in parents], dim=-1) if parents else None
    mech = LinearGaussianMechanism()
    mech.fit_local(data[nm], pa_t)
    model.set_mechanism(nm, mech)

print(f"  Baseline:   E[C] = {model.sample(n=2000)['C'].mean().item():+6.3f}")
do_model = model.intervene({"B": torch.tensor([2.0])})
print(f"  do(B=2.0):  E[C] = {do_model.sample(n=2000)['C'].mean().item():+6.3f}")
print("  -> B is clamped via DiracGaussianMechanism (sigma=1e-4); "
      "the A → B → C path is severed.")


# --- 2. Discrete: S → E ; S → I  --------------------------------------------
print("\n=== Discrete intervention ===")
dedges = [("S", "E"), ("S", "I")]
dmodel = NeuralBayesianNetwork(
    dedges, variables={"S": ("discrete", 2), "E": ("discrete", 3),
                       "I": ("discrete", 4)}, device="cpu",
)
torch.manual_seed(1)
s = torch.randint(0, 2, (1000,))
e = (s + torch.randint(0, 3, (1000,))).clamp(0, 2)
i = (s * 2 + torch.randint(0, 3, (1000,))).clamp(0, 3)

m_s = CategoricalTableMechanism(); m_s.fit_local(s, None); dmodel.set_mechanism("S", m_s)
m_e = CategoricalTableMechanism(); m_e.fit_local(e, s.unsqueeze(-1).float(), parent_cards=[2])
dmodel.set_mechanism("E", m_e)
m_i = CategoricalTableMechanism(); m_i.fit_local(i, s.unsqueeze(-1).float(), parent_cards=[2])
dmodel.set_mechanism("I", m_i)

base_e = dmodel.sample(n=2000)["E"].squeeze(-1).long()
print(f"  Baseline:   P(E=2) = {(base_e == 2).float().mean().item():.3f}")
do_dmodel = dmodel.intervene({"E": torch.tensor([2])})
do_e = do_dmodel.sample(n=2000)["E"].squeeze(-1).long()
print(f"  do(E=2):    P(E=2) = {(do_e == 2).float().mean().item():.3f}")
print("  -> E is clamped via DeterministicMechanism.")


# --- 3. Optional KDE figure --------------------------------------------------
try:
    import matplotlib.pyplot as plt
    from benchmarking.style import NBN_PALETTE, apply_style, savefig_multi

    apply_style()
    fig, ax = plt.subplots(figsize=(5, 3))
    bins = torch.linspace(-8, 12, 60)
    base_c = model.sample(n=4000)["C"].squeeze(-1)
    do_c = do_model.sample(n=4000)["C"].squeeze(-1)
    ax.hist(base_c.detach().numpy(), bins=bins, alpha=0.5, label="P(C)",
            color=NBN_PALETTE["nbn"], density=True)
    ax.hist(do_c.detach().numpy(), bins=bins, alpha=0.5, label="P(C | do(B=2))",
            color=NBN_PALETTE["pgmpy"], density=True)
    ax.set_xlabel("C"); ax.set_ylabel("density"); ax.legend(frameon=False)
    Path("examples/figures").mkdir(parents=True, exist_ok=True)
    savefig_multi(fig, "examples/figures/intervention_effects")
    print("\nSaved examples/figures/intervention_effects.{pdf,png,svg}")
except ImportError:
    pass
