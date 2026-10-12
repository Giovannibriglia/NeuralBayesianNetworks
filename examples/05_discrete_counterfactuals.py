"""Discrete counterfactuals: declared couplings, sharp bounds, calibrated sets.

Ground truth (one MDP transition, ordered health levels 0 < 1 < 2):

    treatment a in {0: none, 1: mild, 2: strong}  ~ uniform (logged policy)
    health'   ~ P(. | a)                          (true table below)

A patient got no treatment (a = 0) and ended in health 0.  Would the strong
treatment have helped?  Data identify only P(health' | a), so the answer
depends on how the noise couples different treatments.  We show:

1. Two declared couplings (Gumbel-max, monotone) with a neural kernel: two
   different point answers from the same fitted kernel.
2. The bounds that hold for every SCM with that kernel: both answers inside.
3. Bounds that also cover estimation error (a confidence box on the kernel).
4. Calibration of the kernel itself: a randomized-PIT band and conformal
   prediction sets on held-out data.

Run:  uv run python examples/05_discrete_counterfactuals.py
"""
import torch

from nbn.scm import (
    StructuralCategoricalTable,
    StructuralCausalModel,
    StructuralNeuralCategorical,
    calibration_by_bins_test,
    conformal_sets,
    counterfactual_bounds_from_sets,
    pit_band_set,
)
from nbn.uncertainty import interval_set  # entry-wise PAC intervals on P(h | a), all entries at once

torch.manual_seed(0)
TRUE = torch.tensor([[0.60, 0.30, 0.10],   # a = 0
                     [0.30, 0.45, 0.25],   # a = 1
                     [0.10, 0.30, 0.60]])  # a = 2


def simulate(n: int, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    a = torch.randint(0, 3, (n,), generator=gen)
    h = torch.multinomial(TRUE[a], 1, generator=gen).reshape(-1)
    return {"a": a.float().unsqueeze(-1), "h": h.float().unsqueeze(-1)}


train, held_out = simulate(3000, seed=1), simulate(1500, seed=2)


# Fit the kernel once; the coupling is a declared reading of it, not something learned.
model = StructuralCausalModel([("a", "h")], {"a": ("discrete", 3), "h": ("discrete", 3)}, device="cpu")
model.set_mechanism("a", StructuralCategoricalTable())
model.set_mechanism("h", StructuralNeuralCategorical(n_classes=3, hidden=(32,), coupling="monotone"))
model.fit(train)
mech = model.mechanisms["h"]

evidence = {"a": torch.zeros(20000, 1), "h": torch.zeros(20000, 1)}   # untreated, ended in 0
print("Counterfactual P(health' = k | did a=0 and got 0, had we done a=2):")
for coupling in ("gumbel_max", "monotone"):
    model.set_mechanism("h", mech.with_coupling(coupling))
    world = model.counterfactual_sample(evidence, do={"a": 2.0})["h"][0, :, 0].long()
    print(f"  {coupling:>10}: {(torch.bincount(world, minlength=3).float() / 20000).cpu().numpy().round(3)}")
mech = model.mechanisms["h"]

with torch.no_grad():
    lo, hi = mech.counterfactual_bounds(torch.zeros(1, 1), torch.zeros(1, 1), torch.full((1, 1), 2.0))
print(f"  every compatible SCM lies in  lower {lo[0].cpu().numpy().round(3)}  upper {hi[0].cpu().numpy().round(3)}")
print("  (both answers lie inside, up to Monte-Carlo error of about 0.005)")

# Estimation error too: a 95% box on every entry of P(h | a) (rows = a).
q_lo, q_hi = interval_set(model, train, "h", delta=0.05).bounds()
lo_s, hi_s = counterfactual_bounds_from_sets(q_lo[0:1], q_lo[2:3], q_hi[2:3], torch.tensor([0]))
print(f"  ... and every kernel in a 95% box: lower {lo_s[0].cpu().numpy().round(3)}  upper {hi_s[0].cpu().numpy().round(3)}")

# Calibration of the fitted kernel on held-out rows.
gen = torch.Generator().manual_seed(0)
band = pit_band_set(model, held_out, "h", delta=0.05, generator=gen)
cdf_lo, cdf_hi = band.cdf_bounds(mech, torch.zeros(3, 1), torch.tensor([[0.0], [1.0], [2.0]]))
print("\nP(health' <= 0 | a) with a 95% band (if the PIT does not depend on a):")
for a in range(3):
    print(f"  a={a}: [{float(cdf_lo[a]):.3f}, {float(cdf_hi[a]):.3f}]   truth {float(TRUE[a, 0]):.3f}")
print("calibration by bins, p-value:",
      round(calibration_by_bins_test(model, held_out, "h", n_bins=3, generator=gen).p_value, 3))
sets = conformal_sets(model, held_out, "h", alpha=0.2).sets(mech, torch.tensor([[0.0], [1.0], [2.0]]))
print("80% conformal prediction sets per treatment:", [torch.nonzero(s).flatten().tolist() for s in sets])
