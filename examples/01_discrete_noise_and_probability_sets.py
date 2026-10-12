"""Learn a discrete noise law and put a guarantee on the probabilities.

Ground truth (hidden from the learner):

    health' = clip(health + exercise * (rested - tired) + U, 0, 3),
    U in {-1, 0, +1} with probabilities (0.15, 0.70, 0.15).

We fit a structural mechanism, read back the noise law, and compare three
probability sets that all contain the true table with probability >= 90%:
per-entry intervals, L1 balls, and the polytope on the shared noise law.

Run:  uv run python examples/01_discrete_noise_and_probability_sets.py
"""
import torch

from nbn.scm import DiscreteAdditiveNoiseMechanism, StructuralCausalModel, noise_parameter_set
from nbn.uncertainty import interval_set, l1_ball_set

torch.manual_seed(0)
TRUE_NOISE = torch.tensor([0.15, 0.70, 0.15])


def simulate(n: int) -> dict:
    health = torch.randint(0, 4, (n,))
    tired = torch.randint(0, 2, (n,))
    exercise = torch.randint(0, 2, (n,))
    u = torch.multinomial(TRUE_NOISE, n, replacement=True) - 1
    nxt = (health + exercise * (1 - 2 * tired) + u).clamp(0, 3)
    return {"health": health.float(), "tired": tired.float(), "exercise": exercise.float(),
            "health'": nxt.float()}


data = simulate(3000)

# 1. A structural causal model: health' gets an explicit noise law.
model = StructuralCausalModel(
    [("health", "health'"), ("tired", "health'"), ("exercise", "health'")],
    {"health": ("discrete", 4), "tired": ("discrete", 2), "exercise": ("discrete", 2),
     "health'": ("discrete", 4)},
    device="cpu",                      # "mps" / "cuda" work the same way
)
model.set_mechanism("health'", DiscreteAdditiveNoiseMechanism(max_offset=1))
model.auto_mechanisms()                # plain tables for the other nodes
model.fit(data)

mech = model.mechanisms["health'"]
print("offsets          ", mech.noise.offsets.tolist())
print("learned noise law", [round(p, 3) for p in mech.noise.probs.tolist()])
print("true noise law   ", [round(p, 3) for p in TRUE_NOISE.tolist()])

# 2. It is still an ordinary NBN: interventional queries work as before.
print("\nP(health' | health=1, rested, do(exercise=1)) =",
      [round(p, 3) for p in model.query(["health'"], evidence={"health": 1, "tired": 0},
                                        do={"exercise": 1}).reshape(-1).tolist()])

# 3. Three probability sets at level 0.90 (valid for all entries at once).
box = interval_set(model, data, "health'", delta=0.1)          # min / max per probability
ball = l1_ball_set(model, data, "health'", delta=0.1)          # one L1 ball per row
polytope = noise_parameter_set(model, data, ["health'"], delta=0.1)   # on the shared law

print("\nmean width of the bounds (smaller = tighter):")
print(f"  per-entry intervals {box.width().mean():.3f}")
print(f"  L1 balls            {ball.width().mean():.3f}")
noise_box = polytope.table_set("health'")
print(f"  noise polytope      {noise_box.width().mean():.3f}")
print("true noise law inside the polytope:", polytope.contains(TRUE_NOISE.double()))

# 4. The bounds of one row, e.g. health=1, rested, exercise=1  (row index in C order).
row = (1 * 2 + 0) * 2 + 1
lower, upper = noise_box.bounds()
print("\nP(health' | health=1, rested, exercise=1):")
for k in range(4):
    print(f"  health'={k}: [{lower[row, k]:.3f}, {upper[row, k]:.3f}]")
