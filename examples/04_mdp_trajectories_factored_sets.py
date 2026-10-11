"""From logged episodes to a factored transition model with a guarantee.

An MDP is a two-slice network: (position_t, battery_t, action_t) -> next slice.

    pos' = clip(pos + act + U, 0, 5),  U in {-1, 0, +1} w.p. (0.1, 0.8, 0.1)   (slip)
    bat' = clip(bat - act + V, 0, 3),  V in { 0, +1}    w.p. (0.7, 0.3)        (recharge)

We flatten episodes into transitions, fit one structural mechanism per
next-state variable, and build per-variable probability sets.  Two details
matter for trajectories:

* visit counts are random and depend on earlier draws, so the sets use
  ``sampling="trajectory"`` (each interval valid for every sample size);
* the kernel is the product of the per-variable tables, so the global budget
  ``delta`` is split between the variables (union bound).  Solving over the
  product of the sets (exact vertices / McCormick) happens downstream.

Run:  uv run python examples/04_mdp_trajectories_factored_sets.py
"""
import torch

from nbn.scm import (
    DiscreteAdditiveNoiseMechanism,
    StructuralCausalModel,
    noise_parameter_set,
    transitions_from_episodes,
)
from nbn.uncertainty import interval_set

torch.manual_seed(0)
SLIP = torch.tensor([0.1, 0.8, 0.1])
RECHARGE = torch.tensor([0.0, 0.7, 0.3])     # offsets -1, 0, +1


def collect(n_episodes: int = 200, horizon: int = 20) -> dict:
    position = torch.zeros(n_episodes, horizon + 1, dtype=torch.long)
    battery = torch.full((n_episodes, horizon + 1), 3, dtype=torch.long)
    action = torch.randint(0, 2, (n_episodes, horizon))            # 0 = stay, 1 = move right
    for t in range(horizon):
        u = torch.multinomial(SLIP, n_episodes, replacement=True) - 1
        v = torch.multinomial(RECHARGE, n_episodes, replacement=True) - 1
        position[:, t + 1] = (position[:, t] + action[:, t] + u).clamp(0, 5)
        battery[:, t + 1] = (battery[:, t] - action[:, t] + v).clamp(0, 3)
    return transitions_from_episodes({"pos": position.float(), "bat": battery.float()},
                                     {"act": action.float()})


data = collect()
print(f"{int(data['episode'].max()) + 1} episodes -> {data['pos'].numel()} transitions")

model = StructuralCausalModel(
    [("pos", "pos'"), ("act", "pos'"), ("bat", "bat'"), ("act", "bat'")],
    {"pos": ("discrete", 6), "act": ("discrete", 2), "bat": ("discrete", 4),
     "pos'": ("discrete", 6), "bat'": ("discrete", 4)},
    device="cpu",
)
model.set_mechanism("pos'", DiscreteAdditiveNoiseMechanism(max_offset=1))
model.set_mechanism("bat'", DiscreteAdditiveNoiseMechanism(max_offset=1))
model.auto_mechanisms()
model.fit({k: data[k] for k in ("pos", "act", "bat", "pos'", "bat'")})
for v in ("pos'", "bat'"):
    print(f"{v} noise law: {[round(p, 3) for p in model.mechanisms[v].noise.probs.tolist()]}")

# One set per next-state variable; the kernel's guarantee is 1 - delta overall.
delta, next_state = 0.05, ["pos'", "bat'"]
per_variable = delta / len(next_state)
box = {v: interval_set(model, data, v, delta=per_variable, sampling="trajectory")
       for v in next_state}
pooled = {v: noise_parameter_set(model, data, [v], delta=per_variable,
                                 sampling="trajectory").table_set(v)
          for v in next_state}

print("\nmean width on seen rows      per-entry box   shared-noise polytope")
for v in next_state:
    seen = ~box[v].unseen
    print(f"  {v:5s}                       {box[v].width()[seen].mean():.3f}"
          f"           {pooled[v].width()[seen].mean():.3f}")

# The transition P(s' | s, a) is the product of the per-variable rows; its
# entry-wise bounds are products of the factor bounds (all factors >= 0).
pos, bat, act = 2, 3, 1
row_pos, row_bat = pos * 2 + act, bat * 2 + act      # C order over (parent, act)
print("\nP(pos'=3, bat'=2 | pos=2, bat=3, act=move)   true value", round(0.8 * 0.7, 3))
for name, sets in (("per-entry box", box), ("noise polytope", pooled)):
    lo_p, hi_p = sets["pos'"].bounds()
    lo_b, hi_b = sets["bat'"].bounds()
    print(f"  {name:15s} [{lo_p[row_pos, 3] * lo_b[row_bat, 2]:.3f}, "
          f"{hi_p[row_pos, 3] * hi_b[row_bat, 2]:.3f}]")
