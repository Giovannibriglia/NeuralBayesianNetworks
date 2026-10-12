"""Check the assumptions behind an SCM: shared noise, fresh noise, confounded loggers.

Three behaviour policies log the same rehab environment.  A hidden strength
``W`` changes the effect of exercising.  The "doctor" sees ``W`` when it
decides, so its data are confounded; the other two loggers are not.

We run the four diagnostics of ``nbn.scm``:

* ``context_invariance_test`` — does P(health' | health, exercise) change with
  the logger?  (policy-as-context test; a rejection flags confounding)
* ``noise_sharing_test`` / ``noise_tying_test`` — may parent configurations /
  several nodes share one noise law?
* ``freshness_test`` — is the noise drawn afresh at every step?

Run:  uv run python examples/02_confounding_and_noise_diagnostics.py
"""
import torch

from nbn.scm import (
    DiscreteAdditiveNoiseMechanism,
    StructuralCausalModel,
    benjamini_hochberg,
    context_invariance_test,
    freshness_test,
    noise_sharing_test,
    noise_tying_test,
    residuals,
    transitions_from_episodes,
)

torch.manual_seed(0)
NOISE = torch.tensor([0.15, 0.70, 0.15])


def episodes(policy: str, n_episodes: int = 300, horizon: int = 10) -> dict:
    """Trajectories of (health, fatigue) under one logger; W is hidden and fresh each step."""
    health = torch.zeros(n_episodes, horizon + 1, dtype=torch.long)
    fatigue = torch.zeros(n_episodes, horizon + 1, dtype=torch.long)
    action = torch.zeros(n_episodes, horizon, dtype=torch.long)
    health[:, 0] = torch.randint(0, 4, (n_episodes,))
    for t in range(horizon):
        strong = torch.rand(n_episodes) < 0.4                      # hidden W
        if policy == "random":
            a = torch.rand(n_episodes) < 0.5
        elif policy == "state_based":
            a = health[:, t] < 2
        else:                                                      # "doctor" sees W
            a = torch.rand(n_episodes) < torch.where(strong, 0.9, 0.1)
        a = a.long()
        effect = a * (2 * strong.long() - 1)                       # helps strong, hurts weak
        u = torch.multinomial(NOISE, n_episodes, replacement=True) - 1
        v = torch.multinomial(NOISE, n_episodes, replacement=True) - 1
        health[:, t + 1] = (health[:, t] + effect + u).clamp(0, 3)
        fatigue[:, t + 1] = (a + v).clamp(0, 1)
        action[:, t] = a
    return transitions_from_episodes({"health": health.float(), "fatigue": fatigue.float()},
                                     {"exercise": action.float()})


loggers = ["random", "state_based", "doctor"]
parts = [episodes(p) for p in loggers]
data = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
data["logger"] = torch.cat([torch.full_like(p["time"], i) for i, p in enumerate(parts)]).float()
data["episode"] = torch.cat([p["episode"] + 1000 * i for i, p in enumerate(parts)])


def model_for(rows: dict) -> StructuralCausalModel:
    model = StructuralCausalModel(
        [("health", "health'"), ("exercise", "health'"), ("exercise", "fatigue'")],
        {"health": ("discrete", 4), "exercise": ("discrete", 2), "health'": ("discrete", 4),
         "fatigue'": ("discrete", 2), "logger": ("discrete", 3)},
        context="logger", device="cpu",
    )
    # health' moves by the exercise effect (+-1, depending on the hidden W) plus
    # U (+-1): seen without W, offsets reach +-2.  With max_offset=1 the fit
    # refuses the data with a clear error instead of silently misfitting.
    model.set_mechanism("health'", DiscreteAdditiveNoiseMechanism(max_offset=2))
    model.set_mechanism("fatigue'", DiscreteAdditiveNoiseMechanism(max_offset=1))
    model.auto_mechanisms()
    model.fit({k: rows[k] for k in ("health", "exercise", "health'", "fatigue'", "logger")})
    return model


# 1. Policy-as-context test: with and without the doctor's data.
print("context invariance of health' (H0: the logger does not matter given the parents)")
for name, keep in [("random + state_based", data["logger"] < 2), ("all three loggers", None)]:
    rows = data if keep is None else {k: v[keep] for k, v in data.items()}
    result = context_invariance_test(model_for(rows), rows, "health'")
    print(f"  {name:22s} p = {result.p_value:.2e}  ->  "
          f"{'confounding or drift detected' if result.rejected() else 'no evidence of confounding'}")

# 2. Noise diagnostics on the unconfounded data only (confounded rows never feed do-estimates).
clean = {k: v[data["logger"] < 2] for k, v in data.items()}
model = model_for(clean)
results = {
    "sharing (health')": noise_sharing_test(model, clean, "health'"),
    "tying (health', fatigue')": noise_tying_test(model, clean, ["health'", "fatigue'"]),
    "freshness (health')": freshness_test(residuals(model, clean, "health'"),
                                          clean["episode"], clean["time"]),
}
reject = benjamini_hochberg(torch.tensor([r.p_value for r in results.values()]), alpha=0.05)
print("\nnoise diagnostics on the clean loggers (Benjamini-Hochberg at 5%)")
for (name, r), rej in zip(results.items(), reject.tolist()):
    print(f"  {name:28s} p = {r.p_value:.3f}  {'REJECT' if rej else 'ok'}   H0: {r.null}")

print("""
Reading the results
  * Only the doctor's data change P(health' | health, exercise): it saw W.
  * Sharing is rejected even on clean data: with W hidden, exercising adds a
    +-1 mixture, so the noise of health' really depends on 'exercise' — one
    shared law would be a wrong (too tight) model.  Use a full table for
    health' (CategoricalTableMechanism), or observe W.
  * Tying health' and fatigue' is rejected for the same reason.
  * Freshness: no evidence against fresh noise (W is drawn anew each step in
    this simulator).  A pass is not a proof; it is the check that per-step
    confounding bounds (Manski / causal Bellman rows) need.""")
