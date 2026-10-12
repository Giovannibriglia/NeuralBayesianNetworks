"""Continuous mechanisms: a neural location, non-Gaussian noise, counterfactuals.

Ground truth:

    dose  ~ N(0, 1)
    level = sin(2 * dose) + U,   U = +-0.8 with prob 1/2 each, plus N(0, 0.2^2)

The noise is bimodal, so a Gaussian is the wrong law: we fit an MLP location
with a Gaussian-mixture noise, check that one noise law fits every dose
(sharing test), bound P(a < level <= b | dose) with a DKW band, and answer a
counterfactual question for observed patients.  Everything is a batched torch
operation; pass device="mps" or "cuda" to run on an accelerator.

Run:  uv run python examples/03_continuous_noise_and_counterfactuals.py
"""
import torch

from nbn.scm import AdditiveNoiseMechanism, StructuralCausalModel, cdf_band_set, noise_sharing_test

torch.manual_seed(0)


def simulate(n: int) -> dict:
    dose = torch.randn(n, 1)
    bimodal = torch.where(torch.rand(n, 1) < 0.5, -0.8, 0.8) + 0.2 * torch.randn(n, 1)
    return {"dose": dose, "level": torch.sin(2 * dose) + bimodal}


train, held_out = simulate(6000), simulate(3000)


def scm(noise: str) -> StructuralCausalModel:
    model = StructuralCausalModel([("dose", "level")],
                                  {"dose": ("continuous", 1), "level": ("continuous", 1)},
                                  device="cpu")
    model.set_mechanism("dose", AdditiveNoiseMechanism("linear"))
    model.set_mechanism("level", AdditiveNoiseMechanism("mlp", noise, n_components=2,
                                                         epochs=150, batch_size=512))
    model.fit(train)
    return model


gaussian, mixture = scm("gaussian"), scm("mixture")
for name, model in (("gaussian noise", gaussian), ("mixture noise ", mixture)):
    print(f"{name}: held-out mean log-likelihood {model.log_prob(held_out).mean():.3f}")

noise = mixture.mechanisms["level"].noise
print("fitted mixture: weights", [round(w, 2) for w in noise.mix_logits.softmax(-1).tolist()],
      " means", [round(m, 2) for m in noise.means.squeeze(-1).tolist()])

# One noise law for every dose?  (scale of the residuals vs dose and dose^2)
sharing = noise_sharing_test(mixture, held_out, "level")
verdict = ("rejected: the noise scale changes with the dose" if sharing.rejected(0.01)
           else "not rejected at the 1% level")
print(f"\nsharing test: p = {sharing.p_value:.3f} -> {verdict}")
print("  (a small p here can also come from MLP misfit: residual error that varies with the dose)")

# A uniform band on the noise CDF, on held-out residuals (valid given the fitted g).
band = cdf_band_set(mixture, held_out, "level", delta=0.05)
dose = torch.tensor([[0.0], [0.7]])
location = mixture.mechanisms["level"].location(dose)
lower, upper = band.interval_probability_bounds(torch.tensor(0.5), torch.tensor(1.5), location)
print("\n95% bounds on P(0.5 < level <= 1.5 | dose):")
for d, lo, hi in zip(dose.squeeze(-1).tolist(), lower.squeeze(-1).tolist(), upper.squeeze(-1).tolist()):
    print(f"  dose = {d:.1f}: [{lo:.3f}, {hi:.3f}]")

# Counterfactual: these three patients got their observed dose; what level
# would each have had with dose 0.7?  The noise is abducted exactly.
patients = {k: v[:3] for k, v in held_out.items()}
world = mixture.counterfactual_sample(patients, do={"dose": 0.7})
print("\ncounterfactual level under do(dose = 0.7):")
for i in range(3):
    print(f"  patient {i}: dose {patients['dose'][i, 0]:+.2f}, level {patients['level'][i, 0]:+.2f}"
          f"  ->  {world['level'][0, i, 0]:+.2f}")
print("(answers under the additive-noise model: the data identify only P(level | dose))")
