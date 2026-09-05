"""Calibration diagnostics for conditional-density mechanisms.

A small, dependency-light harness for assessing how well a fitted mechanism's
predictive conditional distribution is *calibrated* — not just whether its mean
is right. Sampling-based, so it works for any mechanism exposing the standard
``fit_local`` / ``log_prob`` / ``sample`` API.

Metrics (per SCM, per mechanism), computed on held-out test points:

  PIT-KS    Kolmogorov-Smirnov distance of the Probability Integral Transform
            values from Uniform(0, 1). PIT_i = mean(samples_i <= y_i). 0 is
            perfect; large means the predicted CDF is the wrong shape.
  cov50     Empirical coverage of the central 50% predictive interval (target 0.50).
  cov90     Empirical coverage of the central 90% predictive interval (target 0.90).
            cov > target => intervals too wide (under-confident); < => over-confident.
  CRPS      Mean Continuous Ranked Probability Score (energy form). Lower is better;
            a proper score rewarding both accuracy and sharpness.
  sd_ratio  Mean predictive std / true conditional std. ~1 is right sharpness,
            > 1 oversmoothed/under-confident, < 1 over-confident.
  NLL       Mean negative log_prob on the test set (proper score). Lower is better.

Run as a script for the default panel (the three continuous non-parametric
mechanisms across three synthetic SCMs):

    python -m nbn.bench.calibration_diagnostic
    python nbn/bench/calibration_diagnostic.py --n-train 4000 --seed 1

Or import :func:`assess_calibration` to score any mechanism on your own data.
"""
from __future__ import annotations

import argparse
from typing import Callable

import numpy as np
import torch

# ───────────────────────────── synthetic SCMs ──────────────────────────────────
# Each returns (y [N,1], parents [N,1], true_conditional_sd [N,1]).


def scm_homoscedastic(n: int, gen: torch.Generator) -> tuple[torch.Tensor, ...]:
    """Linear-Gaussian, constant noise: y = 1.5 x + N(0, 0.3^2)."""
    x = torch.rand(n, 1, generator=gen) * 4 - 2
    y = 1.5 * x + 0.30 * torch.randn(n, 1, generator=gen)
    return y, x, torch.full((n, 1), 0.30)


def scm_heteroscedastic(n: int, gen: torch.Generator) -> tuple[torch.Tensor, ...]:
    """Nonlinear mean, input-dependent noise: y = sin(2x) + (0.1 + 0.3|x|) eps."""
    x = torch.rand(n, 1, generator=gen) * 4 - 2
    sd = 0.10 + 0.30 * x.abs()
    y = torch.sin(2 * x) + sd * torch.randn(n, 1, generator=gen)
    return y, x, sd


def scm_bimodal(n: int, gen: torch.Generator) -> tuple[torch.Tensor, ...]:
    """Equal-weight bimodal conditional: y ~ 1/2 N(+a, s^2) + 1/2 N(-a, s^2),
    a = 0.8 + 0.4|x|, s = 0.15. True conditional sd = sqrt(s^2 + a^2)."""
    x = torch.rand(n, 1, generator=gen) * 4 - 2
    a = 0.8 + 0.4 * x.abs()
    s = 0.15
    sign = torch.where(torch.rand(n, 1, generator=gen) < 0.5, 1.0, -1.0)
    y = sign * a + s * torch.randn(n, 1, generator=gen)
    true_sd = torch.sqrt(s**2 + a**2)
    return y, x, true_sd


DEFAULT_SCMS: dict[str, Callable] = {
    "homoscedastic": scm_homoscedastic,
    "heteroscedastic": scm_heteroscedastic,
    "bimodal": scm_bimodal,
}


# ─────────────────────────────── core metric ───────────────────────────────────

def _crps_from_samples(samples: torch.Tensor, y: torch.Tensor) -> float:
    """Mean CRPS via the energy form  E|X - y| - 1/2 E|X - X'|  (O(S log S))."""
    n, s = samples.shape
    term1 = (samples - y[:, None]).abs().mean(dim=1)
    s_sorted, _ = torch.sort(samples, dim=1)
    weights = (2 * torch.arange(1, s + 1, dtype=samples.dtype) - s - 1)
    term2 = (s_sorted * weights[None, :]).sum(dim=1) / (s * s)
    return float((term1 - term2).mean())


def assess_calibration(
    mech,
    y_train: torch.Tensor,
    pa_train: torch.Tensor,
    y_test: torch.Tensor,
    pa_test: torch.Tensor,
    true_sd_test: torch.Tensor,
    n_samples: int = 400,
    fit_kwargs: dict | None = None,
) -> dict[str, float]:
    """Fit ``mech`` on the train split and score calibration on the test split.

    The mechanism must implement ``fit_local(x, parents, **kw)``,
    ``log_prob(x, parents)`` and ``sample(parents, n)``. Returns a dict with keys
    ``ks, cov50, cov90, crps, sd_ratio, nll``.
    """
    mech.fit_local(y_train, pa_train, **(fit_kwargs or {}))
    with torch.no_grad():
        samp = mech.sample(pa_test, n=n_samples).squeeze(-1)   # [N_test, S]
        yte = y_test.squeeze(-1)

        pit = (samp <= yte[:, None]).double().mean(dim=1)
        pit_sorted = torch.sort(pit).values.numpy()
        unif = np.arange(1, len(pit_sorted) + 1) / len(pit_sorted)
        ks = float(np.max(np.abs(pit_sorted - unif)))

        def coverage(level: float) -> float:
            lo = torch.quantile(samp, (1 - level) / 2, dim=1)
            hi = torch.quantile(samp, 1 - (1 - level) / 2, dim=1)
            return float(((yte >= lo) & (yte <= hi)).double().mean())

        crps = _crps_from_samples(samp, yte)
        sd_ratio = float((samp.std(dim=1) / true_sd_test.squeeze(-1)).mean())
        nll = float((-mech.log_prob(y_test, pa_test)).mean())
        return dict(ks=ks, cov50=coverage(0.50), cov90=coverage(0.90),
                    crps=crps, sd_ratio=sd_ratio, nll=nll)


# ───────────────────────────────── default panel ───────────────────────────────

def default_mechanisms() -> dict[str, tuple[Callable, dict]]:
    """The three continuous non-parametric mechanisms, default hyper-parameters.

    (factory, fit_kwargs). Imported lazily so the module loads even if optional
    deps are missing.
    """
    from nbn.mechanisms.non_parametric.conditional_kde import ConditionalKDEMechanism
    from nbn.mechanisms.non_parametric.knn_conditional import KNNConditionalMechanism
    from nbn.mechanisms.non_parametric.flexcode import FlexCodeMechanism

    return {
        "ConditionalKDE": (lambda: ConditionalKDEMechanism(), {}),
        "KNNConditional": (lambda: KNNConditionalMechanism(), {}),
        "FlexCode": (lambda: FlexCodeMechanism(epochs=120, n_basis=31), {}),
    }


def run_panel(
    mechanisms: dict[str, tuple[Callable, dict]] | None = None,
    scms: dict[str, Callable] | None = None,
    n_train: int = 2500,
    n_test: int = 500,
    n_samples: int = 400,
    seed: int = 0,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run every (SCM, mechanism) pair and print a table. Returns nested results."""
    mechanisms = mechanisms or default_mechanisms()
    scms = scms or DEFAULT_SCMS
    results: dict[str, dict[str, dict[str, float]]] = {}

    header = (f"{'mechanism':<16}{'PIT-KS':>8}{'cov50':>7}{'cov90':>7}"
              f"{'CRPS':>8}{'sd_ratio':>9}{'NLL':>8}")
    for scm_name, scm in scms.items():
        gen = torch.Generator().manual_seed(seed)
        y_tr, pa_tr, _ = scm(n_train, gen)
        y_te, pa_te, sd_te = scm(n_test, gen)
        print(f"\n=== SCM: {scm_name}   "
              f"(targets: cov50=0.50, cov90=0.90, sd_ratio=1.0, PIT-KS->0) ===")
        print(header)
        results[scm_name] = {}
        for name, (factory, fit_kw) in mechanisms.items():
            r = assess_calibration(factory(), y_tr, pa_tr, y_te, pa_te, sd_te,
                                   n_samples=n_samples, fit_kwargs=fit_kw)
            results[scm_name][name] = r
            print(f"{name:<16}{r['ks']:>8.3f}{r['cov50']:>7.2f}{r['cov90']:>7.2f}"
                  f"{r['crps']:>8.3f}{r['sd_ratio']:>9.2f}{r['nll']:>8.3f}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Calibration diagnostics for mechanisms.")
    p.add_argument("--n-train", type=int, default=2500)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    run_panel(n_train=args.n_train, n_test=args.n_test,
              n_samples=args.n_samples, seed=args.seed)


if __name__ == "__main__":
    main()
