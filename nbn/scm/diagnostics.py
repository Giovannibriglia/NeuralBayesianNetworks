"""Diagnostics for the assumptions behind a structural causal model.

Each test returns a :class:`DiagnosticResult` with a p-value under a stated
null hypothesis.  A small p-value is evidence *against* the null; a large one
is not proof of it (low power is always possible).

================================  ============================================
Test                              Null hypothesis
================================  ============================================
:func:`noise_sharing_test`        one noise law for every parent configuration
:func:`noise_tying_test`          one noise law for several nodes
:func:`freshness_test`            the noise at ``t`` and ``t + 1`` is independent
:func:`residual_dependence_test`  the noises of two nodes are independent
:func:`context_invariance_test`   ``C ⊥ X | pa(X)`` for a dataset/policy id ``C``
:func:`noise_independence_test`   the abducted noise is independent of the parents
:func:`calibration_by_bins_test`  the fitted CDF is calibrated in every parent region
================================  ============================================

Discrete tests use a parametric bootstrap: every replicate is refitted on a
simulated count table, all replicates at once (the offset fit is batched).
Continuous tests use asymptotic laws.  Combine many tests with
:func:`nbn.scm.benjamini_hochberg`.  Data are moved to the model's device.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Literal, Mapping, Sequence

import numpy as np
import torch

from nbn.scm._stats import chi2_sf, ks_two_sample, ks_uniform, normal_two_sided_p, require_cpu_generator
from nbn.scm.fitting import child_law, fit_offset_law, log_likelihood, offset_transfer, rows_table
from nbn.scm.mechanisms import (
    AdditiveNoiseMechanism,
    DiscreteAdditiveNoiseMechanism,
    StructuralMechanism,
)
from nbn.uncertainty.counts import count_table, parent_row_index, scatter_counts
from nbn.utils.device import to_device, to_device_dtype


@dataclass(frozen=True)
class DiagnosticResult:
    """Outcome of one diagnostic test."""

    name: str
    statistic: float
    p_value: float
    null: str
    details: Dict[str, object] = field(default_factory=dict)

    def rejected(self, alpha: float = 0.05) -> bool:
        """``p <= alpha``: a valid p-value has ``P(p <= alpha) <= alpha`` under the null."""
        return self.p_value <= alpha


# ===========================================================================
# Noise sharing (within a node) and tying (across nodes)
# ===========================================================================


@torch.no_grad()
def noise_sharing_test(
    model, data: Mapping[str, torch.Tensor], node: str, n_boot: int = 200
) -> DiagnosticResult:
    """Is one noise law enough for every parent configuration of ``node``?

    Discrete (``DiscreteAdditiveNoiseMechanism``): likelihood ratio of a free
    table per parent configuration against the shared-noise table, with a
    parametric-bootstrap p-value.

    Continuous (``AdditiveNoiseMechanism``): a White-type test (Koenker's
    studentised Breusch–Pagan with squared terms) — regress the squared
    residuals on the parents and their squares; ``n R^2`` is asymptotically
    chi-square with as many degrees of freedom as independent regressors
    (``2 d_pa`` at most; fewer for binary parents, whose squares repeat them).  It detects a noise *scale*
    that varies with the parents, also symmetrically (e.g. with ``|pa|``).
    """
    data = _on_model_device(model, data)
    mech = _structural(model, node)
    if isinstance(mech, DiscreteAdditiveNoiseMechanism):
        counts, _ = count_table(model, data, node)
        transfer = mech.transfer()

        def shared_fit(c: torch.Tensor, start=None):
            """``(G statistic, fitted table, fit)`` of the shared-noise model."""
            (loc,), q, _ = fit_offset_law([transfer], [c], start=start)
            table = rows_table(child_law(transfer, q), loc)
            return _g_against_free_table(c, table), table, ([loc], q)

        observed, null_table, null_fit = shared_fit(counts)
        # Replicates come from the null fit, so it is the right basin to start from.
        replicates, _, _ = shared_fit(_simulate_counts(null_table, counts, n_boot), start=null_fit)
        return DiagnosticResult(
            "noise_sharing", float(observed), _bootstrap_p(observed, replicates),
            null=f"one offset law for every parent configuration of '{node}'",
            details={"method": "parametric bootstrap", "n_boot": n_boot,
                     "null_mean_statistic": float(replicates.mean())},
        )
    if isinstance(mech, AdditiveNoiseMechanism):
        pa = model.parent_values(data, node)
        if pa is None:
            raise ValueError(f"'{node}' has no parents: nothing to share across.")
        u2 = mech.residual(data[node], pa).pow(2).sum(-1)
        regressors = torch.cat([pa, pa.pow(2)], dim=-1).to(u2)
        statistic_value, df = _n_r_squared(u2, regressors)  # df = rank (binary parents: pa**2 == pa)
        return DiagnosticResult(
            "noise_sharing", statistic_value, float(chi2_sf(statistic_value, df)),
            null=f"the noise scale of '{node}' does not depend on its parents",
            details={"method": "White-type n R^2", "df": df},
        )
    raise TypeError(f"No sharing test for {type(mech).__name__}.")


@torch.no_grad()
def noise_tying_test(
    model, data: Mapping[str, torch.Tensor], nodes: Sequence[str], n_boot: int = 200
) -> DiagnosticResult:
    """Can ``nodes`` share one noise law?

    Discrete: likelihood ratio of one law per node against one shared law,
    parametric-bootstrap p-value.  Continuous: two-sample Kolmogorov–Smirnov
    tests between the residuals of every pair of nodes, Bonferroni-combined.
    """
    data = _on_model_device(model, data)
    mechs = [_structural(model, n) for n in nodes]
    if all(isinstance(m, DiscreteAdditiveNoiseMechanism) for m in mechs):
        # A shared law needs one offset range: the widest of the members (as tie_noise does).
        widest = max(m.noise_max_offset for m in mechs if isinstance(m, DiscreteAdditiveNoiseMechanism))
        offsets = torch.arange(-widest, widest + 1, device=model.device)
        transfers = [
            offset_transfer(m.n_classes, offsets) for m in mechs
            if isinstance(m, DiscreteAdditiveNoiseMechanism)
        ]
        counts = [count_table(model, data, n)[0] for n in nodes]
        observed, null_tables, fits = _tied_vs_separate_lr(transfers, counts)
        simulated = [_simulate_counts(t, c, n_boot) for t, c in zip(null_tables, counts)]
        replicates, _, _ = _tied_vs_separate_lr(transfers, simulated, starts=fits)
        return DiagnosticResult(
            "noise_tying", float(observed), _bootstrap_p(observed, replicates),
            null=f"one offset law shared by {list(nodes)}",
            details={"method": "parametric bootstrap", "n_boot": n_boot,
                     "null_mean_statistic": float(replicates.mean())},
        )
    if all(isinstance(m, AdditiveNoiseMechanism) for m in mechs):
        groups = [
            m.residual(data[n], model.parent_values(data, n))[:, 0]
            for n, m in zip(nodes, mechs)
            if isinstance(m, AdditiveNoiseMechanism)
        ]
        statistic, p_value, n_pairs = _pairwise_ks(groups)
        return DiagnosticResult(
            "noise_tying", statistic, p_value, null=f"one noise law shared by {list(nodes)}",
            details={"method": "pairwise KS, Bonferroni", "pairs": n_pairs},
        )
    raise TypeError("noise_tying_test needs nodes of one structural family.")


# ===========================================================================
# Freshness and cross-node dependence of the noise
# ===========================================================================


@torch.no_grad()
def residuals(model, data: Mapping[str, torch.Tensor], node: str) -> torch.Tensor:
    """One scalar noise value per row, ``[N]``.

    Continuous nodes: the exact residual ``x - g(pa)`` (first dimension).
    Discrete nodes: one draw from the posterior of the offset.  Under the
    model, these draws have exactly the law of the true noise, so tests on
    them keep their null distribution (a posterior *mean* would not).
    """
    data = _on_model_device(model, data)
    mech = _structural(model, node)
    return mech.abduct(data[node], model.parent_values(data, node), 1)[0, :, 0]


def freshness_test(
    residual: torch.Tensor, episode: torch.Tensor, time: torch.Tensor
) -> DiagnosticResult:
    """Is the noise drawn afresh at every step?

    Pairs the residual of each step with the next step of the same episode and
    tests zero correlation (Fisher z).  A persistent noise or confounder shows
    up as lag-1 correlation.  Linear dependence only.
    """
    # Move, then cast: MPS has no float64 (see nbn.utils.device.to_device_dtype).
    r, e, t = (to_device_dtype(v.reshape(-1), "cpu", torch.float64) for v in (residual, episode, time))
    order = torch.argsort(e * (t.max() + 1) + t)
    r, e, t = r[order], e[order], t[order]
    consecutive = (e[1:] == e[:-1]) & (t[1:] == t[:-1] + 1)
    rho, p_value = _pearson(r[:-1][consecutive], r[1:][consecutive])
    return DiagnosticResult(
        "freshness", rho, p_value, null="residuals at t and t+1 are uncorrelated",
        details={"pairs": int(consecutive.sum())},
    )


def residual_dependence_test(
    a: torch.Tensor,
    b: torch.Tensor,
    method: Literal["pearson", "hsic"] = "pearson",
    n_permutations: int = 200,
) -> DiagnosticResult:
    """Are two residual series independent?  Dependence suggests a hidden
    common cause (a bidirected edge) or a missing edge.

    ``"pearson"``: zero correlation, Fisher z (linear dependence only).
    ``"hsic"``: Hilbert–Schmidt independence criterion with Gaussian kernels
    (median bandwidth) and a permutation p-value; detects nonlinear
    dependence.  Runs on the residuals' device; memory ``O(N^2)``.
    """
    if method == "pearson":
        rho, p_value = _pearson(a.reshape(-1).cpu().double(), b.reshape(-1).cpu().double())
        return DiagnosticResult("residual_dependence", rho, p_value,
                                null="the two residuals are uncorrelated",
                                details={"method": "pearson"})
    if method == "hsic":
        statistic, p_value = _hsic_permutation(a, b, n_permutations)
        return DiagnosticResult("residual_dependence", statistic, p_value,
                                null="the two residuals are independent",
                                details={"method": "hsic", "permutations": n_permutations})
    raise ValueError(f"Unknown method '{method}'.")


@torch.no_grad()
def noise_independence_test(
    model,
    data: Mapping[str, torch.Tensor],
    node: str,
    n_permutations: int = 200,
    max_rows: int = 2000,
    generator: torch.Generator | None = None,
) -> DiagnosticResult:
    """Is the abducted noise of ``node`` independent of its parents?

    The structural form ``X := f(pa, U)`` assumes ``U ⊥ pa``.  This test
    abducts the noise of every row and runs an HSIC permutation test against
    the parents (the residual-independence check of additive-noise model
    fitting, Peters, Janzing & Schölkopf, *Elements of Causal Inference*,
    §4.2.1).  Use data that were **not** used to fit the mechanism: fitting
    and testing on the same rows can hide or create dependence.  A rejection
    shows the per-parent reading of a :class:`PITBandSet` is unsafe; a pass
    only fails to detect a violation (it does not prove independence).  At
    most ``max_rows`` rows are used (HSIC memory is quadratic).  ``generator``
    (a CPU generator) makes the subsample and the permutations reproducible.
    """
    data = _on_model_device(model, data)
    mech = _structural(model, node)
    parents = model.parent_values(data, node)
    if parents is None:
        raise ValueError(f"'{node}' has no parents: nothing to be independent of.")
    noise = mech.abduct(data[node], parents, 1)[0]
    n = noise.shape[0]
    require_cpu_generator(generator, "noise_independence_test")
    if n > max_rows:
        keep = torch.randperm(n, generator=generator, device="cpu")[:max_rows].to(noise.device)
        noise, parents = noise[keep], parents[keep]
    statistic, p_value = _hsic_permutation(noise, parents, n_permutations, generator)
    return DiagnosticResult(
        "noise_independence", statistic, p_value,
        null=f"the noise of '{node}' is independent of its parents",
        details={"method": "hsic", "permutations": n_permutations, "rows": int(noise.shape[0])},
    )


@torch.no_grad()
def calibration_by_bins_test(
    model, data: Mapping[str, torch.Tensor], node: str, n_bins: int = 4,
    reference: Literal["uniform", "other_bins"] = "uniform",
    generator: torch.Generator | None = None,
) -> DiagnosticResult:
    """Is the fitted CDF calibrated in every region of the inputs?

    Sorts held-out rows by the predicted median ``F_hat^{-1}(1/2 | pa)``
    (continuous children) or mean class index (discrete children),
    splits them into ``n_bins`` equal-count bins and looks at the probability
    integral transform ``W = F_hat(x | pa)`` in each bin:

    * ``reference="uniform"`` — each bin against Uniform(0, 1) (one-sample KS,
      Bonferroni over bins): is the raw model calibrated in every bin?  This
      is the per-bin comparison of model and data of Kochenderfer et al.,
      *Algorithms for Validation*, §2.5.3, pp. 48–49.
    * ``reference="other_bins"`` — bins against each other (pairwise two-sample
      KS, Bonferroni): does ``W`` have the same law everywhere, uniform or not?
      This is the assumption that lets :func:`nbn.scm.pit_band_set` (which
      recalibrates a uniform miscalibration) bound every conditional
      probability.

    A pooled check can hide a model that is right on average and wrong in one
    region; per-bin checks are the remedy.  Binning on the predicted median is
    a 1-D projection of the parents: with several parents it only sees
    miscalibration that varies along that projection.  A pass does not prove
    calibration.  Use rows not used for fitting.

    Discrete children (structural categorical mechanisms) use the randomized
    PIT (drawn with the CPU ``generator``) and bins of the predicted mean class
    index.
    """
    from nbn.scm.calibration import _calibration_rows, _mechanism, pit_values

    if reference not in ("uniform", "other_bins"):
        raise ValueError(f"reference must be 'uniform' or 'other_bins', got {reference!r}.")
    mech = _mechanism(model, node, need=("cdf",))
    x, parents = _calibration_rows(model, data, node, None, None)
    if parents is None:
        raise ValueError(f"'{node}' has no parents: there is nothing to bin.")
    if not 2 <= n_bins <= x.shape[0]:
        raise ValueError(f"n_bins must lie in [2, {x.shape[0]}] (number of rows), got {n_bins}.")
    pit = pit_values(model, data, node, generator=generator)
    if getattr(mech, "is_discrete", False):  # predicted mean class index
        probs = mech.probs(parents, x.shape[0])
        key = (probs * torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)).sum(-1)
    else:
        key = mech.quantile(0.5, parents)
    order = torch.argsort(key)
    bins = [pit[chunk] for chunk in torch.tensor_split(order, n_bins)]
    if reference == "uniform":
        results = [ks_uniform(b) for b in bins]
        statistic, p_min = min(results, key=lambda r: r[1])
        p_value, method = min(1.0, p_min * n_bins), "KS vs Uniform(0, 1) per bin, Bonferroni"
        null = f"the fitted CDF of '{node}' is calibrated in every bin of the predicted median"
    else:
        statistic, p_value, _ = _pairwise_ks(bins)
        method = "pairwise two-sample KS, Bonferroni"
        null = f"the PIT of '{node}' has the same law in every bin of the predicted median"
    return DiagnosticResult(
        "calibration_by_bins", statistic, p_value, null=null,
        details={"method": method, "bins": n_bins, "rows_per_bin": int(bins[-1].numel())},
    )


# ===========================================================================
# Context (regime / policy) invariance
# ===========================================================================


@torch.no_grad()
def context_invariance_test(
    model, data: Mapping[str, torch.Tensor], node: str, context: str | None = None,
    reference: Literal["permutation", "chi2"] = "permutation", n_perm: int = 999,
    generator: torch.Generator | None = None,
) -> DiagnosticResult:
    """Does ``node``'s mechanism change across contexts (datasets, policies)?

    Null: ``C ⊥ X | pa(X)``.  With policy id as ``C`` and ``X`` a next-state
    variable, a rejection means some logger used information the model does
    not see (a confounded action) or the dynamics drifted.

    Discrete ``X`` with discrete parents: the G statistic of ``X`` vs ``C``
    summed over the parent configurations, with one of two references:

    ``reference="permutation"`` (default)
        Under the null the context labels are exchangeable within each parent
        configuration, so each configuration's context-by-value table is
        redrawn ``n_perm`` times with its margins fixed (multivariate
        hypergeometric) and ``p = (1 + #{G_perm >= G_obs}) / (n_perm + 1)``.
        For independent rows the level is valid at any sample size
        (``P(p <= alpha) <= alpha``; conservative when the statistic has
        ties).  For trajectories (the parents making the node Markov) later
        strata depend on earlier outcomes, so the level is approximate; in
        simulations it stays close to nominal.  The smallest p-value is
        ``1 / (n_perm + 1)``: to reject at level ``alpha`` (after a
        Bonferroni or Benjamini-Hochberg correction over ``m`` tests,
        ``alpha / m``) use ``n_perm >= 1 / alpha``.  ``generator`` (CPU) makes
        it reproducible; without one the p-value varies from run to run.
        Counts of ``10**9`` or more rows for one value within a configuration
        exceed numpy's hypergeometric sampler.
    ``reference="chi2"``
        The chi-square approximation with the summed degrees of freedom: fast,
        but anti-conservative when many configurations are sparse: with 64
        configurations, two contexts and 2000 rows it rejects a true null 21%
        of the time at level 0.05 (about 80% with 200 rows; user guide, SCM
        diagnostics; notebook 10, section 1b).

    Continuous ``X``: KS tests of the residuals between every pair of
    contexts, Bonferroni-combined (``reference`` does not apply).
    """
    if reference not in ("permutation", "chi2"):
        raise ValueError(f"reference must be 'permutation' or 'chi2'; got {reference!r}.")
    if isinstance(n_perm, bool) or not isinstance(n_perm, int) or n_perm < 1:
        raise ValueError(f"n_perm must be an integer >= 1; got {n_perm!r}.")
    context = context or getattr(model, "context", None)
    if context is None:
        raise ValueError("No context variable given (and the model declares none).")
    data = _on_model_device(model, data)
    c = data[context].reshape(-1).long()
    n_ctx = int(c.max()) + 1
    variable = model.variables[node]
    if variable.is_discrete:
        parents = model.dag.parents(node)
        cards = [int(model.variables[p].cardinality) for p in parents]
        pa = (torch.stack([data[p].reshape(-1) for p in parents], -1) if parents
              else torch.zeros(c.shape[0], 0, device=c.device))
        rows = parent_row_index(pa, cards) * n_ctx + c
        n_rows = n_ctx
        for card in cards:
            n_rows *= card
        k = int(variable.cardinality)
        table = scatter_counts(rows, data[node], n_rows, k).reshape(-1, n_ctx, k)
        statistic, df = _stratified_g(table)
        null = f"'{context}' ⊥ '{node}' | parents"
        if reference == "chi2":
            return DiagnosticResult(
                "context_invariance", statistic, float(chi2_sf(statistic, df)),
                null=null, details={"method": "G-test, chi-square reference", "df": df},
            )
        require_cpu_generator(generator, "context_invariance_test")
        p_value, g_obs = _stratified_g_permutation(table, n_perm, generator)
        return DiagnosticResult(
            "context_invariance", g_obs, p_value, null=null,
            details={"method": "G-test, permutation reference", "n_perm": n_perm, "df": df},
        )
    mech = _structural(model, node)
    if not isinstance(mech, AdditiveNoiseMechanism):
        raise TypeError(f"No continuous context test for {type(mech).__name__}.")
    res = mech.residual(data[node], model.parent_values(data, node))[:, 0]
    statistic, p_value, n_pairs = _pairwise_ks([res[c == g] for g in torch.unique(c)])
    return DiagnosticResult(
        "context_invariance", statistic, p_value,
        null=f"the residual law of '{node}' is the same in every '{context}'",
        details={"method": "pairwise KS, Bonferroni", "pairs": n_pairs},
    )


# ===========================================================================
# Helpers
# ===========================================================================


def _on_model_device(model, data: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return to_device(dict(data), model.device)


def _structural(model, node: str) -> StructuralMechanism:
    mech = model.mechanisms[node]
    if not isinstance(mech, StructuralMechanism):
        raise TypeError(f"'{node}' has no structural mechanism ({type(mech).__name__}).")
    return mech


def _model_loglik(counts: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """``sum N log P`` with ``P`` floored at the dtype's ``tiny``, so an impossible
    cell costs a huge (finite) penalty instead of producing ``inf``/``nan``."""
    table = table.to(counts)
    return log_likelihood(counts, table.clamp_min(torch.finfo(table.dtype).tiny))


def _g_against_free_table(counts: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """``2 sum N log(p_free / p_model)``, with ``p_free`` the per-row frequencies; ``[...]``."""
    free = counts / counts.sum(-1, keepdim=True).clamp_min(1.0)
    return 2.0 * (log_likelihood(counts, free) - _model_loglik(counts, table))


def _tied_vs_separate_lr(transfers, counts, starts=None):
    """``2 (logL_separate - logL_tied)``, the tied tables and the fits; batched over ``[...]``.

    ``starts`` (the fits of an earlier call) warm-start every fit, as the
    bootstrap does for its replicates.
    """
    tied_fit = fit_offset_law(transfers, counts, start=None if starts is None else starts[0])
    locations, q, _ = tied_fit
    tied_tables = [rows_table(child_law(a, q), loc) for a, loc in zip(transfers, locations)]
    tied = sum(_model_loglik(c, t) for c, t in zip(counts, tied_tables))
    separate, separate_fits = 0.0, []
    for i, (a, c) in enumerate(zip(transfers, counts)):
        fit = fit_offset_law([a], [c], start=None if starts is None else starts[1][i])
        (loc,), q_alone, _ = fit
        separate = separate + _model_loglik(c, rows_table(child_law(a, q_alone), loc))
        separate_fits.append(([loc], q_alone))
    fits = ((locations, q), separate_fits)
    return 2.0 * (separate - tied), tied_tables, fits


def _simulate_counts(table: torch.Tensor, counts: torch.Tensor, n_boot: int) -> torch.Tensor:
    """``[n_boot, R, K]`` multinomial count tables with the observed row totals.

    Drawn as a chain of binomials per value (exact multinomial sampling), so
    memory is ``O(n_boot R K)`` whatever the number of data rows.  Sampled on
    the CPU (``torch.binomial`` is not available on every accelerator).
    """
    probs = table.detach().cpu().double()  # move first: MPS has no float64
    remaining = counts.sum(-1).cpu().double().expand(n_boot, -1).clone()
    mass_left = torch.ones(probs.shape[0], dtype=torch.float64)
    out = torch.zeros(n_boot, *probs.shape, dtype=torch.float64)
    for k in range(probs.shape[1] - 1):
        p = torch.where(mass_left > 0, probs[:, k] / mass_left.clamp_min(1e-300), 0.0)
        draw = torch.binomial(remaining, p.clamp(0.0, 1.0).expand_as(remaining))
        out[..., k] = draw
        remaining -= draw
        mass_left = mass_left - probs[:, k]
    out[..., -1] = remaining
    return out.to(counts.dtype).to(counts.device)  # cast on the CPU: MPS has no float64


def _bootstrap_p(observed: torch.Tensor, replicates: torch.Tensor) -> float:
    """``(1 + #{T_b >= T}) / (1 + B)``."""
    exceed = int((replicates >= observed - 1e-9 * observed.abs().clamp_min(1.0)).sum())
    return (1 + exceed) / (1 + replicates.numel())


def _pairwise_ks(groups) -> tuple[float, float, int]:
    """Largest-evidence two-sample KS over all pairs, Bonferroni-corrected p-value."""
    pairs = list(itertools.combinations(range(len(groups)), 2))
    if not pairs:
        raise ValueError("Need at least two groups to compare.")
    results = [ks_two_sample(groups[i], groups[j]) for i, j in pairs]
    statistic, p_value = min(results, key=lambda r: r[1])
    return statistic, min(1.0, p_value * len(pairs)), len(pairs)


def _n_r_squared(y: torch.Tensor, x: torch.Tensor) -> tuple[float, int]:
    """``n R^2`` of the least-squares regression of ``y`` on ``[1, x]``."""
    design = torch.cat([torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype), x], 1)
    design, y = design.cpu().double(), y.cpu().double()
    coef = torch.linalg.lstsq(design, y.unsqueeze(-1)).solution
    fitted = (design @ coef).squeeze(-1)
    ss_res = (y - fitted).pow(2).sum()
    ss_tot = (y - y.mean()).pow(2).sum().clamp_min(1e-30)
    df = int(torch.linalg.matrix_rank(design)) - 1  # duplicated columns do not count
    return float(y.numel() * (1.0 - ss_res / ss_tot)), max(df, 1)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    n = a.numel()
    if n < 4:
        raise ValueError("Need at least 4 pairs for a correlation test.")
    rho = torch.corrcoef(torch.stack([a, b]))[0, 1].clamp(-0.999999, 0.999999)
    z = torch.atanh(rho) * (n - 3) ** 0.5
    return float(rho), float(normal_two_sided_p(z))


def _hsic_permutation(a: torch.Tensor, b: torch.Tensor, n_permutations: int,
                      generator: torch.Generator | None = None) -> tuple[float, float]:
    a = a.reshape(a.shape[0], -1).to(torch.float32)
    b = to_device_dtype(b.reshape(b.shape[0], -1), a.device, torch.float32)
    n = a.shape[0]

    def centred_gram(z: torch.Tensor) -> torch.Tensor:
        sq = torch.cdist(z, z).pow(2)
        positive = sq[sq > 0]
        bandwidth = positive.median() if positive.numel() else sq.new_tensor(1.0)
        gram = torch.exp(-sq / bandwidth.clamp_min(1e-12))
        return gram - gram.mean(0, keepdim=True) - gram.mean(1, keepdim=True) + gram.mean()

    k_a, k_b = centred_gram(a), centred_gram(b)
    observed = float((k_a * k_b).sum() / n**2)
    exceed = 0
    for _ in range(n_permutations):
        perm = torch.randperm(n, generator=generator, device="cpu").to(a.device)  # CPU generator: reproducible
        exceed += int((k_a * k_b[perm][:, perm]).sum() / n**2 >= observed)
    return observed, (1 + exceed) / (1 + n_permutations)


def _g_numpy(tables: np.ndarray) -> np.ndarray:
    """G statistic of independence of ``[..., C, K]`` contingency tables (float64)."""
    total = tables.sum((-2, -1), keepdims=True)
    expected = tables.sum(-1, keepdims=True) * tables.sum(-2, keepdims=True) / np.maximum(total, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * np.where(tables > 0, tables * np.log(tables / expected), 0.0).sum((-2, -1))


def _stratified_g_permutation(table: torch.Tensor, n_perm: int, generator: torch.Generator | None,
                              max_elements: int = 1 << 22) -> tuple[float, float]:
    """Permutation p-value of the stratified G statistic, and the observed statistic.

    Each stratum's ``[C, K]`` table is redrawn with its row and column sums
    fixed (the context labels permuted within the stratum): one context at a
    time and one value at a time, by sequential hypergeometric draws,
    vectorised over a block of replicates and a block of strata, with at most
    about ``max_elements`` simulated cells per block (plus temporaries a few
    times that size).  Strata with fewer than two contexts or two values have
    a fixed table and are skipped; contexts and values empty in every kept
    stratum are dropped (sparse context ids cost nothing).  The counts are
    copied to the CPU with ``to_device_dtype``: a fused MPS -> CPU float64
    move garbles them.
    """
    t_all = to_device_dtype(table.detach(), "cpu", torch.float64).round().to(torch.int64).numpy()
    informative = ((t_all.sum(2) > 0).sum(1) >= 2) & ((t_all.sum(1) > 0).sum(1) >= 2)
    t_all = t_all[informative]
    t_all = t_all[:, t_all.sum((0, 2)) > 0][:, :, t_all.sum((0, 1)) > 0]
    seed = int(torch.randint(0, 2**62, (1,), generator=generator, device="cpu").item())
    rng = np.random.default_rng(seed)
    n_ctx, k = t_all.shape[1:]
    cells = max(1, n_ctx * k)
    rep_block = max(1, min(n_perm, max_elements // cells))
    strata_block = max(1, max_elements // (rep_block * cells))
    g_obs = float(_g_numpy(t_all.astype(np.float64)).sum()) if t_all.shape[0] else 0.0
    g_sim = np.zeros(n_perm)
    for lo in range(0, t_all.shape[0], strata_block):
        t = t_all[lo:lo + strata_block]                               # [S, C, K]
        for r0 in range(0, n_perm, rep_block):
            n_rep = min(rep_block, n_perm - r0)
            sim = np.empty((n_rep,) + t.shape)
            remaining = np.broadcast_to(t.sum(1), (n_rep,) + t.shape[::2]).copy()    # [P, S, K]
            for r in range(n_ctx):
                left = np.broadcast_to(t[:, r].sum(1), (n_rep, t.shape[0])).copy()   # [P, S]
                for c in range(k):
                    draw = left if c == k - 1 else rng.hypergeometric(
                        remaining[..., c], remaining[..., c + 1:].sum(-1), left)
                    sim[:, :, r, c] = draw
                    remaining[..., c] -= draw
                    left = left - draw
            g_sim[r0:r0 + n_rep] += _g_numpy(sim).sum(1)
    exceed = np.sum(g_sim >= g_obs - 1e-9 * max(1.0, g_obs))      # ties count as exceedances
    return float((1 + exceed) / (n_perm + 1)), g_obs


def _stratified_g(table: torch.Tensor) -> tuple[float, int]:
    """G statistic of ``C ⊥ X`` summed over strata ``[R, C, K]``, and its df (vectorised)."""
    table = table.to(torch.float64) if table.device.type != "mps" else table
    row = table.sum(-1, keepdim=True)            # [R, C, 1]
    col = table.sum(-2, keepdim=True)            # [R, 1, K]
    total = table.sum((-2, -1), keepdim=True)    # [R, 1, 1]
    expected = row * col / total.clamp_min(1.0)
    statistic = 2.0 * torch.xlogy(table, table / expected.clamp_min(1e-30)).sum()
    contexts = (row.squeeze(-1) > 0).sum(-1)     # non-empty contexts per stratum
    values = (col.squeeze(-2) > 0).sum(-1)       # observed values per stratum
    df = int(((contexts - 1).clamp_min(0) * (values - 1).clamp_min(0)).sum())
    return float(statistic), max(df, 1)
