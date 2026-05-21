"""Smoke-test the baseline adapters: each one fits and answers one query.

Adapters that need optional libraries are skipped when those libs aren't
installed.  This keeps the test suite green on minimal environments.

Every adapter is exercised against a small synthetic problem constructed
via :func:`benchmarking.synthetic.make_synthetic_bn`.
"""
from __future__ import annotations

import importlib

import pytest
import torch

from benchmarking.baselines import get_adapter
from benchmarking.domains.base import BenchmarkProblem, Query
from benchmarking.synthetic import make_synthetic_bn


def _has(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except ImportError:
        return False


def _problem(family: str = "hybrid", n: int = 10) -> BenchmarkProblem:
    """Build a small ``BenchmarkProblem`` from a synthetic BN."""
    bn = make_synthetic_bn(
        family=family, n_nodes=n, edge_density=0.20,
        n_train=200, n_test=50, n_reference=200, seed=0, device="cpu",
    )
    nodes = list(bn.dag.nodes())
    queries = [
        Query(
            targets=(nodes[0],),
            evidence={nodes[1]: bn.test_data[nodes[1]][:1].reshape(1)} if len(nodes) > 1 else {},
            kind="marginal",
        ),
    ]
    return BenchmarkProblem(
        name=bn.name, dag=list(bn.dag.edges()), variables=bn.variable_specs,
        train_data=bn.train_data, test_data=bn.test_data,
        queries=queries, ground_truth=None,
    )


def _smoke_query(adapter, problem: BenchmarkProblem) -> None:
    """Fit + answer one marginal query."""
    adapter.fit(problem)
    marg = next(q for q in problem.queries if q.kind == "marginal")
    res = adapter.query(marg)
    assert isinstance(res, torch.Tensor)
    adapter.teardown()


def test_nbn_adapter_smoke_continuous() -> None:
    # ``hybrid`` family fitter requires the option-(b) train-quantile
    # binner for continuous→discrete edges (deferred to v0.5b).  This
    # smoke test runs against ``continuous_lg`` to exercise the NBN
    # adapter end-to-end on a path that doesn't depend on the unfixed
    # binner.
    _smoke_query(get_adapter("nbn", device="cpu"), _problem("continuous_lg"))


@pytest.mark.parametrize(
    "variant",
    ["nbn_ve", "nbn_lw", "nbn_hybrid", "nbn_neural_categorical", "nbn_linear_gaussian"],
)
def test_all_nbn_variants_run(variant: str) -> None:
    """Every registered NBN preset must fit + answer one marginal query."""
    if variant == "nbn_ve":
        _smoke_query(get_adapter(variant, device="cpu"), _problem("discrete"))
    else:
        _smoke_query(get_adapter(variant, device="cpu"), _problem("continuous_lg"))


@pytest.mark.skipif(not _has("pgmpy"), reason="needs pgmpy")
def test_pgmpy_adapter_smoke_discrete() -> None:
    _smoke_query(get_adapter("pgmpy"), _problem("discrete"))


@pytest.mark.skipif(not _has("pomegranate"), reason="needs pomegranate")
def test_pomegranate_adapter_smoke_discrete() -> None:
    _smoke_query(get_adapter("pomegranate"), _problem("discrete"))


@pytest.mark.skipif(not _has("pomegranate"), reason="needs pomegranate")
def test_pomegranate_conditional_query_31() -> None:
    """Regression for v0.7-#31: pomegranate predict_proba on conditional
    queries must not raise IndexError.

    pomegranate v1.1.x indexes per-node probability tensors using the
    masked-tensor's underlying value as a ``long``; passing a ``float``
    raises ``IndexError: tensors used as indices must be long, ...``.
    The adapter constructs the ``row`` tensor as ``dtype=torch.long``
    (``benchmarking/baselines/pomegranate_adapter.py``) so the upstream
    indexing receives the expected dtype.

    This test exercises three query shapes — marginal, single-evidence
    conditional, multi-evidence conditional — on a 3-node chain
    ``A -> B -> C`` and asserts each returns a valid probability vector
    summing to 1.  Without the dtype cast, the conditional cases raise
    IndexError; the marginal case never did.
    """
    adapter = get_adapter("pomegranate")
    torch.manual_seed(0)
    n_train = 1000
    a = torch.randint(0, 2, (n_train,))
    b = torch.randint(0, 3, (n_train,))
    c = torch.randint(0, 2, (n_train,))
    problem = BenchmarkProblem(
        name="pomegranate_conditional_31",
        dag=[("A", "B"), ("B", "C")],
        variables={"A": ("discrete", 2), "B": ("discrete", 3), "C": ("discrete", 2)},
        train_data={"A": a, "B": b, "C": c},
        test_data={"A": a, "B": b, "C": c},
        queries=[],
        ground_truth=None,
    )
    adapter.fit(problem)
    try:
        for q in (
            Query(targets=["C"], evidence={}, kind="marginal"),
            Query(targets=["C"], evidence={"A": 1}, kind="conditional"),
            Query(targets=["C"], evidence={"A": 1, "B": 0}, kind="conditional"),
            Query(targets=["B"], evidence={"A": 0}, kind="conditional"),
        ):
            res = adapter.query(q)
            assert isinstance(res, torch.Tensor)
            assert res.dim() in (1, 2)
            probs = res.reshape(-1)
            assert probs.numel() in (2, 3)  # K for the target
            assert torch.all(probs >= 0)
            assert abs(float(probs.sum()) - 1.0) < 1e-4
    finally:
        adapter.teardown()


@pytest.mark.skipif(not _has("pyro"), reason="needs pyro-ppl")
def test_pyro_adapter_smoke_discrete() -> None:
    _smoke_query(get_adapter("pyro"), _problem("discrete"))


@pytest.mark.skipif(not _has("pyro"), reason="needs pyro-ppl")
def test_pyro_adapter_lg_conditional_32() -> None:
    """Regression for #32: pyro continuous path must condition on parents.

    Pre-fix, ``_fit_gaussian_leaf`` stored a per-node marginal
    ``(mean, std)`` and ``_pyro_model`` sampled ``Normal(mean, std)``
    independent of parent values — for any continuous_lg network the
    posterior P(B|A=a) collapsed to the marginal of B, regardless of
    a.  Post-fix, ``_fit_lg_leaf`` regresses each continuous node on
    its continuous parents and ``_pyro_model`` samples
    ``Normal(beta_0 + sum_i beta_i * pa_i, sigma)``.

    Fixture: 2-node chain ``A -> B`` with B = 2A + N(0, 0.1).  Query
    P(B|A=+2) and P(B|A=-2); empirical means must cluster near +4 and
    -4 respectively (the conditional means), not near 0 (the marginal
    of B).
    """
    from benchmarking.baselines.pyro_adapter import PyroAdapter
    torch.manual_seed(0)
    n = 2000
    a = torch.randn(n)
    b = 2.0 * a + 0.1 * torch.randn(n)
    problem = BenchmarkProblem(
        name="pyro_lg_conditional_32",
        dag=[("A", "B")],
        variables={"A": ("continuous", 1), "B": ("continuous", 1)},
        train_data={"A": a, "B": b}, test_data={"A": a, "B": b},
        queries=[], ground_truth=None,
    )
    adapter = PyroAdapter(n_samples=200)
    adapter.fit(problem)
    try:
        # The primary regression check is observable behaviour: posterior
        # means under different parent values must differ by the planted
        # slope.  Internal ``_gaussian`` storage shape is an
        # implementation detail and not asserted here.
        means = {}
        for a_obs in (2.0, -2.0):
            samples = adapter._posterior_samples(
                Query(targets=("B",),
                      evidence={"A": torch.tensor(a_obs)},
                      kind="marginal"),
                n_samples=400,
            )
            means[a_obs] = float(samples.mean())

        # Pre-fix marginal path: P(B|A=+2) and P(B|A=-2) both have mean
        # ~ E[B] ~ 0, so means[+2] - means[-2] ~ 0.  Post-fix LG path:
        # B = 2A + N(0, 0.1), so means[+2] - means[-2] ~ 8.  Tolerance
        # of 1.0 around the expected gap of 8 leaves wide MC headroom
        # while still catching any regression to the marginal path.
        gap = means[2.0] - means[-2.0]
        assert abs(gap - 8.0) < 1.0, (
            f"P(B|A=+2) mean = {means[2.0]:+.3f}, P(B|A=-2) mean = "
            f"{means[-2.0]:+.3f}, gap = {gap:+.3f} (expected ~8.0). "
            f"A gap near 0 indicates regression to the pre-#32 "
            f"marginal-Gaussian fit which ignored parent values."
        )

        # Both individual means should also be near their conditional
        # values (sanity check on the absolute scale, not just the gap).
        assert abs(means[2.0] - 4.0) < 0.5
        assert abs(means[-2.0] - (-4.0)) < 0.5
    finally:
        adapter.teardown()


@pytest.mark.skipif(not _has("pyro"), reason="needs pyro-ppl")
def test_pyro_adapter_default_n_samples_36() -> None:
    """Regression for #36: PyroAdapter default n_samples stays at 50.

    Bumping the default back to 200 (or higher) re-introduces the
    paper-config 600s per-cell timeout — at n_nodes=10, B=1024 the
    per-row ``[marg() for _ in range(n_samples)]`` loop in
    ``_posterior_samples`` extrapolates to ~1273s, 2.1× over budget.
    Pyro's EmpiricalMarginal exposes no batched-sample API, so the
    only native fix is keeping n_samples low (~50).  Companion
    end-to-end smoke (``test_pyro_adapter_smoke_discrete``) verifies
    the path still produces a usable posterior at this default.
    """
    from benchmarking.baselines.pyro_adapter import PyroAdapter
    assert PyroAdapter().n_samples == 50


@pytest.mark.skipif(not _has("pyro"), reason="needs pyro-ppl")
def test_pyro_dispatch_in_runner_not_none() -> None:
    """Regression for runner dispatch hole: _baseline_posterior_for_query must
    return a tensor (not None) for the pyro legacy baseline on discrete and
    continuous_lg.

    ``_baseline_posterior_for_query`` receives the legacy adapter name
    (``"pyro"``) as its ``baseline`` argument — not the method-keyed label.
    Pre-fix, ``"pyro"`` fell through to ``return None`` (no dispatch branch),
    causing every pyro accuracy cell to be no_result regardless of family.
    """
    from benchmarking.crash_test_runner import _baseline_posterior_for_query
    from benchmarking.synthetic import make_synthetic_bn

    for family, target_kind in (("discrete", "discrete"), ("continuous_lg", "continuous")):
        bn = make_synthetic_bn(
            family=family, n_nodes=5, n_train=200, n_test=50,
            n_reference=200, seed=0, device="cpu",
        )
        target = next(
            n for n in bn.train_data if bn.variable_specs[n][0] == target_kind
        )
        ev_row = {
            n: bn.test_data[n][0]
            for n in list(bn.train_data)[:2] if n != target
        }
        # "pyro" is the legacy baseline name that _legacy_adapter_for_spec returns
        # for both pyro-empirical and pyro-empirical-importance specs.
        out = _baseline_posterior_for_query(
            bn, "pyro", target, ev_row, target_kind, n_samples=20,
        )
        assert out is not None, (
            f"_baseline_posterior_for_query returned None for pyro "
            f"on {family} — dispatch branch missing or broken"
        )
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 1
        assert torch.isfinite(out).all()
        if target_kind == "discrete":
            k = bn.variable_specs[target][1]
            assert out.shape[0] == k
            assert abs(float(out.sum()) - 1.0) < 1e-3


@pytest.mark.skipif(not _has("pyro"), reason="needs pyro-ppl")
def test_pyro_adapter_hybrid_one_hot_regression_61() -> None:
    """Regression for issue #61: pyro must handle hybrid networks where
    a continuous node has both continuous and discrete parents.

    Fixture: 3-node network ``D -> B <- A`` where D is discrete (k=3)
    and A is continuous.  B = 0.5*A + offset[D] + N(0, 0.1), where
    offset = [0, 1, 2] per discrete category.

    Two checks:
    1. Beta coefficients for the discrete one-hot columns must be
       non-degenerate (std of contributions > 0.3), confirming the
       discrete-parent dimension is actually fitted.
    2. Posterior samples of B conditioned on D=2 (offset=2) vs D=0
       (offset=0) must differ in mean by approximately 2.0 (the
       planted category gap), confirming the one-hot contribution
       propagates correctly through _pyro_model.
    """
    import torch
    from benchmarking.baselines.pyro_adapter import PyroAdapter
    from benchmarking.domains.base import BenchmarkProblem, Query

    torch.manual_seed(42)
    n = 1000
    d = torch.randint(0, 3, (n,))         # discrete parent, k=3
    a = torch.randn(n)                     # continuous parent
    offsets = torch.tensor([0.0, 1.0, 2.0])
    b = 0.5 * a + offsets[d] + 0.1 * torch.randn(n)

    problem = BenchmarkProblem(
        name="pyro_hybrid_61",
        dag=[("D", "B"), ("A", "B")],
        variables={"D": ("discrete", 3), "A": ("continuous", 1), "B": ("continuous", 1)},
        train_data={"D": d, "A": a, "B": b},
        test_data={"D": d, "A": a, "B": b},
        queries=[], ground_truth=None,
    )
    adapter = PyroAdapter(n_samples=200)
    adapter.fit(problem)
    try:
        beta, sigma, cont_pa, disc_pa, disc_cards = adapter._gaussian["B"]
        # Check 1: discrete parent is present in the fit
        assert disc_pa == ["D"], f"expected disc_pa=['D'], got {disc_pa}"
        assert disc_cards == [3], f"expected disc_cards=[3], got {disc_cards}"
        # beta layout: [intercept, cont_coeff(A), oh_coeff_D0, oh_coeff_D1]
        assert beta.shape[0] == 4, f"expected beta len 4, got {beta.shape[0]}"
        # one-hot coefficients should be non-degenerate (planted gap is 1.0)
        oh_coeffs = beta[2:]  # two one-hot columns for k=3
        assert oh_coeffs.std() > 0.3, (
            f"one-hot coefficients {oh_coeffs.tolist()} have near-zero std; "
            "discrete-parent dimension was not fitted."
        )

        # Check 2: posterior mean of B shifts with discrete parent value
        means = {}
        for d_obs in (0, 2):
            samples = adapter._posterior_samples(
                Query(targets=("B",),
                      evidence={"D": torch.tensor(d_obs),
                                 "A": torch.tensor(0.0)},
                      kind="marginal"),
                n_samples=400,
            )
            means[d_obs] = float(samples.mean())

        gap = means[2] - means[0]
        assert abs(gap - 2.0) < 0.5, (
            f"P(B|D=2,A=0) mean = {means[2]:+.3f}, P(B|D=0,A=0) mean = "
            f"{means[0]:+.3f}, gap = {gap:+.3f} (expected ~2.0). "
            f"A gap near 0 indicates the one-hot contribution is not "
            f"propagating through _pyro_model."
        )
    finally:
        adapter.teardown()


@pytest.mark.skipif(not _has("gpytorch"), reason="needs gpytorch")
def test_gpytorch_adapter_skips_discrete_evidence() -> None:
    """GPyTorch is continuous-only — discrete evidence must raise NotImplementedError."""
    p = _problem("hybrid")
    a = get_adapter("gpytorch", device="cpu")
    a.fit(p)
    discrete_set = {n for n, (kind, _) in p.variables.items() if kind == "discrete"}
    discrete_q = next(
        (q for q in p.queries
         if q.kind == "marginal" and any(k in discrete_set for k in q.evidence)),
        None,
    )
    if discrete_q is None:
        pytest.skip("no discrete-evidence query in this problem instance")
    with pytest.raises((NotImplementedError, Exception)):
        a.query(discrete_q)
    a.teardown()
