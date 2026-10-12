"""Contracts every benchmark adapter honours, parametrised over the adapters.

These used to be copy-pasted into each ``test_<library>_adapter.py`` /
``test_<library>_param_recovery.py``; one parametrised test per contract
keeps the adapters in lock-step (a new adapter is added to ``_ADAPTERS`` and
inherits every check).  Adapter-specific behaviour stays in the per-library
files.

* ``is_applicable`` delegates to the applicability registry (the registry
  itself is pinned by ``test_applicability.py``); an unknown name is never
  applicable.
* ``query`` before ``fit`` raises ``RuntimeError("... not fitted ...")``.
* ``fit`` accepts the runner's ``epochs=`` keyword, also when it ignores it.
* ``extract_learned_cpts`` is deterministic and uses the canonical layout:
  parents sorted by name, first parent slowest -- whatever order the adapter
  stores them in.  (Each library used to have its own layout test against its
  private storage; the canonical-row check below is the contract those
  guarded, stated once.)
"""
from __future__ import annotations

import pytest
import torch

from nbn.bench.adapters import NBNAdapter, PgmpyAdapter, PomegranateAdapter, PyroAdapter
from nbn.bench.core.applicability import is_applicable as registry_is_applicable
from nbn.bench.domains.base import BenchmarkProblem, Query

# ---- problems -----------------------------------------------------------------


def _problem(family: str, n: int = 300, seed: int = 0) -> BenchmarkProblem:
    g = torch.Generator().manual_seed(seed)
    if family == "discrete":
        dag = [("X0", "X1"), ("X1", "X2")]
        data = {k: torch.randint(0, 2, (n,), generator=g) for k in ("X0", "X1", "X2")}
        variables = dict.fromkeys(data, ("discrete", 2))
    elif family == "continuous_lg":
        dag = [("X0", "X1"), ("X1", "X2")]
        x0 = torch.randn(n, generator=g)
        x1 = 0.5 * x0 + 0.5 * torch.randn(n, generator=g)
        x2 = 0.5 * x1 + 0.5 * torch.randn(n, generator=g)
        data = {"X0": x0, "X1": x1, "X2": x2}
        variables = dict.fromkeys(data, ("continuous", None))
    elif family == "hybrid":
        dag = [("X0", "X1")]
        x0 = torch.randint(0, 2, (n,), generator=g)
        data = {"X0": x0, "X1": x0.float() + 0.5 * torch.randn(n, generator=g)}
        variables = {"X0": ("discrete", 2), "X1": ("continuous", None)}
    else:
        raise ValueError(family)
    return BenchmarkProblem(name=f"contract_{family}", dag=dag, variables=variables,
                            train_data=data, test_data=data, queries=[])


def _multiparent_problem(n: int = 4000) -> BenchmarkProblem:
    """X2 = (X0 + 2*X1) mod 3, a deterministic function of both parents.

    The DAG lists X1 before X0 (the reverse of the canonical name order), so
    an adapter that leaks its storage order into the canonical layout puts
    the mass in the wrong row.
    """
    g = torch.Generator().manual_seed(0)
    x0 = torch.randint(0, 2, (n, 1), generator=g)
    x1 = torch.randint(0, 2, (n, 1), generator=g)
    data = {"X0": x0, "X1": x1, "X2": (x0 + 2 * x1) % 3}
    variables = {"X0": ("discrete", 2), "X1": ("discrete", 2), "X2": ("discrete", 3)}
    return BenchmarkProblem(
        name="mp", dag=[("X1", "X2"), ("X0", "X2")], variables=variables,
        train_data=data, test_data=data, queries=[], true_model=None,
        family="discrete", problem_id="3", seed=0,
    )


# ---- adapters -----------------------------------------------------------------

_ADAPTERS = {
    "nbn-cat-lw": lambda: NBNAdapter(mechanism="cat", engine="lw", device="cpu"),
    "nbn-mdn-lw": lambda: NBNAdapter(mechanism="mdn", engine="lw", device="cpu"),
    "pgmpy-mle-ve": lambda: PgmpyAdapter(param_method="mle", inference_method="ve"),
    "pgmpy-lg-predict": lambda: PgmpyAdapter(param_method="lg", inference_method="predict"),
    "pomegranate": lambda: PomegranateAdapter(device="cpu"),
    "pyro": lambda: PyroAdapter(mechanism="empirical", inference_method="importance",
                                n_samples=20, device="cpu"),
}

#: Adapters that can fit the small discrete problem (for the fit-level checks).
_DISCRETE = ["nbn-cat-lw", "pgmpy-mle-ve", "pomegranate", "pyro"]

#: Adapters exposing ``extract_learned_cpts`` for parameter recovery.
_CPT_EXTRACTORS = {
    "nbn-cat-ve": lambda: NBNAdapter(mechanism="cat", engine="ve", device="cpu"),
    "pgmpy-mle-ve": lambda: PgmpyAdapter(param_method="mle", inference_method="ve"),
    "pgmpy-bayes-ve": lambda: PgmpyAdapter(param_method="bayes", inference_method="ve"),
    "pomegranate": lambda: PomegranateAdapter(device="cpu"),
    "pyro": lambda: PyroAdapter(mechanism="empirical", inference_method="importance",
                                device="cpu"),
}


# ---- applicability ------------------------------------------------------------


@pytest.mark.parametrize("family", ["discrete", "continuous_lg", "hybrid"])
@pytest.mark.parametrize("adapter_id", sorted(_ADAPTERS))
def test_is_applicable_delegates_to_the_registry(adapter_id, family):
    adapter = _ADAPTERS[adapter_id]()
    want = registry_is_applicable(adapter.name, family)
    assert adapter.is_applicable(_problem(family, n=20)) is want


@pytest.mark.parametrize("adapter_id", sorted(_ADAPTERS))
def test_unknown_name_is_never_applicable(adapter_id):
    adapter = _ADAPTERS[adapter_id]()
    adapter.name = f"{adapter.name}-not-in-registry"
    for family in ("discrete", "continuous_lg", "hybrid"):
        assert adapter.is_applicable(_problem(family, n=20)) is False


# ---- fit / query lifecycle ----------------------------------------------------


@pytest.mark.parametrize("adapter_id", sorted(_ADAPTERS))
def test_query_before_fit_raises(adapter_id):
    adapter = _ADAPTERS[adapter_id]()
    with pytest.raises(RuntimeError, match="not fitted"):
        adapter.query(Query(targets=("X0",), evidence={}, kind="marginal"))


@pytest.mark.parametrize("adapter_id", _DISCRETE)
def test_fit_accepts_the_runner_epochs_keyword(adapter_id):
    """Runner API compatibility: ``epochs=`` is passed to every adapter."""
    adapter = _ADAPTERS[adapter_id]()
    adapter.fit(_problem("discrete", n=200, seed=1), epochs=2)
    post = adapter.query(Query(targets=("X2",), evidence={"X0": 1}, kind="marginal"))
    assert post.probs is not None and torch.isfinite(post.probs).all()


@pytest.mark.parametrize("adapter_id", ["pomegranate", "pyro"])
def test_constructor_tolerates_unknown_kwargs(adapter_id):
    """These two take ``**kwargs`` so a newer YAML key does not break them."""
    if adapter_id == "pomegranate":
        adapter = PomegranateAdapter(some_future_kwarg=42)
    else:
        adapter = PyroAdapter(mechanism="empirical", inference_method="importance",
                              some_future_kwarg=True)
    assert adapter.name == _ADAPTERS[adapter_id]().name


# ---- parameter recovery: extract_learned_cpts ---------------------------------


@pytest.mark.parametrize("adapter_id", sorted(_CPT_EXTRACTORS))
def test_extract_learned_cpts_is_deterministic(adapter_id):
    adapter = _CPT_EXTRACTORS[adapter_id]()
    adapter.fit(_problem("discrete", n=300, seed=5))
    first, second = adapter.extract_learned_cpts(), adapter.extract_learned_cpts()
    assert set(first) == set(second)
    for node in first:
        assert torch.equal(first[node], second[node]), node


@pytest.mark.parametrize("adapter_id", sorted(_CPT_EXTRACTORS))
def test_extracted_multiparent_cpt_uses_the_canonical_row_order(adapter_id):
    adapter = _CPT_EXTRACTORS[adapter_id]()
    adapter.fit(_multiparent_problem())
    cpt = adapter.extract_learned_cpts()["X2"].float()
    assert tuple(cpt.shape) == (4, 3)
    torch.testing.assert_close(cpt.sum(-1), torch.ones(4), atol=1e-5, rtol=0)
    for a in (0, 1):            # X0, the slower axis (name order, not DAG order)
        for b in (0, 1):        # X1
            row = cpt[a * 2 + b]
            assert int(row.argmax()) == (a + 2 * b) % 3, (adapter_id, a, b, row)
            assert float(row.max()) > 0.95, (adapter_id, a, b, row)
