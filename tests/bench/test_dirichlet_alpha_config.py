"""Dirichlet alpha of the count-table CPTs, configurable from YAML (#160).

``CategoricalTableMechanism(alpha=...)`` always took a pseudo-count, but the
benchmark could only reach alpha=0 (``cat``) and alpha=1 (``cat-bayes``).
A baseline now sets it with ``extra_kwargs: {alpha: <float>}``; unset keeps
every historical label and behaviour, and a non-default value is tagged into
the label so results with different priors stay apart.
"""
from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import pytest
import torch
import yaml

from nbn.bench._paper_agg import approach
from nbn.bench.adapters.nbn_adapter import NBNAdapter
from nbn.bench.core.config import BaselineSpec, build_adapter
from nbn.bench.core.runner import _fit_identity_key
from nbn.bench.core.yaml_config import load_runner_config
from nbn.bench.domains.base import BenchmarkProblem
from nbn.mechanisms import (
    CategoricalTableMechanism,
    SmoothedEmpiricalCategoricalMechanism,
)


def _mech(adapter: NBNAdapter):
    return adapter._make_mech("discrete", 3, ["discrete"])


# ---- defaults are untouched -------------------------------------------------

@pytest.mark.parametrize("mechanism,engine,label,alpha", [
    ("cat", "ve", "nbn-cat-ve", 0.0),
    ("cat", None, "nbn-cat", 0.0),
    ("cat-bayes", "lw", "nbn-cat-bayes-lw", 1.0),
    ("cat-bayes", None, "nbn-cat-bayes", 1.0),
])
def test_unset_alpha_keeps_labels_and_mechanism_defaults(mechanism, engine, label, alpha):
    adapter = NBNAdapter(mechanism=mechanism, engine=engine, device="cpu")
    assert adapter.name == label
    assert adapter.alpha == alpha
    assert _mech(adapter).alpha == alpha


@pytest.mark.parametrize("mechanism,alpha", [("cat", 0.0), ("cat-bayes", 1.0)])
def test_alpha_equal_to_the_default_keeps_the_label(mechanism, alpha):
    adapter = NBNAdapter(mechanism=mechanism, engine="ve", device="cpu", alpha=alpha)
    assert adapter.name == f"nbn-{mechanism}-ve"


# ---- a configured alpha reaches the mechanism and the label -----------------

@pytest.mark.parametrize("mechanism,cls", [
    ("cat", CategoricalTableMechanism),
    ("cat-bayes", SmoothedEmpiricalCategoricalMechanism),
])
def test_alpha_reaches_the_table_mechanism_and_the_label(mechanism, cls):
    adapter = NBNAdapter(mechanism=mechanism, engine="ve", device="cpu", alpha=0.5)
    mech = _mech(adapter)
    assert type(mech) is cls and mech.alpha == 0.5
    assert adapter.name == f"nbn-{mechanism}-alpha0.5-ve"
    # The tag never changes applicability: same families as the base label.
    problem = _problem()
    assert adapter.is_applicable(problem)


def test_alpha_applies_to_the_table_fallback_of_continuous_mechanisms():
    adapter = NBNAdapter(mechanism="mdn", engine="lw", device="cpu", alpha=2)
    assert _mech(adapter).alpha == 2.0
    assert adapter.name == "nbn-mdn-alpha2-lw"
    # Discrete children of continuous parents stay neural (no table).
    neural = adapter._make_mech("discrete", 3, ["continuous"])
    assert not isinstance(neural, CategoricalTableMechanism)


def test_tiny_alpha_label_has_no_dash_and_parses():
    adapter = NBNAdapter(mechanism="cat", engine="ve", device="cpu", alpha=1e-5)
    assert adapter.name == "nbn-cat-alpha0.00001-ve"
    assert approach(adapter.name) == ("nbn-cat-alpha0.00001", "ve")


# ---- validation -------------------------------------------------------------

@pytest.mark.parametrize("bad", [-1.0, -1e-9, math.nan, math.inf, -math.inf,
                                 "abc", "-1", True, [0.5]])
def test_invalid_alpha_is_rejected_by_the_adapter(bad):
    with pytest.raises(ValueError, match="alpha"):
        NBNAdapter(mechanism="cat", engine="ve", device="cpu", alpha=bad)


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_invalid_alpha_is_rejected_by_the_mechanism(bad):
    # Regression (#160): master accepted any float here, so a negative or
    # NaN pseudo-count silently produced invalid CPTs.
    with pytest.raises(ValueError, match="alpha"):
        CategoricalTableMechanism(alpha=bad)
    with pytest.raises(ValueError, match="alpha"):
        SmoothedEmpiricalCategoricalMechanism(alpha=bad)


def test_alpha_for_a_mechanism_without_tables_is_an_error():
    with pytest.raises(ValueError, match="neuralcat"):
        NBNAdapter(mechanism="neuralcat", engine="ve", device="cpu", alpha=0.5)
    assert NBNAdapter(mechanism="neuralcat", engine="ve", device="cpu").alpha is None


# ---- YAML -> runner -> fitted model -----------------------------------------

def _config(tmp_path: Path, baselines: list[dict]) -> Path:
    d = {
        "version": "v0.13",
        "benchmark": "synthetic",
        "config_name": "alpha_test",
        "source": {
            "families": ["discrete"], "n_nodes_list": [5], "seeds": [0],
            "n_train": 100, "n_reference": 200, "edge_density": 0.20,
            "max_in_degree": 2, "cardinality": 4, "fraction_continuous": 0.0,
        },
        "baselines": baselines,
        "n_queries_per_cell": 2,
        "per_cell_timeout_s": 30.0,
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def test_yaml_extra_kwargs_alpha_builds_distinct_baselines(tmp_path):
    cfg = load_runner_config(_config(tmp_path, [
        {"library": "nbn", "mechanism": "cat", "param_method": "mle",
         "inference_method": "ve"},
        {"library": "nbn", "mechanism": "cat", "param_method": "mle",
         "inference_method": "ve", "extra_kwargs": {"alpha": 0.5}},
        {"library": "nbn", "mechanism": "cat-bayes", "param_method": "mle",
         "inference_method": "ve", "extra_kwargs": {"alpha": 0.01}},
    ]), jsonl_path=tmp_path / "m.jsonl")
    names = [build_adapter(s, require_engine=True).name for s in cfg.baselines]
    assert names == ["nbn-cat-ve", "nbn-cat-alpha0.5-ve", "nbn-cat-bayes-alpha0.01-ve"]
    assert len(set(names)) == len(names)


def test_bad_yaml_alpha_fails_when_the_adapter_is_built(tmp_path):
    cfg = load_runner_config(_config(tmp_path, [
        {"library": "nbn", "mechanism": "cat", "param_method": "mle",
         "inference_method": "ve", "extra_kwargs": {"alpha": -0.5}},
    ]), jsonl_path=tmp_path / "m.jsonl")
    with pytest.raises(ValueError, match="alpha"):
        build_adapter(cfg.baselines[0])


def test_alpha_is_part_of_the_fit_identity():
    """Baselines that differ only in alpha must not share a cached base fit."""
    problem = _problem()

    def spec(mechanism="cat", **extra):
        return BaselineSpec(library="nbn", mechanism=mechanism, param_method="mle",
                            inference_method="ve", extra_kwargs=extra)

    assert _fit_identity_key(spec(), problem) != _fit_identity_key(spec(alpha=0.5), problem)
    assert (_fit_identity_key(spec(alpha=0.5), problem)
            == _fit_identity_key(spec(alpha="0.5"), problem))
    # Regression (review #301): unset and the mechanism's default are the same
    # fit and the same label, so they must share one base fit.
    assert _fit_identity_key(spec(), problem) == _fit_identity_key(spec(alpha=0.0), problem)
    assert (_fit_identity_key(spec("cat-bayes"), problem)
            == _fit_identity_key(spec("cat-bayes", alpha=1), problem))


def test_yaml_exponent_without_a_dot_is_accepted(tmp_path):
    """Regression (review #301): PyYAML reads ``1e-5`` as the string '1e-5';
    the adapter converts numeric strings, and still rejects other strings."""
    p = tmp_path / "raw.yaml"
    p.write_text(yaml.safe_dump(yaml.safe_load(_config(tmp_path, []).read_text()))
                 .replace("baselines: []", "baselines:\n- {library: nbn, mechanism: cat, "
                          "param_method: mle, inference_method: ve, "
                          "extra_kwargs: {alpha: 1e-5}}"))
    cfg = load_runner_config(p, jsonl_path=tmp_path / "m.jsonl")
    assert cfg.baselines[0].extra_kwargs["alpha"] == "1e-5"  # what PyYAML gives
    adapter = build_adapter(cfg.baselines[0])
    assert adapter.alpha == 1e-5 and adapter.name == "nbn-cat-alpha0.00001-ve"
    with pytest.raises(ValueError, match="alpha"):
        NBNAdapter(mechanism="cat", engine="ve", device="cpu", alpha="lots")


@pytest.mark.parametrize("label,approach_key,engine,short", [
    ("nbn-cat-alpha0.5-ve", "nbn-cat-alpha0.5", "ve", "NBN cat+VE (α=0.5)"),
    ("nbn-cat-bayes-ve", "nbn-cat-bayes", "ve", "NBN cat-Bayes+VE"),
    ("nbn-cat-bayes-alpha0.01-lw", "nbn-cat-bayes-alpha0.01", "lw",
     "NBN cat-Bayes+LW (α=0.01)"),
    ("nbn-mdn-alpha2", "nbn-mdn-alpha2", "", "NBN MDN (α=2)"),
])
def test_plot_labels_keep_the_engine_and_the_alpha(label, approach_key, engine, short):
    """Regression (review #301): the alpha tag fell back to the grey colour and
    ``short_label`` dropped the engine ('NBN cat+ALPHA0.5'); the same engine
    loss hit 'nbn-cat-bayes-ve' ('NBN cat+BAYES') on master."""
    from nbn.bench._paper_agg import _APPROACH_COLOR, approach, method_style
    from nbn.bench._paper_panels import short_label

    assert approach(label) == (approach_key, engine)
    base = "nbn-" + label.split("-")[1] + ("-bayes" if "-bayes" in label else "")
    assert method_style(label)[0] == _APPROACH_COLOR[base]
    assert short_label(label) == short


def _problem(n: int = 40, seed: int = 0) -> BenchmarkProblem:
    g = torch.Generator().manual_seed(seed)
    dag = nx.DiGraph([("X0", "X1")])
    x0 = torch.randint(0, 3, (n,), generator=g)
    x0[:] = torch.where(x0 == 2, torch.zeros_like(x0), x0)  # class 2 never seen
    data = {"X0": x0, "X1": torch.randint(0, 3, (n,), generator=g)}
    return BenchmarkProblem(
        name="alpha", dag=dag,
        variables={"X0": ("discrete", 3), "X1": ("discrete", 3)},
        train_data=data, test_data=data, queries=[],
    )


@pytest.mark.parametrize("alpha", [0.0, 0.5, 3.0])
def test_fitted_tables_match_a_directly_built_mechanism(alpha):
    problem = _problem()
    adapter = NBNAdapter(mechanism="cat", engine="ve", device="cpu", alpha=alpha)
    adapter.fit(problem)
    ref = CategoricalTableMechanism(alpha=alpha)
    ref.fit_local(problem.train_data["X0"], None, n_classes=3)
    got = adapter.model.mechanisms["X0"]
    assert got.alpha == alpha
    torch.testing.assert_close(torch.softmax(got.tabulate([]), -1),
                               torch.softmax(ref.tabulate([]), -1))


def test_larger_alpha_pulls_the_root_prior_towards_uniform():
    problem = _problem()
    probs = []
    for alpha in (0.0, 0.5, 5.0):
        adapter = NBNAdapter(mechanism="cat", engine="ve", device="cpu", alpha=alpha)
        adapter.fit(problem)
        probs.append(torch.softmax(adapter.model.mechanisms["X0"].tabulate([]).detach(), -1))
    # Class 2 is never observed: its mass grows strictly with alpha.
    assert float(probs[0][2]) < float(probs[1][2]) < float(probs[2][2])
