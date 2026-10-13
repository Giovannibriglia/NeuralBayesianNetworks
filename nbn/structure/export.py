"""Export factored probability sets, with their sharing, for planning and verification tools.

Robust value iteration and interval / parametric model checkers (PRISM, Storm)
plan with the sets NBN learns.  They need more than the tensors: which
(state, action) rows use which **factor row** (one factor parameter
``p_j(. | u)`` is shared by every row whose parents take the value ``u``), the
guarantee of each set, the confidence budget, the unseen rows, and which
quantifier the sets are meant for.  Without the sharing a tool can only build
per-row boxes (sound but pessimistic); with it, it can build the coupled
parametric model.

Format (``schema = "nbn.factored_sets/1"``): a directory with

``manifest.json``
    Variables and roles, the planning space, one record per factor (node,
    parents, set type, guarantee, level, sampling, assumptions,
    graph preservation), shared parameter blocks (noise polytopes), the
    hypotheses, the composition rule and the quantifiers.  JSON Schema:
    ``nbn/structure/schemas/factored_sets_v1.schema.json``.
``arrays.npz``
    Per factor ``<id>/lower``, ``<id>/upper`` ``[R, K]``, ``<id>/unseen``
    ``[R]``, optional ``center`` / ``radius`` (L1 balls), ``samples``
    (posterior draws), ``counts`` ``[R]`` (visits per factor row), ``index``
    (noise polytope) and, with ``exact=True``, outward-rounded rationals
    ``lower_num`` / ``upper_num`` over one ``denominator``.  One row map
    ``row_index/<j>`` ``[|S||A|]`` per distinct parent tuple (the factor row
    each planning row uses; the factor record names it).

Turning factor sets into joint per-row sets is the downstream tool's job: a
joint row is the product of its factor rows (multilinear in the factor
parameters), and entry-wise bounds of a product of non-negative factors are
products of the factor bounds (:func:`joint_row_bounds`).

References used for the content of the manifest (via the ``badings_claude``
check in #314): an IMDP / RMDP needs per-row intervals or an L1 centre and
radius; a parametric MDP needs polynomial transition functions over a
parameter space (Badings, *Robust Verification of Stochastic Systems*, 2025,
§3.3 and Ch. 8); graph preservation matters for some robust solvers and for
smooth solution functions (Suilen et al. 2024, Remark 1 and Def. 5); robust
planning ("for all nature, max policy") and parameter synthesis ("exists
parameters") use different quantifiers (Suilen et al. 2024, §4.1).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
import itertools
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple, Union

import numpy as np
import torch

from nbn.structure.agreement import AgreementSet
from nbn.structure.hypotheses import FactorSets, HypothesisSet
from nbn.uncertainty.sets import L1BallSet, NoiseParameterSet, ProbabilitySet, SampleSet

SCHEMA = "nbn.factored_sets/1"
SCHEMA_FILE = Path(__file__).parent / "schemas" / "factored_sets_v1.schema.json"
QUANTIFIERS = {
    "robust_planning": "for all kernels in the sets (nature), max over policies",
    "parameter_synthesis": "exists factor parameters in the sets, for all / some policies",
}
PlanningSets = Union[FactorSets, Mapping[str, Union[ProbabilitySet, AgreementSet, NoiseParameterSet]]]


def export_factored_sets(
    path: Union[str, Path],
    sets: PlanningSets,
    source: Any,
    state_vars: Sequence[str],
    action_vars: Sequence[str],
    *,
    next_suffix: str = "'",
    reward: str | None = None,
    termination: str | None = None,
    exact: bool = False,
    denominator: int = 10**9,
    max_rows: int = 1 << 22,
    overwrite: bool = False,
) -> Path:
    """Write ``manifest.json`` + ``arrays.npz`` for a factored transition model.

    Parameters
    ----------
    path:
        Output directory (created).  Refuses a non-empty directory unless
        ``overwrite``, which replaces a previous export there (and refuses a
        directory that holds other files).
    sets:
        Either a :class:`FactorSets` (every factor of a :class:`HypothesisSet`;
        the hypotheses are its product of candidates), or a mapping
        ``node -> set`` for one model: a :class:`ProbabilitySet` (parents from
        ``source``'s DAG), an :class:`AgreementSet` (its own parents), or a
        :class:`NoiseParameterSet` (the node is tied to that shared polytope;
        nodes given the same object share one parameter block).
    source:
        A fitted model (for its DAG and variables), a :class:`HypothesisSet`,
        or a variables mapping (enough for :class:`AgreementSet` values).
    state_vars, action_vars:
        The variables that form ``(s, a)``.  Every factor's parents must be
        among them.  Next-state variables are ``state + next_suffix``;
        ``reward`` and ``termination`` name optional extra nodes.
    exact:
        Also write outward-rounded rational bounds over one ``denominator``
        ``D``: ``lower_num / D <= lower`` and ``upper_num / D >= upper``
        exactly, for the factor boxes and the noise-polytope constraints.
    max_rows:
        Largest planning space ``|S||A|`` for which row maps are written;
        above it they are omitted (``planning_space.n_rows`` is null) and a
        tool computes rows from each factor's parents and the row layout
        (:func:`joint_row_bounds` does).

    Returns
    -------
    The output directory.
    """
    out = Path(path)
    _prepare_dir(out, overwrite)
    if isinstance(state_vars, str) or isinstance(action_vars, str):
        raise TypeError("state_vars and action_vars must be sequences of names.")
    state_vars, action_vars = tuple(state_vars), tuple(action_vars)
    if not 1 <= denominator <= 10**15:
        raise ValueError("denominator must be in [1, 1e15] (int64-safe numerators).")

    variables = _variables_of(source if not isinstance(sets, FactorSets) else sets.hypothesis_set)
    entries, hypotheses, candidates = _collect(sets, source)
    roles = _roles(variables, state_vars, action_vars, next_suffix, reward, termination)

    plan_vars = (*state_vars, *action_vars)
    for v in plan_vars:
        if v not in variables or not variables[v].is_discrete:
            raise ValueError(f"'{v}' must be a discrete variable to index the planning space.")
    plan_cards = [int(variables[v].cardinality) for v in plan_vars]
    n_plan = math.prod(plan_cards)
    write_rows = n_plan <= max_rows

    arrays: Dict[str, np.ndarray] = {}
    factors: List[dict] = []
    params: List[dict] = []
    param_ids: Dict[int, str] = {}
    row_maps: Dict[Tuple[str, ...], str] = {}     # one row map per distinct parent tuple
    events: List[Tuple[str, float, str]] = []      # (id, delta, guarantee)
    for fid, (node, parents, obj, counts) in entries.items():
        parents = tuple(parents)
        missing = [p for p in parents if p not in plan_vars]
        if missing:
            raise ValueError(f"Factor {node} <- {parents}: parents {missing} are not state or action variables.")
        k = int(variables[node].cardinality)
        record: Dict[str, Any] = {
            "id": fid, "node": node, "parents": list(parents),
            "parent_cards": [int(variables[p].cardinality) for p in parents], "n_classes": k,
        }
        if isinstance(obj, NoiseParameterSet):
            pid = param_ids.get(id(obj))
            if pid is None:
                pid = f"q{len(params)}"
                param_ids[id(obj)] = pid
                params.append(_param_record(pid, obj, arrays, exact, denominator))
                events.append((pid, 1.0 - obj.level, obj.guarantee))
            ps: ProbabilitySet = obj.table_set(node)
            record.update(set_type="noise_polytope", parameter=pid, guarantee=obj.guarantee,
                          level=obj.level, delta=None, sampling=None, assumptions=[])
            arrays[f"{fid}/index"] = _np(obj.index[node]).astype(np.int64)
        else:
            ps = obj.set if isinstance(obj, AgreementSet) else obj
            set_type = ("l1" if isinstance(ps, L1BallSet) else
                        "samples" if isinstance(ps, SampleSet) else "interval")
            record.update(set_type=set_type, guarantee=ps.guarantee, level=float(ps.level),
                          delta=float(1.0 - ps.level),
                          sampling=getattr(obj, "sampling", None),
                          assumptions=list(getattr(obj, "assumptions", ())))
            if isinstance(obj, AgreementSet):
                record["agreement"] = {
                    "max_parents": obj.max_parents, "survivors": [list(c) for c in obj.survivors],
                    "assumption_refuted": bool(obj.assumption_refuted),
                }
            if isinstance(ps, L1BallSet):
                arrays[f"{fid}/center"] = _np(ps.center)
                arrays[f"{fid}/radius"] = _np(ps.radius)
            if isinstance(ps, SampleSet):
                arrays[f"{fid}/samples"] = _np(ps.samples)
            events.append((fid, float(1.0 - ps.level), ps.guarantee))
        if record.get("sampling") is None and isinstance(sets, FactorSets):
            record["sampling"] = sets.sampling
        lower, upper = ps.bounds()                  # once per factor
        lo, hi = _np(lower), _np(upper)
        arrays[f"{fid}/lower"] = lo
        arrays[f"{fid}/upper"] = hi
        if exact:
            arrays[f"{fid}/lower_num"], arrays[f"{fid}/upper_num"] = _outward(lo, hi, denominator)
        arrays[f"{fid}/unseen"] = _np(ps.unseen).astype(bool)
        if counts is not None:
            arrays[f"{fid}/counts"] = _np(counts.sum(-1))
        rows_ok = _graph_preserving_rows(lower, upper) & ~ps.unseen.to(lower.device)
        record["graph_preserving"] = bool(rows_ok.all())
        record["n_rows"] = int(lower.shape[0])
        record["row_index"] = None
        if write_rows:
            key = row_maps.get(parents)
            if key is None:
                key = f"row_index/{len(row_maps)}"
                row_maps[parents] = key
                arrays[key] = _row_index(parents, plan_vars, plan_cards)
            record["row_index"] = key
        factors.append(record)

    delta_total = float(sum(d for _, d, _ in events))
    guarantees = sorted({g for _, _, g in events})
    manifest = {
        "schema": SCHEMA,
        "nbn_version": _nbn_version(),
        "variables": [
            {"name": v, "cardinality": int(variables[v].cardinality) if variables[v].is_discrete else None,
             "kind": "discrete" if variables[v].is_discrete else "continuous", "role": roles.get(v, "other")}
            for v in variables
        ],
        "planning_space": {
            "state_vars": list(state_vars), "action_vars": list(action_vars),
            "cards": plan_cards,
            "n_rows": n_plan if write_rows else None,
            "layout": "C order over state_vars + action_vars (last fastest)",
        },
        "row_layout": "C order over the factor's parents (last fastest), as nbn.uncertainty.parent_row_index",
        "factors": factors,
        "parameters": params,
        "hypotheses": {"candidates": candidates, "list": hypotheses},
        "composition": {
            "rule": "union_bound",
            "events": [e for e, _, _ in events],
            "delta_total": delta_total,
            "level": max(0.0, 1.0 - delta_total),
            "guarantees": guarantees,
            "note": ("All listed sets hold together with probability >= level (union bound); "
                     "'bayesian' or 'asymptotic' sets carry no frequentist statement."),
        },
        "quantifiers": QUANTIFIERS,
        "exact": {"denominator": int(denominator)} if exact else None,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=False))
    np.savez_compressed(out / "arrays.npz", **arrays)
    return out


@dataclass(eq=False)
class FactoredSetsExport:
    """What :func:`load_factored_sets` returns: the manifest and the arrays (NumPy)."""

    manifest: dict
    arrays: Dict[str, np.ndarray] = field(repr=False)

    def factor(self, fid: str) -> dict:
        for f in self.manifest["factors"]:
            if f["id"] == fid:
                return f
        raise KeyError(f"No factor '{fid}'.")

    def array(self, fid: str, name: str) -> np.ndarray:
        if name == "row_index":
            key = self.factor(fid).get("row_index")
            if key is None:
                raise KeyError(f"No row map for '{fid}' (planning space above max_rows); "
                               "use factor_row() instead.")
            return self.arrays[key]
        return self.arrays[f"{fid}/{name}"]

    def factor_row(self, fid: str, row: int) -> int:
        """The factor row that planning row ``row`` uses (from the row map or the layout)."""
        f = self.factor(fid)
        if f.get("row_index") is not None:
            return int(self.arrays[f["row_index"]][row])
        space = self.manifest["planning_space"]
        plan_vars = space["state_vars"] + space["action_vars"]
        digits = dict(zip(plan_vars, np.unravel_index(int(row), space["cards"])))
        index = 0
        for p, card in zip(f["parents"], f["parent_cards"]):
            index = index * card + int(digits[p])
        return index

    @property
    def n_hypotheses(self) -> int:
        if self.manifest["hypotheses"]["list"]:
            return len(self.manifest["hypotheses"]["list"])
        return math.prod(len(v) for v in self.manifest["hypotheses"]["candidates"].values())

    def hypotheses(self) -> Iterator[Dict[str, str]]:
        """Every hypothesis as ``{node: factor id}``, lazily (product of the candidates)."""
        listed = self.manifest["hypotheses"]["list"]
        if listed:
            yield from listed
            return
        cands = self.manifest["hypotheses"]["candidates"]
        nodes = list(cands)
        for choice in itertools.product(*(cands[n] for n in nodes)):
            yield dict(zip(nodes, choice))


def load_factored_sets(path: Union[str, Path]) -> FactoredSetsExport:
    """Read an export written by :func:`export_factored_sets`."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Unknown schema {manifest.get('schema')!r}; expected {SCHEMA!r}.")
    with np.load(path / "arrays.npz", allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    return FactoredSetsExport(manifest=manifest, arrays=arrays)


def joint_row_bounds(
    export: FactoredSetsExport, hypothesis: Mapping[str, str], row: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Entry-wise bounds of the joint next-state distribution of one planning row.

    ``hypothesis`` maps each next-state node to a factor id.  The joint
    probability of ``(x'_1, .., x'_n)`` is the product of the factor rows the
    planning row uses, so its bounds are products of the factor bounds (all
    non-negative).  Returns ``lower``, ``upper`` with one axis per node (C
    order) and the node order.  Unseen factor rows contribute ``[0, 1]``.
    Works with or without stored row maps.
    """
    nodes = list(hypothesis)
    lower = np.ones(())
    upper = np.ones(())
    for node in nodes:
        fid = hypothesis[node]
        r = export.factor_row(fid, row)
        lower = np.multiply.outer(lower, export.array(fid, "lower")[r])
        upper = np.multiply.outer(upper, export.array(fid, "upper")[r])
    return lower, np.minimum(upper, 1.0), nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect(sets: PlanningSets, source: Any):
    """Factor entries {id: (node, parents, set, counts)}, hypotheses list, candidates."""
    entries: Dict[str, tuple] = {}
    if isinstance(sets, FactorSets):
        hs = sets.hypothesis_set
        if source is not None and source is not hs and not isinstance(source, Mapping):
            raise ValueError("A FactorSets export takes its HypothesisSet (or None) as source.")
        candidates: Dict[str, List[str]] = {}
        for i, (node, parents) in enumerate(hs.factors):
            fid = f"f{i}"
            entries[fid] = (node, parents, sets.sets[(node, parents)], sets.counts.tables[(node, parents)])
            candidates.setdefault(node, []).append(fid)
        return entries, [], candidates
    if not isinstance(sets, Mapping) or not sets:
        raise TypeError("sets must be a FactorSets or a non-empty mapping node -> set.")
    hypothesis: Dict[str, str] = {}
    for i, (node, obj) in enumerate(sets.items()):
        fid = f"f{i}"
        if isinstance(obj, AgreementSet):
            parents = obj.parents
            if obj.node != node:
                raise ValueError(f"The agreement set for '{obj.node}' is listed under '{node}'.")
        elif isinstance(obj, (ProbabilitySet, NoiseParameterSet)):
            dag = getattr(source, "dag", None)
            if dag is None:
                raise ValueError(f"The parents of '{node}' come from a model's DAG: pass the model as source.")
            parents = tuple(dag.parents(node))
            if isinstance(obj, NoiseParameterSet) and node not in obj.index:
                raise ValueError(f"'{node}' is not tied to this noise polytope.")
        else:
            raise TypeError(f"Unsupported set for '{node}': {type(obj).__name__}.")
        entries[fid] = (node, parents, obj, None)
        hypothesis[node] = fid
    return entries, [hypothesis], {n: [f] for n, f in hypothesis.items()}


def _variables_of(source: Any) -> Mapping[str, Any]:
    if isinstance(source, HypothesisSet):
        return source.variables
    if hasattr(source, "variables"):
        return source.variables
    if isinstance(source, Mapping):
        from nbn.core.network import _build_variable

        return {k: _build_variable(k, v) for k, v in source.items()}
    raise TypeError("source must be a model, a HypothesisSet or a variables mapping.")


def _roles(variables, state_vars, action_vars, suffix, reward, termination) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    for v in state_vars:
        roles[v] = "state"
        if v + suffix in variables:
            roles[v + suffix] = "next_state"
    for v in action_vars:
        roles[v] = "action"
    for name, role in ((reward, "reward"), (termination, "termination")):
        if name is not None:
            if name not in variables:
                raise KeyError(f"Unknown {role} variable '{name}'.")
            roles[name] = role
    return roles


def _param_record(pid: str, obj: NoiseParameterSet, arrays: Dict[str, np.ndarray],
                  exact: bool, denominator: int) -> dict:
    lo, hi = _np(obj.lower), _np(obj.upper)
    arrays[f"{pid}/expressions"] = _np(obj.expressions).astype(np.int8)
    arrays[f"{pid}/lower"] = lo
    arrays[f"{pid}/upper"] = hi
    if exact:
        arrays[f"{pid}/lower_num"], arrays[f"{pid}/upper_num"] = _outward(lo, hi, denominator)
    return {
        "id": pid, "kind": "noise_polytope", "n_values": int(obj.expressions.shape[1]),
        "n_expressions": int(obj.expressions.shape[0]), "guarantee": obj.guarantee,
        "level": float(obj.level), "delta": float(1.0 - obj.level), "nodes": sorted(obj.index),
        "constraints": "q in the simplex and lower_e <= expressions[e] @ q <= upper_e for every e; "
                       "table entry (r, k) of a tied node is expressions[index[r, k]] @ q",
    }


def _outward(lo: np.ndarray, hi: np.ndarray, denominator: int) -> Tuple[np.ndarray, np.ndarray]:
    """Numerators with ``num_lo / D <= lo`` and ``num_hi / D >= hi`` exactly.

    The float product ``x * D`` is within half an ulp of the exact value, so
    stepping one ulp outward before floor / ceil is always on the safe side
    (an exact integer product loses at most 1 / D of tightness).  Probability
    bounds lie in [0, 1], so clamping to [0, D] keeps soundness.
    """
    d = float(denominator)
    num_lo = np.floor(np.nextafter(lo * d, -np.inf))
    num_hi = np.ceil(np.nextafter(hi * d, np.inf))
    num_lo = np.where(lo <= 0.0, 0.0, np.maximum(num_lo, 0.0))
    num_hi = np.where(hi >= 1.0, d, np.minimum(num_hi, d))
    return num_lo.astype(np.int64), num_hi.astype(np.int64)


def _row_index(parents: Tuple[str, ...], plan_vars: Tuple[str, ...], plan_cards: List[int]) -> np.ndarray:
    """Factor row of every planning row, computed digit by digit (no dense grid)."""
    n_plan = math.prod(plan_cards)
    strides = [math.prod(plan_cards[i + 1:]) for i in range(len(plan_cards))]
    rows = np.arange(n_plan, dtype=np.int64)
    index = np.zeros(n_plan, dtype=np.int64)
    for p in parents:
        i = plan_vars.index(p)
        index = index * plan_cards[i] + (rows // strides[i]) % plan_cards[i]
    return index


def _prepare_dir(out: Path, overwrite: bool) -> None:
    if out.exists() and not out.is_dir():
        raise FileExistsError(f"{out} exists and is not a directory; the export is a directory.")
    if out.is_dir():
        present = {p.name for p in out.iterdir()}
        if present and not overwrite:
            raise FileExistsError(f"{out} is not empty; pass overwrite=True to replace a previous export.")
        others = present - {"manifest.json", "arrays.npz"}
        if others:
            raise FileExistsError(f"{out} holds other files {sorted(others)}; refusing to overwrite.")
        for name in present:
            (out / name).unlink()


def _graph_preserving_rows(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Rows whose every entry is either fixed at zero or bounded away from zero."""
    return ((upper <= 0) | (lower > 0)).all(-1)


def _np(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    if torch.is_floating_point(t):
        t = t.double()
    return t.numpy()


def _nbn_version() -> str:
    try:
        from nbn import __version__

        return str(__version__)
    except Exception:  # pragma: no cover
        return "unknown"
