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
from nbn.uncertainty.confounded import ConfoundedSet
from nbn.uncertainty.sets import L1BallSet, NoiseParameterSet, ProbabilitySet, SampleSet

SCHEMA = "nbn.factored_sets/1"
SCHEMA_FILE = Path(__file__).parent / "schemas" / "factored_sets_v1.schema.json"
QUANTIFIERS = {
    "robust_planning": "for all kernels in the sets (nature), max over policies",
    "parameter_synthesis": "exists factor parameters in the sets, for all / some policies",
}
PlanningSets = Union[
    FactorSets, Mapping[str, Union[ProbabilitySet, AgreementSet, ConfoundedSet, NoiseParameterSet]]
]


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
    parents: Mapping[str, Sequence[str]] | None = None,
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
        ``parents[node]`` if given, else from ``source``'s DAG, in that
        order: a table counted in another parent order must say so through
        ``parents``, since equal shapes cannot reveal it), an :class:`AgreementSet` (its own parents), or a
        :class:`NoiseParameterSet` (the node is tied to that shared polytope;
        nodes given the same object share one parameter block), or a
        :class:`~nbn.uncertainty.ConfoundedSet` (interventional rows of a
        confounded action; refuted rows are recorded as such).
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
        Bounds at 0 and 1 stay exact (a structural zero stays impossible).
        A factor's ``graph_preserving`` flag then holds for both the float
        and the rational bounds.
    max_rows:
        Largest total size of the row maps: ``|S||A|`` times the number of
        distinct parent tuples.  Above it no row map is written
        (``planning_space.n_rows`` is null) and a tool computes rows from
        each factor's parents and the row layout (:func:`joint_row_bounds`
        does).
    parents:
        Optional ``node -> parent tuple`` for :class:`ProbabilitySet` values
        (see ``sets``); with a model as ``source`` it must name the DAG's
        parents, in any order.

    Returns
    -------
    The output directory.
    """
    out = Path(path)
    _check_target(out, overwrite)            # nothing is deleted before the export is complete
    if isinstance(state_vars, str) or isinstance(action_vars, str):
        raise TypeError("state_vars and action_vars must be sequences of names.")
    state_vars, action_vars = tuple(state_vars), tuple(action_vars)
    plan_vars = (*state_vars, *action_vars)
    if len(set(plan_vars)) != len(plan_vars):
        raise ValueError(f"state_vars and action_vars must be distinct names; got {list(plan_vars)}.")
    if not 1 <= denominator <= 10**15:
        raise ValueError("denominator must be in [1, 1e15] (int64-safe numerators).")

    variables = _variables_of(source if not isinstance(sets, FactorSets) else sets.hypothesis_set)
    entries, hypotheses, candidates = _collect(sets, source, parents)
    roles = _roles(variables, state_vars, action_vars, next_suffix, reward, termination)

    for v in plan_vars:
        if v not in variables or not variables[v].is_discrete:
            raise ValueError(f"'{v}' must be a discrete variable to index the planning space.")
    plan_cards = [int(variables[v].cardinality) for v in plan_vars]
    n_plan = math.prod(plan_cards)
    n_maps = len({tuple(e[1]) for e in entries.values()})
    write_rows = n_plan * n_maps <= max_rows

    arrays: Dict[str, np.ndarray] = {}
    factors: List[dict] = []
    params: List[dict] = []
    param_ids: Dict[int, str] = {}
    row_maps: Dict[Tuple[str, ...], str] = {}     # one row map per distinct parent tuple
    events: List[Tuple[str, float, str]] = []      # (id, delta, guarantee)
    for fid, (node, f_parents, obj, counts) in entries.items():
        f_parents = tuple(f_parents)
        missing = [p for p in f_parents if p not in plan_vars]
        if missing:
            raise ValueError(f"Factor {node} <- {f_parents}: parents {missing} are not state or action variables.")
        k = int(variables[node].cardinality)
        record: Dict[str, Any] = {
            "id": fid, "node": node, "parents": list(f_parents),
            "parent_cards": [int(variables[p].cardinality) for p in f_parents], "n_classes": k,
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
            idx = _np(obj.index[node]).astype(np.int64)
            if idx.shape != (math.prod(record["parent_cards"]), k):
                raise ValueError(f"Noise index of '{node}' has shape {idx.shape}; its parents "
                                 f"{f_parents} and {k} classes give {(math.prod(record['parent_cards']), k)}.")
            arrays[f"{fid}/index"] = idx
        else:
            ps = obj.set if isinstance(obj, (AgreementSet, ConfoundedSet)) else obj
            set_type = ("l1" if isinstance(ps, L1BallSet) else
                        "samples" if isinstance(ps, SampleSet) else "interval")
            record.update(set_type=set_type, guarantee=ps.guarantee, level=float(ps.level),
                          delta=float(1.0 - ps.level),
                          sampling=getattr(obj, "sampling", None),
                          assumptions=list(getattr(obj, "assumptions", ())))
            if isinstance(obj, ConfoundedSet):
                record["confounded"] = {
                    "action": obj.action, "n_regimes": obj.n_regimes, "method_used": list(obj.method_used),
                    "declared_unconfounded": list(obj.declared_unconfounded),
                    "refuted_configurations": int(obj.refuted.sum()), "notes": list(obj.notes),
                }
                # Refuted rows are the simplex and `unseen`; this keeps the reason.
                arrays[f"{fid}/refuted"] = _np(obj.refuted_rows).astype(bool)
            if isinstance(obj, AgreementSet):
                record["agreement"] = {
                    "max_parents": obj.max_parents, "survivors": [list(c) for c in obj.survivors],
                    "assumption_refuted": bool(obj.assumption_refuted), "refute": bool(obj.refute),
                }
            if isinstance(ps, L1BallSet):
                arrays[f"{fid}/center"] = _np(ps.center)
                arrays[f"{fid}/radius"] = _np(ps.radius)
            if isinstance(ps, SampleSet):
                arrays[f"{fid}/samples"] = ps.samples.detach().cpu().numpy()     # keep the draws' dtype
            events.append((fid, float(1.0 - ps.level), ps.guarantee))
        if record.get("sampling") is None and isinstance(sets, FactorSets):
            record["sampling"] = sets.sampling
        lower, upper = ps.bounds()                  # once per factor
        n_expected = math.prod(record["parent_cards"])
        if lower.shape != (n_expected, k):
            raise ValueError(f"The set of '{node}' has shape {tuple(lower.shape)}; its parents {f_parents} "
                             f"and {k} classes give {(n_expected, k)} (wrong set or wrong parent order).")
        if ps.unseen.numel() == 0:                  # e.g. bootstrap sets: no unseen info
            unseen = torch.zeros(n_expected, dtype=torch.bool, device=lower.device)
        elif ps.unseen.numel() == n_expected:
            unseen = ps.unseen.reshape(-1)
        else:
            raise ValueError(f"The set of '{node}' has an unseen mask of {ps.unseen.numel()} rows, "
                             f"not {n_expected}.")
        lo, hi = _np(lower), _np(upper)
        arrays[f"{fid}/lower"] = lo
        arrays[f"{fid}/upper"] = hi
        unseen_np = _np(unseen).astype(bool)
        arrays[f"{fid}/unseen"] = unseen_np
        if counts is not None:
            arrays[f"{fid}/counts"] = _np(counts.sum(-1))
        rows_ok = _graph_preserving_rows_np(lo, hi) & ~unseen_np
        if exact:
            lo_num, hi_num = _outward(lo, hi, denominator)
            arrays[f"{fid}/lower_num"], arrays[f"{fid}/upper_num"] = lo_num, hi_num
            rows_ok &= _graph_preserving_rows_np(lo_num, hi_num)      # rounding can floor a tiny lower to 0
        record["graph_preserving"] = bool(rows_ok.all())
        record["n_rows"] = int(lower.shape[0])
        record["row_index"] = None
        if write_rows:
            key = row_maps.get(f_parents)
            if key is None:
                key = f"row_index/{len(row_maps)}"
                row_maps[f_parents] = key
                arrays[key] = _row_index(f_parents, plan_vars, plan_cards)
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
            "note": _composition_note(isinstance(sets, FactorSets)),
        },
        "quantifiers": QUANTIFIERS,
        "exact": {"denominator": int(denominator)} if exact else None,
    }
    for prm in params:                         # list only the tied nodes that are exported
        prm["nodes"] = sorted(f["node"] for f in factors if f.get("parameter") == prm["id"])
    _write_atomically(out, manifest, arrays)
    return out


@dataclass(eq=False)
class FactoredSetsExport:
    """What :func:`load_factored_sets` returns: the manifest and the arrays (NumPy)."""

    manifest: dict
    arrays: Dict[str, np.ndarray] = field(repr=False)
    _by_id: Dict[str, dict] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {f["id"]: f for f in self.manifest["factors"]}

    def factor(self, fid: str) -> dict:
        try:
            return self._by_id[fid]
        except KeyError:
            raise KeyError(f"No factor '{fid}'.") from None

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
        return int(_layout_rows(np.array([int(row)]), plan_vars, space["cards"], f["parents"],
                                f["parent_cards"])[0])

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
    order) and the node order.  Unseen factor rows contribute ``[0, 1]``
    whatever their stored bounds (e.g. a prior-only Dirichlet row).  Works
    with or without stored row maps.
    """
    nodes = list(hypothesis)
    lower = np.ones(())
    upper = np.ones(())
    for node in nodes:
        fid = hypothesis[node]
        r = export.factor_row(fid, row)
        lo, hi = export.array(fid, "lower")[r], export.array(fid, "upper")[r]
        if bool(export.array(fid, "unseen")[r]):
            lo, hi = np.zeros_like(lo), np.ones_like(hi)
        lower = np.multiply.outer(lower, lo)
        upper = np.multiply.outer(upper, hi)
    return lower, np.minimum(upper, 1.0), nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _composition_note(hypothesis_set: bool) -> str:
    note = ("All listed sets hold together with probability >= level (union bound); "
            "'bayesian' or 'asymptotic' sets carry no frequentist statement.")
    if hypothesis_set:
        note += (" For a hypothesis set, a factor's set bounds the true conditional only if its "
                 "parents contain the true parents; a factor that misses one bounds a "
                 "policy-dependent average. So the level speaks for the hypotheses whose factors "
                 "are all supersets of the true parents (which ones is the structural question).")
    return note


def _collect(sets: PlanningSets, source: Any, parents_override: Mapping[str, Sequence[str]] | None = None):
    """Factor entries {id: (node, parents, set, counts)}, hypotheses list, candidates."""
    entries: Dict[str, tuple] = {}
    if parents_override is not None and isinstance(sets, FactorSets):
        raise ValueError("parents applies to a mapping of sets; a FactorSets carries its own parents.")
    if isinstance(sets, FactorSets):
        hs = sets.hypothesis_set
        if source is not None and source is not hs and not isinstance(source, Mapping):
            raise ValueError("A FactorSets export takes its HypothesisSet (or None) as source.")
        candidates: Dict[str, List[str]] = {}
        missing = [f for f in hs.factors if f not in sets.sets]
        if missing:
            raise ValueError(
                f"These FactorSets have no count-based set for {missing} (continuous or mixed factors); "
                "export a mapping node -> set for one hypothesis instead."
            )
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
        if isinstance(obj, (AgreementSet, ConfoundedSet)):
            parents = obj.parents
            if obj.node != node:
                raise ValueError(f"The set for '{obj.node}' is listed under '{node}'.")
        elif isinstance(obj, (ProbabilitySet, NoiseParameterSet)):
            dag = getattr(source, "dag", None)
            given = None if parents_override is None else parents_override.get(node)
            if isinstance(given, str):
                raise TypeError(f"parents['{node}'] must be a tuple of names, not {given!r}.")
            if given is not None and isinstance(obj, NoiseParameterSet):
                raise ValueError(f"parents['{node}']: a noise polytope's index follows the model's DAG.")
            if given is not None:
                parents = tuple(given)
                if dag is not None and sorted(parents) != sorted(dag.parents(node)):
                    raise ValueError(f"parents['{node}'] = {parents} are not the DAG's parents "
                                     f"{tuple(dag.parents(node))}.")
            elif dag is None:
                raise ValueError(f"The parents of '{node}' come from a model's DAG: pass the model as "
                                 f"source, or give them in parents.")
            else:
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
                       "table entry (r, k) of a tied node is expressions[index[r, k]] @ q, "
                       "except for rows marked unseen in the factor: their location is a guess, "
                       "so those rows are unconstrained (the whole simplex)",
    }


def _outward(lo: np.ndarray, hi: np.ndarray, denominator: int) -> Tuple[np.ndarray, np.ndarray]:
    """Numerators with ``num_lo / D <= lo`` and ``num_hi / D >= hi`` exactly.

    The float product ``x * D`` is within half an ulp of the exact value, so
    stepping one ulp outward before floor / ceil is always on the safe side
    (an exact integer product loses at most 1 / D of tightness).  Probability
    bounds lie in [0, 1], so clamping to [0, D] keeps soundness.  The end
    points 0 and 1 are exact in both bounds: an upper bound of 0 (an
    impossible transition) stays 0, and a lower bound of 1 stays ``D``.
    """
    d = float(denominator)
    num_lo = np.floor(np.nextafter(lo * d, -np.inf))
    num_hi = np.ceil(np.nextafter(hi * d, np.inf))
    num_lo = np.where(lo <= 0.0, 0.0, np.where(lo >= 1.0, d, np.clip(num_lo, 0.0, d)))
    num_hi = np.where(hi >= 1.0, d, np.where(hi <= 0.0, 0.0, np.clip(num_hi, 0.0, d)))
    return num_lo.astype(np.int64), num_hi.astype(np.int64)


def _layout_rows(rows: np.ndarray, plan_vars, plan_cards, parents, parent_cards) -> np.ndarray:
    """Factor row of each planning row: C-order digits of ``rows``, re-raveled over ``parents``."""
    digits = np.unravel_index(rows, list(plan_cards)) if len(plan_cards) else ()
    if not parents:
        return np.zeros(rows.shape, dtype=np.int64)
    cols = [digits[list(plan_vars).index(p)] for p in parents]
    return np.ravel_multi_index(cols, list(parent_cards)).astype(np.int64)


def _row_index(parents: Tuple[str, ...], plan_vars: Tuple[str, ...], plan_cards: List[int]) -> np.ndarray:
    """Factor row of every planning row (int32 when it fits)."""
    n_plan = math.prod(plan_cards)
    cards = [plan_cards[plan_vars.index(p)] for p in parents]
    index = _layout_rows(np.arange(n_plan, dtype=np.int64), plan_vars, plan_cards, parents, cards)
    return index.astype(np.int32) if math.prod(cards) < 2**31 else index


def _check_target(out: Path, overwrite: bool) -> None:
    """Refuse a non-export target up front; nothing is deleted here."""
    if out.exists() and not out.is_dir():
        raise FileExistsError(f"{out} exists and is not a directory; the export is a directory.")
    if out.is_dir():
        present = {p.name for p in out.iterdir()}
        if present and not overwrite:
            raise FileExistsError(f"{out} is not empty; pass overwrite=True to replace a previous export.")
        others = {p for p in present - {"manifest.json", "arrays.npz"} if not _os_metadata(p)}
        if others:
            raise FileExistsError(f"{out} holds other files {sorted(others)}; refusing to overwrite.")


def _os_metadata(name: str) -> bool:
    """Files a desktop creates by itself (Finder, Explorer); an overwrite may drop them."""
    return name in (".DS_Store", "Thumbs.db", "desktop.ini") or name.startswith("._")


def _write_atomically(out: Path, manifest: dict, arrays: Dict[str, np.ndarray]) -> None:
    """Write into a temporary sibling directory, then swap it in (an old export survives a failure)."""
    import os
    import shutil
    import tempfile

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    # mkdtemp makes a 0700 directory; give the export the mode a plain mkdir
    # would (or the previous export's), so other users and tools can read it.
    if out.exists():
        mode = out.stat().st_mode & 0o7777
    else:                                       # what mkdir gives under the umask (thread-safe probe)
        probe = tmp / "probe"
        probe.mkdir()
        mode = probe.stat().st_mode & 0o7777
        probe.rmdir()
    os.chmod(tmp, mode)
    try:
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=False))
        np.savez_compressed(tmp / "arrays.npz", **arrays)
        if out.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{out.name}.old.", dir=out.parent))
            backup.rmdir()
            os.replace(out, backup)
            try:
                os.replace(tmp, out)
            except BaseException:
                os.replace(backup, out)                  # put the old export back
                raise
            shutil.rmtree(backup, ignore_errors=True)
        else:
            os.replace(tmp, out)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def _graph_preserving_rows_np(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
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
