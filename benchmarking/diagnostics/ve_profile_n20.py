"""v0.6b round-1 diagnostic: localise the 4 GiB allocation in
``TensorVariableElimination.query_batch`` for the v0.5b §A.5 OOM case.

Runs on cpu (where the same allocation pattern occurs but the 16 GiB
host allocator absorbs it).  Produces structured findings:

* the cached elimination plan (what order ``_plan`` returns),
* a step-by-step shape progression through the elimination loop
  (factor scopes after each contraction; B-axis presence),
* peak allocation point: op, input shapes, dtype, MiB,
* tracemalloc + torch.profiler peaks,
* hypothesis routing (A/B/C/Other) with the matching signature.

This script is the v0.6b round-1 deliverable.  It does not patch
anything; it produces evidence to route the round-2 fix.

Run:
    python -m benchmarking.diagnostics.ve_profile_n20

Output:
    docs/v0.6b/profiler_trace.json   (raw findings)
    stdout                           (human-readable summary)
"""
from __future__ import annotations

import json
import pathlib
import tracemalloc
import traceback
from typing import Any, Dict, List, Tuple

import torch
import torch.profiler

from benchmarking.synthetic import make_synthetic_bn
from nbn.inference.tensor_ve import (
    TensorVariableElimination,
    _condition_factor_batched,
    _log_factor_product_batched,
)


# ---------------------------------------------------------------------- #
# Min-fill elimination order (prototype for round-2's
# nbn/inference/_elimination_order.py).  Pure-Python graph algorithm,
# no torch.  Operates on the moralised + evidence-pruned undirected
# graph.
# ---------------------------------------------------------------------- #


def _moralise_and_prune(
    model, evidence_keys: Tuple[str, ...],
) -> Dict[str, set]:
    """Build the undirected moralised graph of ``model.dag``, then prune
    evidence variables (connecting their neighbours pairwise so the
    residual graph still contains every induced clique that would have
    appeared during elimination)."""
    adj: Dict[str, set] = {n: set() for n in model.dag.topological_order()}
    for n in adj:
        for p in model.dag.parents(n):
            adj[p].add(n)
            adj[n].add(p)
        # Moralise: connect every pair of parents of n.
        ps = model.dag.parents(n)
        for i, p1 in enumerate(ps):
            for p2 in ps[i + 1:]:
                adj[p1].add(p2)
                adj[p2].add(p1)
    # Prune evidence: remove each evidence node, fully connecting its
    # remaining neighbours so any clique that would have formed during
    # ordinary elimination is preserved.
    for ev in evidence_keys:
        if ev not in adj:
            continue
        nbrs = list(adj[ev])
        for i, a in enumerate(nbrs):
            for b in nbrs[i + 1:]:
                adj[a].add(b)
                adj[b].add(a)
        for nbr in nbrs:
            adj[nbr].discard(ev)
        del adj[ev]
    return adj


def min_fill_order(
    model, target: str, evidence_keys: Tuple[str, ...],
) -> List[str]:
    """Min-fill elimination order (Kjaerulff 1990).

    At each step pick the non-target variable whose elimination would
    add the fewest fill-in edges to the residual graph; ties broken by
    fewest neighbours then by name (deterministic).  The variable is
    eliminated by adding any required fill-in edges to make its
    neighbours a clique, then removing it from the graph.

    Returns a permutation of ``set(model.dag.topological_order()) -
    {target} - set(evidence_keys)``.
    """
    adj = _moralise_and_prune(model, evidence_keys)
    candidates = set(adj.keys()) - {target}
    order: List[str] = []
    while candidates:
        best_var = None
        best_key: Tuple[int, int, str] | None = None
        for v in candidates:
            nbrs = adj[v] & candidates
            nbrs_list = list(nbrs)
            fill = 0
            for i, a in enumerate(nbrs_list):
                for b in nbrs_list[i + 1:]:
                    if b not in adj[a]:
                        fill += 1
            key = (fill, len(nbrs_list), v)
            if best_key is None or key < best_key:
                best_key = key
                best_var = v
        assert best_var is not None
        order.append(best_var)
        nbrs = list(adj[best_var] & candidates)
        for i, a in enumerate(nbrs):
            for b in nbrs[i + 1:]:
                adj[a].add(b)
                adj[b].add(a)
        for nbr in list(adj[best_var]):
            adj[nbr].discard(best_var)
        del adj[best_var]
        candidates.discard(best_var)
    return order


def validate_elimination_order(
    model, target: str, evidence_keys: Tuple[str, ...], order: List[str],
) -> Dict[str, Any]:
    """Sanity-check an elimination order.  Round-2 will replicate this
    in tests/unit/test_min_fill_order.py."""
    all_vars = set(model.dag.topological_order())
    expected = all_vars - {target} - set(evidence_keys)
    actual = set(order)

    return {
        "is_permutation_of_expected": actual == expected,
        "missing_from_order": sorted(expected - actual),
        "spurious_in_order": sorted(actual - expected),
        "target_in_order": target in actual,
        "any_evidence_in_order": any(e in actual for e in evidence_keys),
        "len_order": len(order),
        "len_expected": len(expected),
        "no_duplicates": len(order) == len(actual),
    }



# ---------------------------------------------------------------------- #
# Algebraic-shape walk (no torch ops yet — pure dimensional analysis)
# ---------------------------------------------------------------------- #


def _walk_elimination_shapes(
    model, target: str, evidence_keys: Tuple[str, ...], B: int,
    *, plan: List[str] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Walk a given elimination plan algebraically and report the
    shape of each intermediate factor product/marginalisation.

    If ``plan`` is None, uses the engine's cached topological-order
    plan via ``_plan``.  Pass an explicit plan (e.g. min-fill) to
    compare alternative orderings on the same DAG.

    Does not run torch — it computes ``len(scope)`` and ``prod(cards)``
    so we can identify the worst-case allocation purely from the plan.
    """
    eng = TensorVariableElimination()
    factors = eng._extract_factors(model)
    if plan is None:
        plan = eng._plan(model, target, evidence_keys)

    # Each entry: (vars_set, has_batch).  cardinality is uniformly K.
    state: List[Tuple[set, bool]] = []
    for node, f in factors.items():
        vars_ = set(f.variables)
        has_b = bool(vars_ & set(evidence_keys))
        state.append((vars_, has_b))

    K = next(iter(factors.values())).cardinalities.get(target, 4)

    steps: List[Dict[str, Any]] = []
    peak_step: Dict[str, Any] | None = None

    for step_i, var in enumerate(plan):
        relevant = [s for s in state if var in s[0]]
        rest = [s for s in state if var not in s[0]]
        if not relevant:
            steps.append({
                "step": step_i, "var": var, "n_relevant": 0,
                "skipped": True,
            })
            continue

        # Product scope is the union of all relevant scopes.
        prod_vars = set().union(*(s[0] for s in relevant))
        prod_has_b = any(s[1] for s in relevant)

        n_axes = len(prod_vars)
        elements = K ** n_axes
        if prod_has_b:
            elements *= B
        bytes_ = elements * 4  # float32

        info = {
            "step": step_i,
            "var": var,
            "n_relevant_factors": len(relevant),
            "product_scope_size": n_axes,
            "product_scope_vars": sorted(prod_vars),
            "product_has_b": prod_has_b,
            "product_elements": elements,
            "product_mib": bytes_ / 1024 ** 2,
        }
        steps.append(info)

        if peak_step is None or info["product_mib"] > peak_step["product_mib"]:
            peak_step = info

        # Marginalise `var` out: scope drops by one
        prod_vars.discard(var)
        # Update state: drop relevant, add the marginalised product
        state = rest + [(prod_vars, prod_has_b)]

    summary = {
        "total_steps": sum(1 for s in steps if not s.get("skipped")),
        "peak_step_index": peak_step["step"] if peak_step else None,
        "peak_var_eliminated": peak_step["var"] if peak_step else None,
        "peak_product_scope_size": peak_step["product_scope_size"] if peak_step else None,
        "peak_product_mib": peak_step["product_mib"] if peak_step else None,
        "peak_product_has_b": peak_step["product_has_b"] if peak_step else None,
        "K": K,
    }
    return steps, summary


# ---------------------------------------------------------------------- #
# Real-execution profile
# ---------------------------------------------------------------------- #


def _profile_query_batch(
    bn, target: str, evidence: Dict[str, torch.Tensor], device: str,
) -> Dict[str, Any]:
    eng = TensorVariableElimination()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    error: Dict[str, Any] | None = None
    top_10: List[Dict[str, Any]] = []
    peak_event: Dict[str, Any] = {}

    tracemalloc.start()
    try:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            with torch.profiler.record_function("query_batch_n20"):
                _ = eng.query_batch(bn.true_model, [target], evidence)
    except Exception as exc:
        error = {
            "type": type(exc).__name__,
            "msg": str(exc),
            "traceback": traceback.format_exc(),
        }

    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if error is None:
        events = prof.key_averages(group_by_input_shape=True)
        sorted_events = sorted(
            events,
            key=lambda e: (
                getattr(e, "self_cuda_memory_usage", 0)
                + getattr(e, "self_cpu_memory_usage", 0)
            ),
            reverse=True,
        )
        for ev in sorted_events[:10]:
            top_10.append({
                "op": ev.key,
                "shape": str(getattr(ev, "input_shapes", None))[:240],
                "self_cpu_mib": getattr(ev, "self_cpu_memory_usage", 0) / 1024 ** 2,
                "self_cuda_mib": getattr(ev, "self_cuda_memory_usage", 0) / 1024 ** 2,
                "count": ev.count,
            })
        if top_10:
            peak_event = top_10[0]

    out: Dict[str, Any] = {
        "error": error,
        "tracemalloc_peak_mib": peak / 1024 ** 2,
        "peak_event": peak_event,
        "top_10_by_memory": top_10,
    }
    if device == "cuda":
        out["cuda_max_allocated_mib"] = (
            torch.cuda.max_memory_allocated() / 1024 ** 2
        )
    return out


# ---------------------------------------------------------------------- #
# Manual step-by-step factor shape walk (real torch tensors)
# ---------------------------------------------------------------------- #


def _real_shapes_walk(
    bn, target: str, evidence: Dict[str, torch.Tensor], device: str,
    cap_mib: float = 1024.0,
) -> List[Dict[str, Any]]:
    """Re-run the elimination loop from query_batch but record actual
    tensor shapes after each contraction, stopping early if a single
    intermediate exceeds ``cap_mib``."""
    eng = TensorVariableElimination()
    model = bn.true_model
    factors = eng._extract_factors(model)

    ev_norm: Dict[str, torch.Tensor] = {}
    B = 1
    for k, v in evidence.items():
        t = v.to(device=device, dtype=torch.long)
        ev_norm[k] = t
        B = max(B, t.shape[0])
    ev_norm = {
        k: (v.expand(B) if v.shape[0] == 1 else v) for k, v in ev_norm.items()
    }

    conditioned = []
    for f in factors.values():
        lv, vars_, has_b = _condition_factor_batched(
            f.log_values, f.variables, ev_norm,
        )
        conditioned.append((lv, vars_, has_b))

    plan = eng._plan(model, target, tuple(sorted(ev_norm.keys())))
    walk: List[Dict[str, Any]] = []

    for step_i, var in enumerate(plan):
        relevant = [f for f in conditioned if var in f[1]]
        rest = [f for f in conditioned if var not in f[1]]
        if not relevant:
            walk.append({"step": step_i, "var": var, "skipped": True})
            continue

        try:
            prod_lv, prod_vars, prod_has_b = relevant[0]
            for lv2, vars2, has_b2 in relevant[1:]:
                prod_lv, prod_vars, prod_has_b = _log_factor_product_batched(
                    prod_lv, prod_vars, prod_has_b, lv2, vars2, has_b2,
                )
                if prod_lv.numel() * prod_lv.element_size() / 1024 ** 2 > cap_mib:
                    walk.append({
                        "step": step_i, "var": var,
                        "stage": "during_product_chain",
                        "shape": list(prod_lv.shape),
                        "scope": prod_vars,
                        "has_b": prod_has_b,
                        "mib": prod_lv.numel() * prod_lv.element_size() / 1024 ** 2,
                        "dtype": str(prod_lv.dtype),
                        "aborted": True,
                    })
                    return walk
            mib = prod_lv.numel() * prod_lv.element_size() / 1024 ** 2
            walk.append({
                "step": step_i, "var": var,
                "stage": "after_product_before_marginalise",
                "shape": list(prod_lv.shape),
                "scope": list(prod_vars),
                "has_b": bool(prod_has_b),
                "mib": mib,
                "dtype": str(prod_lv.dtype),
                "n_relevant": len(relevant),
            })

            dim = prod_vars.index(var) + (1 if prod_has_b else 0)
            prod_lv = torch.logsumexp(prod_lv, dim=dim)
            prod_vars = [v for v in prod_vars if v != var]
            conditioned = rest + [(prod_lv, prod_vars, prod_has_b)]
        except Exception as exc:
            walk.append({
                "step": step_i, "var": var,
                "error": f"{type(exc).__name__}: {exc}",
                "aborted": True,
            })
            return walk

    return walk


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #


def main() -> None:
    print("=== v0.6b round-1 diagnostic — nbn_ve OOM at n_nodes=20 ===")
    print()

    seed = 0
    device = "cpu"
    bn = make_synthetic_bn(
        family="discrete",
        n_nodes=20,
        edge_density=0.20,
        max_in_degree=4,
        cardinality=4,
        seed=seed,
        device=device,
        n_train=2000,
        n_test=500,
        n_reference=5000,
    )

    target = bn.column_order[5]
    evidence_names = [bn.column_order[0], bn.column_order[3]]
    B = 16
    evidence = {
        name: torch.zeros(B, dtype=torch.long, device=device)
        for name in evidence_names
    }

    eng = TensorVariableElimination()
    plan = eng._plan(bn.true_model, target, tuple(sorted(evidence_names)))

    # Algebraic shape walk — naive (current) plan
    alg_steps, alg_summary = _walk_elimination_shapes(
        bn.true_model, target, tuple(sorted(evidence_names)), B,
    )

    # ---- v0.6b round-1 follow-up: min-fill ordering verification ----
    mf_order = min_fill_order(
        bn.true_model, target, tuple(sorted(evidence_names)),
    )
    mf_validation = validate_elimination_order(
        bn.true_model, target, tuple(sorted(evidence_names)), mf_order,
    )
    mf_steps, mf_summary = _walk_elimination_shapes(
        bn.true_model, target, tuple(sorted(evidence_names)), B,
        plan=mf_order,
    )

    naive_plan = plan  # noqa: F821 — `plan` is bound above when the engine's _plan is called
    side_by_side = []
    for i in range(max(len(naive_plan), len(mf_order))):
        naive_step = next(
            (s for s in alg_steps if s["step"] == i and not s.get("skipped")),
            None,
        )
        mf_step = next(
            (s for s in mf_steps if s["step"] == i and not s.get("skipped")),
            None,
        )
        side_by_side.append({
            "step": i,
            "naive_var": naive_plan[i] if i < len(naive_plan) else None,
            "naive_scope": naive_step["product_scope_size"] if naive_step else None,
            "naive_mib": naive_step["product_mib"] if naive_step else None,
            "minfill_var": mf_order[i] if i < len(mf_order) else None,
            "minfill_scope": mf_step["product_scope_size"] if mf_step else None,
            "minfill_mib": mf_step["product_mib"] if mf_step else None,
        })

    # Real shape walk (caps a single product at 1 GiB to avoid OOM
    # on whatever sandbox we're on; if it aborts that's itself a
    # finding that matches the OOM signature).
    real_walk = _real_shapes_walk(bn, target, evidence, device, cap_mib=2048.0)

    # Real profile (only if real walk completes; otherwise the
    # profiler will OOM on this sandbox too).
    aborted = any(s.get("aborted") for s in real_walk)
    if aborted:
        profile = {
            "skipped_reason": (
                "real_walk aborted at >2 GiB intermediate; running full "
                "query_batch under torch.profiler would OOM the sandbox"
            ),
        }
    else:
        profile = _profile_query_batch(bn, target, evidence, device)

    findings = {
        "config": {
            "n_nodes": 20,
            "cardinality": 4,
            "max_in_degree": 4,
            "B": B,
            "device": device,
            "seed": seed,
            "target": target,
            "evidence_names": evidence_names,
            "edge_density": 0.20,
        },
        "plan": {
            "type": "topological elimination order (naive)",
            "len": len(plan),
            "first_5": plan[:5],
            "last_5": plan[-5:],
            "full": plan,
        },
        "algebraic_shape_walk": {
            "summary": alg_summary,
            "steps": alg_steps,
        },
        "min_fill_comparison": {
            "min_fill_order": mf_order,
            "validation": mf_validation,
            "summary_minfill": mf_summary,
            "summary_naive": alg_summary,
            "side_by_side": side_by_side,
            "peak_reduction_factor": (
                alg_summary["peak_product_mib"] / mf_summary["peak_product_mib"]
                if mf_summary["peak_product_mib"] and mf_summary["peak_product_mib"] > 0
                else None
            ),
        },
        "real_shape_walk": {
            "aborted_at_cap": aborted,
            "n_steps_completed": len(real_walk),
            "steps": real_walk,
        },
        "torch_profile": profile,
    }

    out_dir = pathlib.Path("docs/v0.6b")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "profiler_trace.json"
    out_path.write_text(json.dumps(findings, indent=2, default=str))

    # Human-readable summary
    print("--- Algebraic peak (pure dimensional analysis) ---")
    print(f"  total elim steps: {alg_summary['total_steps']}")
    print(f"  peak step #{alg_summary['peak_step_index']}: "
          f"eliminate '{alg_summary['peak_var_eliminated']}'")
    print(f"  peak scope size: {alg_summary['peak_product_scope_size']} vars "
          f"(K={alg_summary['K']}, B-axis={alg_summary['peak_product_has_b']})")
    print(f"  peak intermediate: {alg_summary['peak_product_mib']:.1f} MiB "
          f"({alg_summary['peak_product_mib']/1024:.2f} GiB)")
    print()
    print("--- Real walk (executed up to the cap) ---")
    for s in real_walk[:8]:
        if s.get("skipped"):
            continue
        if s.get("error"):
            print(f"  step {s['step']:>2} elim '{s['var']}': ERROR {s['error']}")
            continue
        print(
            f"  step {s['step']:>2} elim '{s['var']}': "
            f"|scope|={len(s['scope']):>2}  has_b={s['has_b']!s:5}  "
            f"shape={s['shape']}  {s['mib']:.1f} MiB"
        )
    if aborted:
        last = real_walk[-1]
        print(f"  ABORTED at step {last['step']} ('{last['var']}'): "
              f"{last.get('mib', '?')} MiB > 2048 MiB cap")
        print(f"  shape={last.get('shape')}  scope={last.get('scope')}")

    print()
    print("--- Min-fill ordering verification ---")
    print(f"  validation: {mf_validation}")
    print(f"  min-fill peak step: #{mf_summary['peak_step_index']} "
          f"eliminate '{mf_summary['peak_var_eliminated']}'  "
          f"|scope|={mf_summary['peak_product_scope_size']}  "
          f"{mf_summary['peak_product_mib']:.3f} MiB")
    print(f"  naive    peak step: #{alg_summary['peak_step_index']} "
          f"eliminate '{alg_summary['peak_var_eliminated']}'  "
          f"|scope|={alg_summary['peak_product_scope_size']}  "
          f"{alg_summary['peak_product_mib']:.1f} MiB")
    if (mf_summary["peak_product_mib"]
            and mf_summary["peak_product_mib"] > 0):
        ratio = (alg_summary["peak_product_mib"]
                 / mf_summary["peak_product_mib"])
        print(f"  peak reduction: {ratio:.1f}× "
              f"(naive / min-fill)")

    print()
    print("--- Side-by-side per-step (naive vs min-fill) ---")
    print(f"  {'step':>4}  {'naive_var':>9} {'naive_scope':>11} {'naive_mib':>11}"
          f"   |  {'mf_var':>6} {'mf_scope':>8} {'mf_mib':>10}")
    for s in side_by_side:
        nv = s["naive_var"] or "—"
        mv = s["minfill_var"] or "—"
        ns = s["naive_scope"]
        ms = s["minfill_scope"]
        nm = s["naive_mib"]
        mm = s["minfill_mib"]
        ns_s = "—" if ns is None else str(ns)
        ms_s = "—" if ms is None else str(ms)
        nm_s = "—" if nm is None else f"{nm:.3f}"
        mm_s = "—" if mm is None else f"{mm:.3f}"
        print(f"  {s['step']:>4}  {nv:>9} {ns_s:>11} {nm_s:>11}   |  "
              f"{mv:>6} {ms_s:>8} {mm_s:>10}")

    print()
    print(f"--- Saved findings to {out_path} ---")


if __name__ == "__main__":
    main()
