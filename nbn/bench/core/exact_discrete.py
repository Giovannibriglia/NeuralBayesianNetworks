"""Exact discrete posteriors for the benchmark oracle (D1).

The discrete oracle used to be a rejection filter over a 5,000-row forward
sample: its Monte-Carlo error (TV 0.05-0.07 per query) was larger than the
differences it was meant to resolve. This module computes ``P(T | E = e)``
exactly by variable elimination on the *true* model's CPTs.

It is deliberately self-contained (tables + a small log-space VE) rather than
calling ``nbn.inference.TensorVariableElimination`` or pgmpy's
``VariableElimination``: both are *baselines* of the benchmark, and an oracle
that shares code with a method under test would score that method's bugs as
perfect answers. Correctness is pinned by brute-force enumeration tests.

Sources of tables (``tabular_source``):

* a ``NeuralBayesianNetwork`` true model (synthetic families): each discrete
  node's CPT is read through ``mechanism.forward(parent_configs).probs``, the
  same path the parameter-recovery readout uses;
* a pgmpy ``DiscreteBayesianNetwork`` (bnlearn ``.bif`` networks): the
  ``TabularCPD`` values, axis order = ``state_names`` order, which is exactly
  the integer encoding ``BnlearnProblemSource`` uses for the data.

Elimination is restricted to the ancestral closure of ``{T} ∪ E`` (every other
node is barren and sums to one), the order is greedy min-weight (product of
cardinalities of the eliminated clique), and the plan is checked *before* any
tensor is built: a clique above ``max_entries`` raises :class:`Intractable`
so the caller can fall back to likelihood weighting.
"""
from __future__ import annotations

import heapq
import itertools
import math
from typing import Any, Callable

import torch

__all__ = [
    "Intractable",
    "InvalidEvidence",
    "ZeroProbabilityEvidence",
    "TabularSource",
    "tabular_source",
    "exact_posterior",
    "DEFAULT_MAX_ENTRIES",
]

#: Largest intermediate factor (number of entries) the oracle will build.
#: 2**23 float64 entries = 64 MiB per factor; a product holds ~3 such tensors.
DEFAULT_MAX_ENTRIES = 1 << 23


class Intractable(RuntimeError):
    """The elimination plan needs a factor larger than the configured cap."""


class ZeroProbabilityEvidence(ValueError):
    """``P(evidence) = 0`` under the true model: no conditional exists."""


class InvalidEvidence(ValueError):
    """An evidence state outside ``0..K-1`` of its node (a caller bug)."""


class TabularSource:
    """Lazily materialised CPTs of a discrete (sub)network.

    ``parents(node)`` -> tuple of parent names, ``card(node)`` -> int,
    ``log_table(node)`` -> float64 CPU tensor ``[*parent_cards, K]`` (axes in
    ``parents(node)`` order, last axis = the node). ``is_discrete(node)``
    tells the oracle whether a node can enter an exact computation.
    """

    def __init__(
        self,
        parents: Callable[[str], tuple[str, ...]],
        card: Callable[[str], int | None],
        table: Callable[[str], torch.Tensor],
    ) -> None:
        self._parents = parents
        self._card = card
        self._table = table
        self._cache: dict[str, torch.Tensor] = {}

    def parents(self, node: str) -> tuple[str, ...]:
        return tuple(self._parents(node))

    def card(self, node: str) -> int:
        c = self._card(node)
        if c is None:
            raise KeyError(node)
        return int(c)

    def is_discrete(self, node: str) -> bool:
        return self._card(node) is not None

    def log_table(self, node: str) -> torch.Tensor:
        t = self._cache.get(node)
        if t is None:
            probs = self._table(node).detach().to("cpu", torch.float64)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-300)
            t = torch.log(probs)
            self._cache[node] = t
        return t


def tabular_source(problem: Any) -> TabularSource | None:
    """A :class:`TabularSource` over ``problem.true_model``, or ``None`` when
    the true model is absent or of a type with no tabular CPTs."""
    tm = getattr(problem, "true_model", None)
    if tm is None:
        return None
    variables = problem.variables

    def card(node: str) -> int | None:
        kind, k = variables[node]
        return int(k) if kind == "discrete" else None

    try:
        from nbn.core.network import NeuralBayesianNetwork
    except ImportError:  # pragma: no cover - core always importable
        NeuralBayesianNetwork = ()  # type: ignore[assignment]  # noqa: N806

    if isinstance(tm, NeuralBayesianNetwork):
        def parents(node: str) -> tuple[str, ...]:
            return tuple(tm.dag.parents(node))

        def table(node: str) -> torch.Tensor:
            pa = parents(node)
            k = card(node)
            pcards = [card(p) for p in pa]
            if k is None or any(c is None for c in pcards):
                raise Intractable(f"node {node!r} has a continuous parent")
            mech = tm.mechanisms[node]
            dev = getattr(tm, "device", "cpu")
            with torch.no_grad():
                if not pa:
                    return mech.forward(None).probs.reshape(k)
                configs = torch.tensor(
                    list(itertools.product(*(range(c) for c in pcards))),
                    dtype=torch.long,
                ).to(dev)
                probs = mech.forward(configs).probs
            return probs.reshape(*pcards, k)

        return TabularSource(parents, card, table)

    if hasattr(tm, "get_cpds") and hasattr(tm, "nodes"):
        # pgmpy DiscreteBayesianNetwork (bnlearn .bif).
        def parents(node: str) -> tuple[str, ...]:
            return tuple(tm.get_cpds(node).variables[1:])

        def table(node: str) -> torch.Tensor:
            import numpy as np

            cpd = tm.get_cpds(node)
            values = np.asarray(cpd.values, dtype=np.float64)  # [K, *parents]
            return torch.from_numpy(np.moveaxis(values, 0, -1).copy())

        return TabularSource(parents, card, table)

    return None


# ---------------------------------------------------------------------------
# Variable elimination
# ---------------------------------------------------------------------------

def _ancestral_closure(src: TabularSource, nodes) -> list[str]:
    need, stack = set(), list(nodes)
    while stack:
        nd = stack.pop()
        if nd in need:
            continue
        need.add(nd)
        stack.extend(src.parents(nd))
    return sorted(need)


def _plan(scopes: list[tuple[str, ...]], cards: dict[str, int],
          keep: str, max_entries: int) -> list[str]:
    """Greedy min-weight elimination order over the interaction graph of
    ``scopes``; raises :class:`Intractable` if any clique exceeds the cap."""
    adj: dict[str, set[str]] = {}
    for sc in scopes:
        for v in sc:
            adj.setdefault(v, set()).update(u for u in sc if u != v)
    log_cap = math.log(max_entries)

    def weight(v: str) -> float:
        # sorted(): float summation order must not follow set (hash) order,
        # or near-ties could break differently in different processes.
        return math.log(cards[v]) + sum(math.log(cards[u]) for u in sorted(adj[v]))

    heap = [(weight(v), v) for v in adj if v != keep]
    heapq.heapify(heap)
    eliminated: set[str] = set()
    order: list[str] = []
    while heap:
        w, v = heapq.heappop(heap)
        if v in eliminated:
            continue
        current = weight(v)
        if current > w + 1e-12:  # stale entry
            heapq.heappush(heap, (current, v))
            continue
        if current > log_cap:
            raise Intractable(
                f"elimination clique of {math.exp(current):.3g} entries exceeds "
                f"the oracle cap of {max_entries}"
            )
        nbrs = adj.pop(v)
        for u in nbrs:
            adj[u].discard(v)
            adj[u].update(x for x in nbrs if x != u)
        eliminated.add(v)
        order.append(v)
        for u in nbrs:
            if u != keep:
                heapq.heappush(heap, (weight(u), u))
    return order


def _product(factors: list[tuple[tuple[str, ...], torch.Tensor]]):
    """Log-space product of factors over the union of their scopes."""
    scope: list[str] = []
    for sc, _ in factors:
        for v in sc:
            if v not in scope:
                scope.append(v)
    out = None
    for sc, t in factors:
        pos = [scope.index(v) for v in sc]
        perm = sorted(range(len(sc)), key=lambda i: pos[i])
        shape = [1] * len(scope)
        for i in range(len(sc)):
            shape[pos[i]] = t.shape[i]
        aligned = (t.permute(*perm) if sc else t).reshape(shape)
        out = aligned if out is None else out + aligned
    return tuple(scope), out


def exact_posterior(
    src: TabularSource,
    target: str,
    evidence: dict[str, int],
    *,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> torch.Tensor:
    """Exact ``P(target | evidence)`` as a float64 CPU vector ``[K]``.

    ``evidence`` maps node -> integer state. Raises :class:`Intractable` when
    the ancestral closure of ``{target} ∪ evidence`` contains a continuous
    node or the elimination needs a factor above ``max_entries``, and
    ``ValueError`` when the evidence has probability zero under the model.
    """
    k = src.card(target)
    if target in evidence and not 0 <= int(evidence[target]) < k:
        raise InvalidEvidence(
            f"evidence {target}={evidence[target]} is outside 0..{k - 1}")
    if target in evidence:
        out = torch.zeros(k, dtype=torch.float64)
        out[int(evidence[target])] = 1.0
        return out

    for nd, val in evidence.items():
        kn = src.card(nd) if src.is_discrete(nd) else None
        if kn is not None and not 0 <= int(val) < kn:
            raise InvalidEvidence(
                f"evidence {nd}={val} is outside the node's states 0..{kn - 1}")

    nodes = _ancestral_closure(src, [target, *evidence])
    for nd in nodes:
        if not src.is_discrete(nd):
            raise Intractable(f"node {nd!r} in the ancestral closure is continuous")
    cards = {nd: src.card(nd) for nd in nodes}

    # Symbolic pass first: conditioned scopes, then the plan (cheap), so an
    # intractable query is rejected before any table is materialised.
    raw_scopes = {nd: (*src.parents(nd), nd) for nd in nodes}
    scopes = [tuple(v for v in sc if v not in evidence) for sc in raw_scopes.values()]
    order = _plan(scopes + [(target,)], cards, target, max_entries)

    factors: list[tuple[tuple[str, ...], torch.Tensor]] = []
    for nd in nodes:
        sc = raw_scopes[nd]
        t = src.log_table(nd)
        if tuple(t.shape) != tuple(cards[v] for v in sc):
            raise ValueError(
                f"CPT of {nd!r} has shape {tuple(t.shape)}, expected "
                f"{tuple(cards[v] for v in sc)}"
            )
        idx = tuple(int(evidence[v]) if v in evidence else slice(None) for v in sc)
        t = t[idx]
        factors.append((tuple(v for v in sc if v not in evidence), t))

    for v in order:
        touching = [f for f in factors if v in f[0]]
        if not touching:
            continue
        factors = [f for f in factors if v not in f[0]]
        sc, t = _product(touching)
        ax = sc.index(v)
        factors.append((sc[:ax] + sc[ax + 1:], torch.logsumexp(t, dim=ax)))

    vec = torch.zeros(k, dtype=torch.float64)
    log_const = torch.zeros((), dtype=torch.float64)
    for sc, t in factors:
        if not sc:
            # Scalar: a normalising constant for the target, but -inf means a
            # factor disconnected from the target made the evidence impossible
            # (e.g. two observed nodes inconsistent with their CPTs). Skipping
            # it returned a confident posterior for P(e) = 0.
            log_const = log_const + t
            continue
        if sc != (target,):  # pragma: no cover - plan guarantees this
            raise RuntimeError(f"unexpected residual scope {sc}")
        vec = vec + t
    if not torch.isfinite(torch.logsumexp(vec, 0) + log_const):
        raise ZeroProbabilityEvidence(
            "evidence has zero probability under the true model")
    return torch.softmax(vec, dim=0)
