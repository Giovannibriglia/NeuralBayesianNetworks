from __future__ import annotations

import logging
from typing import Dict, List

import networkx as nx
import torch

from nbn.inference._memory import is_oom_error, release_cached_memory
from nbn.inference.base import InferenceEngine
from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
from nbn.inference.tensor_ve import MPSTensorLimitError, TensorVariableElimination

logger = logging.getLogger(__name__)


def _batched(*dicts) -> bool:
    """True when some evidence / do value holds more than one row."""
    for d in dicts:
        for v in (d or {}).values():
            if torch.as_tensor(v).numel() > 1:
                return True
    return False


class HybridRouter(InferenceEngine):
    """Auto-selecting inference engine.

    Routing logic:

    1. Any continuous mechanism → LW.
    2. Several discrete targets → exact joint VE (``[K_1, …, K_n]`` per
       row) when the joint table has at most ``max_joint_elements``
       entries, else LW's ``(weights, samples)`` particles (R6; a
       multi-target query used to crash VE, B6).
    3. Induced treewidth > τ → LW (a cheap structural pre-filter).  With
       ``query_width_check=True``, a network over τ is given a second look:
       the width of the query's own pruned plan decides (a query touching a
       small part of a large dense network can still be answered exactly).
    4. Otherwise price the *actual* VE plan of this query: its estimated
       peak bytes for one row against the device memory budget (VE splits
       larger batches into rows-chunks that fit).  Fits → exact
       ``TensorVariableElimination``; does not → LW.  The unweighted
       treewidth alone ignored cardinalities: a 40-node, in-degree-5,
       card-4 network had width 21 (→ VE) and an estimated 3.3 TB peak (B5).
       On CPU / MPS the budget is ``ve_host_fraction`` of the *stable*
       memory capacity, so the route does not flip with momentary memory
       pressure; on CUDA it is the live headroom (see
       ``TensorVariableElimination.memory_budget_bytes``).
    5. If VE still runs out of memory on any device (CUDA OOM, MPS
       "out of memory", CPU allocator failure), the query is retried with LW.

    Every downgrade from exact VE to approximate LW for a discrete query is
    logged at WARNING, once per query shape, and ``last_engine`` names the
    engine that answered the latest query.

    Parameters
    ----------
    treewidth_threshold: int
        Treewidth limit above which we fall back to LW.
    n_samples: int
        Number of samples for the likelihood-weighting fallback.
    memory_budget: float, optional
        Hard cap in bytes on VE's estimated peak; combined with the device
        default (see ``TensorVariableElimination.memory_budget_bytes``).
    max_joint_elements: int
        Largest joint table (product of the target cardinalities) a
        multi-target discrete query is answered with exactly.
    ve_host_fraction: float
        On CPU / MPS, route to VE only when one row's estimated peak fits
        this fraction of the stable memory capacity (leaving room for the
        rest of the process).  Called directly, VE itself rejects only plans
        beyond the whole capacity.
    query_width_check: bool
        When the whole network's width exceeds ``treewidth_threshold``, also
        measure the width of the query's own pruned plan and route to exact
        VE if that is within the threshold (and the plan fits memory).  More
        answers become exact; some of them can be slow.  Default False keeps
        the historical routing.
    fallback: str
        What answers a discrete query that exact VE cannot: ``"lw"`` (default,
        plain likelihood weighting, no guarantee) or ``"certified"``
        (``certified_query``: the same point estimate plus simultaneous
        ``1 - certified_delta`` intervals in ``last_certificate``; one discrete
        target, discrete evidence; other queries still use LW), or
        ``"condition"``: answer a single-target discrete query that VE refuses
        (memory, treewidth threshold, MPS tensor limits) *exactly* by
        conditioning (``TensorVariableElimination(on_oom="condition")``):
        slower, same answer.  If conditioning would need more than
        ``condition_max_instantiations`` instantiations, LW answers.
    """

    # Bound on remembered "already warned" query shapes.
    _MAX_WARNED = 1024

    def __init__(
        self,
        treewidth_threshold: int = 25,
        n_samples: int = 2048,
        memory_budget: float | None = None,
        ve_host_fraction: float = 0.75,
        max_joint_elements: int = 1 << 20,
        query_width_check: bool = False,
        fallback: str = "lw",
        certified_samples: int = 200_000,
        certified_delta: float = 0.05,
        condition_max_instantiations: int = 1_000_000,
    ) -> None:
        if fallback not in ("lw", "certified", "condition"):
            raise ValueError("fallback must be 'lw', 'certified' or 'condition'.")
        self.fallback = fallback
        self._cert = None
        if fallback == "certified":
            from nbn.inference.certified import CertifiedEngine
            self._cert = CertifiedEngine(certified_samples, certified_delta)
        self.treewidth_threshold = treewidth_threshold
        self.query_width_check = bool(query_width_check)
        self.max_joint_elements = int(max_joint_elements)
        self.ve_host_fraction = float(ve_host_fraction)
        self._ve = TensorVariableElimination(
            treewidth_threshold, memory_budget=memory_budget,
        )
        self._cond = None
        if fallback == "condition":
            self._cond = TensorVariableElimination(
                treewidth_threshold, memory_budget=memory_budget, on_oom="condition",
                max_conditioning=condition_max_instantiations,
            )
        self._lw = LikelihoodWeightingEngine(n_samples)
        self._last_engine: str | None = None
        self._warned: set = set()

    @property
    def last_engine(self) -> str | None:
        """Who answered last: ``"tensor_ve"``, ``"conditioning"``, ``"certified"``
        or ``"likelihood_weighting"``."""
        return self._last_engine

    def _warn_once(self, key, msg: str, *args) -> None:
        """WARNING the first time a query shape is downgraded, DEBUG after.

        A silent downgrade from exact to approximate answers is a behaviour
        change the caller must be able to see; repeating it per row of a
        sweep would drown the log.
        """
        if getattr(self, "fallback", "lw") == "certified":
            # The certified engine answers next when it applies; say so.
            msg = msg.replace(
                "with likelihood weighting (approximate).",
                "with the certified fallback (intervals at the declared level; "
                "plain likelihood weighting if the query cannot be certified).")
        if key in self._warned:
            logger.debug(msg, *args)
            return
        if len(self._warned) >= self._MAX_WARNED:
            self._warned.clear()
        self._warned.add(key)
        logger.warning(msg, *args)

    def _use_lw(self, reason: str, *args, warn_key=None) -> InferenceEngine:
        if self.__dict__.get("_cond") is not None:
            # fallback="condition": conditioning is tried next and warns
            # itself if it is refused, so this is not (yet) a downgrade.
            warn_key = None
        if warn_key is None:
            logger.debug(reason, *args)
        else:
            self._warn_once(warn_key, reason, *args)
        self._last_engine = "likelihood_weighting"
        return self._lw

    def _select(
        self,
        model,
        targets: List[str] | None = None,
        evidence: Dict[str, torch.Tensor] | None = None,
        do: Dict[str, torch.Tensor] | None = None,
        order: str = "min_fill",
    ) -> InferenceEngine:
        multi = targets is not None and len(targets) != 1
        all_discrete = all(
            m.is_discrete for m in model.mechanisms.values()
        )
        if not all_discrete:
            self._last_engine = "likelihood_weighting"
            return self._lw
        if multi:
            size = 1
            for t in targets:
                size *= int(model.mechanisms[t].n_classes)
            if not targets or size > self.max_joint_elements:
                return self._use_lw(
                    "HybridRouter: the joint table over %s has %d entries "
                    "(> max_joint_elements=%d); answering with likelihood-"
                    "weighting particles instead of an exact joint.",
                    list(targets), size, self.max_joint_elements,
                    warn_key=("joint",) + _shape_key(model, targets, evidence, do),
                )
        try:
            tw = model.dag.induced_width()
        except (nx.NetworkXError, ValueError):
            # Treewidth could not be computed for this graph (a malformed DAG
            # surfaces as nx.NetworkXError; a degenerate greedy min-fill step as
            # ValueError) -> fall back to LW. Narrowed from a bare `except`
            # (#232): the bare catch silently masked the induced_width bugs for
            # months (#226/#231), degrading exact VE to stochastic LW. Other
            # exceptions (AttributeError / TypeError / RuntimeError) signal real
            # bugs and must propagate, not silently downgrade the engine.
            tw = self.treewidth_threshold + 1

        if (tw > self.treewidth_threshold and targets is not None
                and getattr(self, "query_width_check", False)):
            # The global width is a property of the whole network; the query
            # only eliminates its own pruned scope.  Before giving up on
            # exactness, measure that scope's plan width (cheap: the plan is
            # cached and reused by VE if it is chosen).  The memory check
            # below still prices the plan.
            try:
                qw = self._ve.plan_width(
                    model, tuple(targets) if multi else targets[0],
                    tuple(evidence or {}), tuple(do or {}), order=order)
            except (nx.NetworkXError, ValueError, KeyError):
                qw = None
            if qw is not None and qw <= self.treewidth_threshold:
                tw = qw
        if tw > self.treewidth_threshold:
            return self._use_lw(
                "HybridRouter: induced treewidth %d > threshold %d for %s; "
                "answering with likelihood weighting (approximate).",
                tw, self.treewidth_threshold,
                _describe(targets, evidence, do),
                warn_key=("tw",) + _shape_key(model, targets, evidence, do),
            )
        if targets is not None and not self._ve_fits(
            model, tuple(targets) if multi else targets[0], evidence, do, order,
        ):
            return self._use_lw(
                "HybridRouter: the exact VE plan for %s would exceed the %s "
                "memory budget (or, on MPS, the largest tensor MPS reduces "
                "correctly); answering with likelihood weighting "
                "(approximate).", _describe(targets, evidence, do), model.device,
                warn_key=_shape_key(model, targets, evidence, do),
            )
        self._last_engine = "tensor_ve"
        return self._ve

    def _ve_fits(self, model, target, evidence, do, order) -> bool:
        """Whether one row of VE's plan for this query fits the memory budget.

        ``target`` is a name, or a tuple of names for a joint query.
        """
        evidence, do = evidence or {}, do or {}
        names = (target,) if isinstance(target, str) else tuple(target)
        if all(t in evidence or t in do for t in names):
            return True  # answered as a point mass, no elimination
        return self._ve.fits_memory(
            model, target, tuple(evidence), tuple(do), order=order,
            host_fraction=self.ve_host_fraction,
        )

    def _dispatch(self, method: str, model, targets, evidence, kwargs):
        """Route to the selected engine with a VE→LW out-of-memory safety net.

        VE's memory guard (and the allocators themselves) surface an
        over-budget plan as an out-of-memory error — ``torch.OutOfMemoryError``
        on CUDA and from the guard on every device, a ``RuntimeError`` with
        an "out of memory" / "can't allocate memory" message on MPS and CPU.
        For the router that must not be terminal: LW answers the same query
        in bounded memory, so degrade to it (with a warning) instead of
        failing the query.  Only OOM is caught — any other exception signals
        a real bug and propagates (#232 precedent: broad catches here
        silently masked engine bugs).
        """
        if not targets:
            raise ValueError(f"targets must be distinct node names, got {targets!r}.")
        engine = self._select(
            model, targets, evidence, kwargs.get("do"),
            kwargs.get("order", "min_fill"),
        )
        if engine is self._lw and self._certify(model, targets, evidence, kwargs, method):
            engine = self._cert
            self._last_engine = "certified"
        if engine is self._lw and self._conditionable(model, targets):
            out = self._try_condition(method, model, targets, evidence, kwargs)
            if out is not None:
                return out
        try:
            call_kwargs = kwargs
            if engine is self._ve:
                # VE sizes its guard and chunks with the router's share of host
                # memory, the same share the route was priced with.
                call_kwargs = {**kwargs, "host_fraction": self.ve_host_fraction}
            return getattr(engine, method)(model, targets, evidence, **call_kwargs)
        except Exception as exc:
            # An MPS plan past the reliable tensor size is refused by VE; like
            # an OOM, LW can still answer it.
            if engine is self._lw or engine is self.__dict__.get("_cert") or not (
                    is_oom_error(exc) or isinstance(exc, MPSTensorLimitError)):
                raise  # LW itself failed, or not a resource limit at all.
            oom = type(exc).__name__
        # Retry OUTSIDE the except block: inside it the exception's traceback
        # keeps VE's frames — and every intermediate tensor they hold — alive,
        # so LW would run on a device still full of the failed attempt.  With
        # the exception gone, hand the cached blocks back before retrying.
        device = getattr(model, "device", None)
        if device is not None:
            release_cached_memory(device)
        self._warn_once(
            ("oom",) + _shape_key(model, targets, evidence, kwargs.get("do")),
            "HybridRouter: exact VE ran out of memory on %s (%s); answering "
            "with likelihood weighting (approximate).", device, oom,
        )
        if self._certify(model, targets, evidence, kwargs, method):
            self._last_engine = "certified"
            return getattr(self._cert, method)(model, targets, evidence, **kwargs)
        if self._conditionable(model, targets):
            out = self._try_condition(method, model, targets, evidence, kwargs)
            if out is not None:
                return out
        self._last_engine = "likelihood_weighting"
        return getattr(self._lw, method)(model, targets, evidence, **kwargs)

    def _conditionable(self, model, targets) -> bool:
        cond = self.__dict__.get("_cond")
        return (cond is not None and targets is not None and len(targets) == 1
                and all(m.is_discrete for m in model.mechanisms.values()))

    def _try_condition(self, method, model, targets, evidence, kwargs):
        """Exact answer by conditioning, or ``None`` if it is refused."""
        from nbn.inference.conditioning import ConditioningLimitError

        try:
            out = getattr(self._cond, method)(
                model, targets, evidence,
                **{**kwargs, "host_fraction": self.ve_host_fraction})
        except ConditioningLimitError as exc:
            self._warn_once(
                ("cond",) + _shape_key(model, targets, evidence, kwargs.get("do")),
                "HybridRouter: conditioning refused (%s); answering with the "
                "fallback sampler (approximate).", str(exc)[:200])
            return None
        except Exception as exc:  # noqa: BLE001 - only allocator OOM is a fallback
            if not (is_oom_error(exc) or isinstance(exc, MPSTensorLimitError)):
                raise
            self._warn_once(
                ("cond-oom",) + _shape_key(model, targets, evidence, kwargs.get("do")),
                "HybridRouter: conditioning ran out of memory (%s); answering with the "
                "fallback sampler (approximate).", str(exc)[:200])
            return None
        self._last_engine = ("conditioning" if self._cond.last_conditioning is not None
                             else "tensor_ve")
        return out

    def _certify(self, model, targets, evidence, kwargs, method="query") -> bool:
        cert = getattr(self, "_cert", None)
        if cert is None:
            return False
        if method == "query" and _batched(evidence, kwargs.get("do")):
            # A certificate is per row; batched rows go through query_batch.
            # Plain LW answers a batched ``query`` as before (no certificate).
            return False
        return cert.applicable(model, targets, evidence, kwargs.get("do"))

    @property
    def last_certificate(self):
        """The interval answer of the last query if the certified engine
        answered it (``None`` otherwise; after ``query_batch``, the last row)."""
        cert = getattr(self, "_cert", None)
        if cert is None or getattr(self, "_last_engine", None) != "certified":
            return None
        return cert.last_certificate

    def query(
        self,
        model,
        targets: List[str],
        evidence: Dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._dispatch("query", model, targets, evidence, kwargs)

    def query_batch(
        self,
        model,
        targets: List[str],
        evidence: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        return self._dispatch("query_batch", model, targets, evidence, kwargs)


def _shape_key(model, targets, evidence, do) -> tuple:
    """Query shape for log de-duplication (values do not matter)."""
    return (
        id(model), tuple(targets or ()),
        tuple(sorted(evidence or {})), tuple(sorted(do or {})),
    )


def _describe(targets, evidence, do) -> str:
    """``P(A, B | E1, E2; do(X))`` — names only, interventions kept apart."""
    given = [", ".join(sorted(evidence))] if evidence else []
    if do:
        given.append(f"do({', '.join(sorted(do))})")
    head = ", ".join(targets or ())
    return f"P({head} | {'; '.join(given)})" if given else f"P({head})"
