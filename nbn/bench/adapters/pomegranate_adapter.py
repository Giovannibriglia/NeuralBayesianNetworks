"""pomegranate adapter for v0.13 BaselineAdapter protocol.

Stateful fit-then-query contract. Covers one inference label:

  - pomegranate-discrete-ve

Returns Posterior(probs=...) — the marginal or conditional probability
vector for the target node, computed via pomegranate's predict_proba.

fit() builds per-node empirical (raw maximum-likelihood) CPTs from training
data and passes them as pre-built distribution objects.  This avoids
pomegranate's own optimiser path.  No smoothing is applied: a parent
configuration never seen in training yields a zero (NaN-normalised) CPT row,
i.e. the honest library-default behaviour with no adapter-side prior
(v0.14 — adapter-side +1 Laplace removed for methodology fidelity).

KEY BUG FIX PRESERVED (v0.5b / issue #31):
  row tensor must use dtype=torch.long before wrapping in
  torch.masked.masked_tensor.  pomegranate v1.1.x uses evidence values as
  long indices internally; float dtype triggers:
      IndexError: tensors used as indices must be long, int, byte or bool,
                  but got torch.FloatTensor
  Regression test:
      tests/integration/test_adapters_smoke.py::test_pomegranate_conditional_query_31

The old adapter at nbn/bench/baselines/pomegranate_adapter.py is
untouched and continues to serve the v0.12 runner.

Reference: docs/v0.13-benchmark-redesign.md §4.1
Old (functional) adapter: nbn/bench/baselines/pomegranate_adapter.py
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import torch

from nbn.bench.core._device import resolve_device
from nbn.bench.core.applicability import BASELINE_FAMILY_APPLICABILITY as _BASELINE_APPLICABILITY
from nbn.bench.core.interfaces import default_query_batch
from nbn.bench.domains.base import BenchmarkProblem, Query
from nbn.bench.domains.posterior import Posterior

logger = logging.getLogger(__name__)


class PomegranateAdapter:
    """Stateful pomegranate adapter implementing the v0.13 BaselineAdapter protocol.

    Handles one inference label: pomegranate-discrete-ve.

    Construction::

        adapter = PomegranateAdapter()
    """

    name = "pomegranate-discrete-ve"

    # v0.14 (#148) §5.6: real library-batched query_batch override
    # (PR 3) — swept by the speed benchmark. See nbn_adapter for why
    # this is an explicit flag.
    supports_batched_queries: bool = True

    # Parameter-recovery capability (#109 PR 4): exposes learned discrete CPTs
    # via extract_learned_cpts. Pomegranate is discrete-only via is_applicable;
    # the family gate's not_applicable path is structurally unreachable here, so
    # extraction returns a non-empty dict on every applicable cell. (Precedent
    # from PR 3: flag=True when the adapter implements extraction; the cell-level
    # gate decides applicability.)
    supports_param_recovery: bool = True

    def __init__(self, device: str | None = None, **kwargs: Any) -> None:
        # None / "auto" -> cuda-if-available-else-cpu; concrete passes through.
        # pomegranate 1.x is torch-based: CPTs are built on CPU (counting is
        # cheap there) and the assembled BayesianNetwork is moved to
        # self.device via model.to() in fit().  Verified: predict_proba runs
        # correctly on CUDA when the whole module (incl. the lazily-built
        # factor graph) is moved together; building distributions directly on
        # CUDA instead leaves factor-graph message buffers on CPU and crashes.
        self.device = resolve_device(device)
        # State populated by fit()
        self.model: Any | None = None
        self._node_to_idx: dict[str, int] = {}
        self._cards: dict[str, int] = {}
        self._topo: list[str] = []
        self.problem: BenchmarkProblem | None = None

    # -------------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------------

    def fit(self, problem: BenchmarkProblem, **kwargs: Any) -> None:
        """Fit on problem.train_data. State stored on self.

        kwargs:
            epochs (int): accepted for runner API compatibility; not used —
                CPTs are computed analytically from counts, not via gradient
                descent.

        Discrete-only: raises ValueError if problem contains continuous nodes.
        """
        try:
            from pomegranate.bayesian_network import BayesianNetwork
            from pomegranate.distributions import Categorical, ConditionalCategorical
        except ImportError as exc:
            raise ImportError(
                "PomegranateAdapter needs pomegranate>=1.0: "
                "pip install 'pomegranate>=1.0'"
            ) from exc

        for k, (kind, _) in problem.variables.items():
            if kind != "discrete":
                raise ValueError(
                    f"PomegranateAdapter is discrete-only; "
                    f"got {k!r} ({kind!r})."
                )

        import networkx as nx

        g = nx.DiGraph()
        g.add_nodes_from(problem.variables)
        g.add_edges_from(problem.dag)
        self._topo = list(nx.topological_sort(g))
        self._node_to_idx = {n: i for i, n in enumerate(self._topo)}
        self._cards = {n: int(problem.variables[n][1]) for n in self._topo}

        dist_objs: list[Any] = []
        for node in self._topo:
            k = self._cards[node]
            parents = [p for p, c in problem.dag if c == node]
            x = problem.train_data[node].cpu().long().reshape(-1)

            if not parents:
                counts = torch.zeros(k)
                counts.scatter_add_(0, x, torch.ones_like(x, dtype=torch.float))
                probs = counts / counts.sum()
                dist_objs.append(Categorical(probs.reshape(1, k).double()))
            else:
                pa_cards = [self._cards[p] for p in parents]
                strides: list[int] = []
                stride = 1
                for c in reversed(pa_cards):
                    strides.append(stride)
                    stride *= c
                strides = list(reversed(strides))
                n_pa = stride

                pa_idx = torch.zeros_like(x)
                for d, p in enumerate(parents):
                    pa_idx = (
                        pa_idx
                        + problem.train_data[p].cpu().long().reshape(-1) * strides[d]
                    )

                flat = pa_idx * k + x
                cnt = torch.zeros(n_pa * k)
                cnt.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
                cnt = cnt.reshape(n_pa, k)
                probs = cnt / cnt.sum(-1, keepdim=True)
                # Reshape into pomegranate's [*pa_cards, K] layout
                probs = probs.reshape(*pa_cards, k).double()
                dist_objs.append(ConditionalCategorical([probs]))

        edges_obj = [
            (
                dist_objs[self._node_to_idx[parent]],
                dist_objs[self._node_to_idx[child]],
            )
            for parent, child in problem.dag
        ]
        self.model = BayesianNetwork(distributions=dist_objs, edges=edges_obj)
        # Move the whole module (distributions + lazily-built factor graph) to
        # the resolved device. Build stays on CPU above; only the final model
        # is relocated. No-op for "cpu".
        if self.device != "cpu":
            self.model.to(self.device)
        self.problem = problem

    # -------------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------------

    def query(self, q: Query) -> Posterior:
        """Query the fitted model. Returns Posterior(probs=...) for the target.

        Uses pomegranate's predict_proba on a masked tensor where observed
        entries are the evidence values and the target + unobserved nodes
        are masked out.

        Bug fix (v0.5b / issue #31): row tensor uses dtype=torch.long.
        See module docstring for details.

        Raises:
            RuntimeError: if called before fit().
        """
        if self.model is None:
            raise RuntimeError(
                "Adapter not fitted. Call fit() before query()."
            )

        from torch.masked import masked_tensor

        target = q.targets[0]
        n = len(self._topo)
        # dtype=torch.long is mandatory — see module docstring / bug fix note.
        # Tensors live on self.device so they match the (possibly CUDA) model.
        row = torch.zeros((1, n), dtype=torch.long, device=self.device)
        mask = torch.zeros((1, n), dtype=torch.bool, device=self.device)

        for k, v in q.evidence.items():
            if v is None:
                # Phase 3 empty mode: leave this node unobserved (masked) so
                # predict_proba marginalizes over it.
                continue
            i = self._node_to_idx[k]
            if isinstance(v, torch.Tensor):
                scalar = int(v.reshape(-1)[0].item())
            else:
                scalar = int(v)
            row[0, i] = scalar
            mask[0, i] = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mt = masked_tensor(row, mask)
            out = self.model.predict_proba(mt)

        idx = self._node_to_idx[target]
        result = out[idx][0]
        # `result` may itself be a masked tensor when the target is observed;
        # .get_data() extracts the underlying probability vector.
        if hasattr(result, "get_data"):
            result = result.get_data()
        # Back to CPU for downstream metric computation (consistent with the
        # other adapters, which return CPU posteriors).
        probs = result.float().detach().cpu()
        return Posterior(probs=probs)

    def query_batch(self, queries: list[Query]) -> list[Posterior]:
        """Library-level batching via pomegranate's row-batched
        ``predict_proba`` API.

        Assumes all queries share (targets, frozenset(evidence_keys)) —
        the selector contract per design doc §1.2.  Heterogeneous batches
        (different targets or evidence keys, or mixed concrete/None
        evidence values) fall back to the sequential default helper.

        Stacks B rows into a single [B, n] MaskedTensor with the shared
        mask pattern repeated, calls ``predict_proba`` once, and indexes
        the per-variable output list at the target's column ([B, K]).

        Mirrors NBNAdapter.query_batch's contract checks and edge cases.
        See docs/v0.14-batched-queries-design.md §3.2.
        """
        if not queries:
            return []
        # B=1: single-query path — identical semantics, no stacking overhead.
        if len(queries) == 1:
            return [self.query(queries[0])]
        if self.model is None:
            raise RuntimeError(
                "Adapter not fitted. Call fit() before query_batch()."
            )

        # Verify the shared (targets, evidence_keys) contract; fall back to
        # the sequential helper on any heterogeneity.
        first = queries[0]
        targets = first.targets
        evidence_keys = frozenset(first.evidence.keys())
        for q in queries[1:]:
            if q.targets != targets:
                return default_query_batch(self, queries)
            if frozenset(q.evidence.keys()) != evidence_keys:
                return default_query_batch(self, queries)

        from torch.masked import masked_tensor

        target = targets[0]
        b = len(queries)
        n = len(self._topo)
        # dtype=torch.long is mandatory — see module docstring / bug fix
        # note (v0.5b / issue #31). Tensors live on self.device so they
        # match the (possibly CUDA) model. Same conventions as query().
        rows = torch.zeros((b, n), dtype=torch.long, device=self.device)
        mask = torch.zeros((b, n), dtype=torch.bool, device=self.device)

        for key in evidence_keys:
            values = [q.evidence[key] for q in queries]
            none_count = sum(v is None for v in values)
            if none_count == len(values):
                # Phase 3 empty mode: leave this column unobserved (masked)
                # so predict_proba marginalizes over it, as query() does.
                continue
            if none_count > 0:
                return default_query_batch(self, queries)  # mixed modes
            i = self._node_to_idx[key]
            rows[:, i] = torch.tensor(
                [
                    int(v.reshape(-1)[0].item())
                    if isinstance(v, torch.Tensor) else int(v)
                    for v in values
                ],
                dtype=torch.long, device=self.device,
            )
            mask[:, i] = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mt = masked_tensor(rows, mask)
            out = self.model.predict_proba(mt)

        idx = self._node_to_idx[target]
        result = out[idx]  # [B, K] — masked tensor iff target observed
        if hasattr(result, "get_data"):
            result = result.get_data()
        probs = result.float().detach().cpu()
        if probs.dim() != 2 or probs.shape[0] != b:
            raise RuntimeError(
                f"Unexpected predict_proba output for {self.name}: "
                f"shape {tuple(probs.shape)}, expected [{b}, K]."
            )
        return [Posterior(probs=probs[i]) for i in range(b)]

    # -------------------------------------------------------------------------
    # Applicability
    # -------------------------------------------------------------------------

    def is_applicable(self, problem: BenchmarkProblem) -> bool:
        """Return True if this adapter can handle problem's family.

        Delegates to _BASELINE_APPLICABILITY using self.name as the key.
        Family is inferred from variable kinds (same logic as NBNAdapter):
          all discrete    → "discrete"
          all continuous  → "continuous_lg"
          mixed           → "hybrid"
        """
        kinds = {kind for kind, _ in problem.variables.values()}
        if kinds == {"discrete"}:
            family = "discrete"
        elif kinds == {"continuous"}:
            family = "continuous_lg"
        else:
            family = "hybrid"

        entry = _BASELINE_APPLICABILITY.get(self.name)
        if entry is None:
            return False
        return family in entry.families

    def extract_learned_cpts(self) -> dict[str, torch.Tensor]:
        """Learned discrete CPTs in the canonical layout (param-recovery, #109).

        Returns ``{node: probs[n_parent_configs, K]}`` for every node, in the
        canonical layout the recovery metric compares cell-by-cell against the
        true model (parents sorted lex, configs row-major with the first parent
        slowest, classes ``0..K-1``, each row a distribution).

        READS the stored distributions rather than re-estimating. The adapter
        builds the FULL declared grid by counting at fit time
        (``_fit_discrete`` sizes every CPT to the declared cardinalities), so
        the stored ``Categorical`` / ``ConditionalCategorical`` parameters
        already span ``0..card-1`` on every axis — reading them reflects the
        deployed model's actual parameters. Re-estimation would re-run identical
        arithmetic: pure cost, zero correctness gain. (Contrast PR 3's pgmpy
        path, which re-estimates because pgmpy's ``bn.fit`` truncates the grid
        to observed states.)

        Unseen-parent-config fill: a parent configuration with zero training
        observations is stored as ``NaN`` in the fitted distribution
        (``counts.sum() == 0`` → ``0/0``; a known adapter artifact, filed
        separately as #218). Extraction fills those rows with uniform ``1/K`` to
        match pgmpy-mle's fit-time uniform-fill under the same condition, keeping
        cross-adapter recovery numbers directly comparable. If the fit-time NaN
        is fixed in the adapter, this extraction-time fill can be removed.

        Hard zeros (a class with zero count under an OBSERVED parent config) are
        preserved — pomegranate is unsmoothed MLE, so they yield ``+inf``
        recovery KL by the locked no-second-clamp rule, exactly like pgmpy-mle.
        """
        if self.model is None or self.problem is None:
            raise RuntimeError(
                "Adapter not fitted. Call fit() before extract_learned_cpts()."
            )
        variables = self.problem.variables
        out: dict[str, torch.Tensor] = {}
        for node in self._topo:
            k = int(self._cards[node])
            parents_dag = [p for p, c in self.problem.dag if c == node]
            dist = self.model.distributions[self._node_to_idx[node]]

            if not parents_dag:
                cpt = dist.probs.detach().cpu().reshape(1, k).float()
            else:
                raw = dist.probs[0].detach().cpu().float()  # [*pa_cards(dag), K]
                canon = sorted(parents_dag)
                # Permute parent axes dag-order -> canonical lex; class axis last.
                perm = [parents_dag.index(p) for p in canon] + [len(parents_dag)]
                n_configs = 1
                for p in canon:
                    n_configs *= int(variables[p][1])
                cpt = raw.permute(*perm).reshape(n_configs, k)

            # Unseen-config rows (0/0 -> NaN) -> uniform 1/K (pgmpy convention).
            bad = torch.isnan(cpt).any(dim=-1)
            if bool(bad.any()):
                cpt = cpt.clone()
                cpt[bad] = 1.0 / k

            # Loud validation: every row must be a distribution.
            row_sums = cpt.sum(dim=-1)
            if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
                i = int((row_sums - 1.0).abs().argmax())
                raise ValueError(
                    f"node {node!r}: extracted CPT row {i} does not sum to 1 "
                    f"(sum={float(row_sums[i]):.6f})"
                )
            out[node] = cpt
        return out
