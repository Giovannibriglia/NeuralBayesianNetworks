"""pomegranate v1.x adapter (PyTorch-backed).

pomegranate v1.0+ ships its Bayesian-network module on a PyTorch backend,
so it's a fair and lightweight comparison point against NBN on the
*discrete* regime. It does not natively support the kind of non-Gaussian
continuous CPDs NBN handles, hence ``supports = {"discrete"}``.

Inference path: pomegranate's ``predict_proba`` takes a
``torch.masked.masked_tensor`` whose unmasked entries are the observed
states. We construct that tensor from the query's evidence dict; the
returned per-node probability vectors are extracted via ``.get_data()``.

Conditional-query handling
--------------------------
pomegranate v1.1.x has a known internal bug in ``predict_proba`` where it
indexes per-node probability tensors with the masked-tensor's underlying
value as if it were a ``long``, raising ``IndexError: tensors used as
indices must be long, ...`` when the value tensor is ``float``. The
workaround (PR-B §A.3, v0.5b) is to construct the ``row`` tensor as
``dtype=torch.long`` before wrapping it in ``torch.masked.masked_tensor``
(line 126 below); pomegranate's internal indexing then receives a
long-typed value, which avoids the float-vs-long coercion bug. Marginal
queries were never affected.

Verified end-to-end at v0.6c-d (15/15 ok cells for ``pomegranate-discrete-ve``
in the inference paper parquet) and locked in by
``tests/integration/test_adapters_smoke.py::test_pomegranate_conditional_query_31``.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from benchmarking.baselines.base import BaselineAdapter
from benchmarking.domains.base import BenchmarkProblem, Query


class PomegranateAdapter(BaselineAdapter):
    """Wraps ``pomegranate.bayesian_network.BayesianNetwork`` (v1.x torch backend)."""

    name = "pomegranate"
    supports = {"discrete"}

    def __init__(self) -> None:
        self.model = None
        self._node_to_idx: Dict[str, int] = {}
        self._cards: Dict[str, int] = {}
        self._topo: List[str] = []

    def fit(self, problem: BenchmarkProblem, epochs: int = 20) -> None:
        try:
            from pomegranate.bayesian_network import BayesianNetwork
            from pomegranate.distributions import Categorical, ConditionalCategorical
        except ImportError as e:
            raise ImportError(
                "PomegranateAdapter needs pomegranate>=1.0: pip install 'pomegranate>=1.0'"
            ) from e

        # Discrete only
        for k, (kind, _) in problem.variables.items():
            if kind != "discrete":
                raise ValueError(f"pomegranate adapter is discrete-only; got {k} ({kind}).")

        # Topological order
        import networkx as nx
        g = nx.DiGraph(); g.add_nodes_from(problem.variables); g.add_edges_from(problem.dag)
        self._topo = list(nx.topological_sort(g))
        self._node_to_idx = {n: i for i, n in enumerate(self._topo)}
        self._cards = {n: int(problem.variables[n][1]) for n in self._topo}

        # Pomegranate v1 wants distribution objects + a list of edges between them.
        # Build per-node distribution from the empirical CPT.
        dist_objs = []
        for node in self._topo:
            k = self._cards[node]
            parents = [p for p, c in problem.dag if c == node]
            x = problem.train_data[node].cpu().long().reshape(-1)
            if not parents:
                counts = torch.zeros(k)
                counts.scatter_add_(0, x, torch.ones_like(x, dtype=torch.float))
                probs = (counts + 1.0) / (counts.sum() + k)
                dist_objs.append(Categorical(probs.reshape(1, k).double()))
            else:
                pa_cards = [self._cards[p] for p in parents]
                strides, stride = [], 1
                for c in reversed(pa_cards):
                    strides.append(stride); stride *= c
                strides = list(reversed(strides))
                n_pa = stride
                pa_idx = torch.zeros_like(x)
                for d, p in enumerate(parents):
                    pa_idx = pa_idx + problem.train_data[p].cpu().long().reshape(-1) * strides[d]
                flat = pa_idx * k + x
                cnt = torch.zeros(n_pa * k)
                cnt.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
                cnt = cnt.reshape(n_pa, k) + 1.0
                probs = cnt / cnt.sum(-1, keepdim=True)
                # Reshape into pomegranate's [*pa_cards, K] layout
                probs = probs.reshape(*pa_cards, k).double()
                dist_objs.append(ConditionalCategorical([probs]))

        # Build edges as (parent_dist_obj, child_dist_obj) pairs
        edges_obj = []
        for parent, child in problem.dag:
            edges_obj.append((dist_objs[self._node_to_idx[parent]],
                              dist_objs[self._node_to_idx[child]]))
        self.model = BayesianNetwork(distributions=dist_objs, edges=edges_obj)

    def query(self, q: Query) -> torch.Tensor:
        assert self.model is not None
        # pomegranate v1.x's predict_proba expects a torch masked-tensor:
        # observed entries are real values; unobserved entries are masked out.
        from torch.masked import masked_tensor
        import warnings

        target = q.targets[0]
        n = len(self._topo)
        # PR-B §A.3 (v0.5b) / v0.7-#31: pomegranate v1.1's predict_proba
        # uses evidence values as indices into per-node probability tensors;
        # if ``row`` is float we hit
        # ``IndexError: tensors used as indices must be long, ...``.
        # The adapter is discrete-only, so all evidence values can safely
        # be encoded as long.  Regression test:
        # ``tests/integration/test_adapters_smoke.py::test_pomegranate_conditional_query_31``.
        row = torch.zeros((1, n), dtype=torch.long)
        mask = torch.zeros((1, n), dtype=torch.bool)
        for k, v in q.evidence.items():
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
        # `out[idx]` may itself be a masked tensor when target is observed; call
        # ``.get_data()`` to extract the underlying probability vector.
        result = out[idx][0]
        if hasattr(result, "get_data"):
            result = result.get_data()
        return result.float().detach()

    def teardown(self) -> None:
        self.model = None
