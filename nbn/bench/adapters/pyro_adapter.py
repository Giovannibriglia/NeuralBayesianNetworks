"""pyro adapter for v0.13 BaselineAdapter protocol.

Stateful fit-then-query contract. Covers one inference label:

  - pyro-empirical-importance  (discrete + continuous_lg + hybrid,
                                Importance-sampling posterior)

For discrete targets, query() returns Posterior(probs=...) built from the
empirical histogram of n_samples importance-weighted particles.

For continuous targets, query() returns Posterior(samples=...) — the full
(n_samples,) sample tensor from _posterior_samples(), NOT the mean.  This
is a behaviour change from the old adapter's query(), which returned
marg.mean(0) (a scalar point estimate).  The change is justified by the
v0.13 Posterior contract: W1 accuracy requires a sample distribution, not
a point mass.  The _posterior_samples() method was already correct; the
fix is one line in query().

Device: None/"auto" now auto-detect (cuda-if-available-else-cpu) via
resolve_device(), consistent with the other adapters.  CAVEAT: PR #102
measured GPU as 11× slower than CPU for pyro's Importance sampler — each
pyro.sample() triggers a CUDA kernel launch (~5–10 µs), and at benchmark
scale (50 particles × 1024 queries × up to 50 nodes) the launch overhead
dominates.  Pin ``device: cpu`` in the config for a pyro baseline whose
wall-clock matters; auto-detect is the default so smoke runs exercise the
GPU path locally.  The lstsq driver follows self.device (gels on CUDA,
gelsd on CPU).

n_samples: default 50.  Reduced from the old default of 200 to fit the
600s per-cell budget (200 particles projected ~1273s at B=1024 batch size;
50 projects ~318s, leaving 282s margin).  MC standard error doubles
(√1/200 ≈ 0.07 → √1/50 ≈ 0.14), acceptable for this noisy IS baseline.

lstsq driver: gelsd on CPU (rank-deficient safe), gels on CUDA (full-rank
QR only, PyTorch constraint).  Preserved from PR #102.

Hybrid-family LG-conditional path (PR #100) preserved:
  _cpt_parents[node] — discrete-only parent list for CPT construction;
  _fit_lg_leaf       — one-hot encodes discrete parents (k-1 columns each);
  _pyro_model()      — reconstructs μ from both cont and one-hot blocks.

Note: pyro-empirical (parameter-learning label, no inference_method suffix)
is out of scope for Phase 1b-iii. Inference configs always use
inference_method: importance → label "pyro-empirical-importance".

The old adapter at nbn/bench/baselines/pyro_adapter.py is untouched
and continues to serve the v0.12 runner.

Reference: docs/v0.13-benchmark-redesign.md §4.1
Old (functional) adapter: nbn/bench/baselines/pyro_adapter.py
"""
from __future__ import annotations

from typing import Any

import torch

from nbn.bench.core._device import resolve_device
from nbn.bench.core.applicability import BASELINE_FAMILY_APPLICABILITY as _BASELINE_APPLICABILITY
from nbn.bench.core.interfaces import default_query_batch
from nbn.bench.domains.base import BenchmarkProblem, Query
from nbn.bench.domains.posterior import Posterior

_VALID_MECHANISMS = frozenset({"empirical"})
_VALID_INFERENCE_METHODS = frozenset({"importance"})


class PyroAdapter:
    """Stateful pyro adapter implementing the v0.13 BaselineAdapter protocol.

    Handles one inference label: pyro-empirical-importance.

    Construction::

        adapter = PyroAdapter(mechanism="empirical", inference_method="importance")
        adapter = PyroAdapter(mechanism="empirical", inference_method="importance",
                              n_samples=50)  # device auto-detects

    The ``name`` attribute is derived:
    ``"pyro-{mechanism}-{inference_method}"`` → ``"pyro-empirical-importance"``.
    """

    # v0.14 (#148) §5.6: sequential-only (query_batch is the default
    # helper) — the speed-benchmark sweep runs this once at batch_size=1.
    supports_batched_queries: bool = False

    # Parameter-recovery capability (#109 PR 5): exposes learned discrete CPTs
    # via extract_learned_cpts. Pyro is MIXED-applicable (discrete, continuous_lg,
    # hybrid), so unlike pomegranate the family gate's not_applicable path IS
    # reachable — continuous/hybrid cells get not_applicable. (Precedent: flag=True
    # when the adapter implements extraction; the cell-level gate decides.)
    supports_param_recovery: bool = True

    def __init__(
        self,
        mechanism: str,
        inference_method: str | None,
        n_samples: int = 50,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        if mechanism not in _VALID_MECHANISMS:
            raise ValueError(
                f"Unknown mechanism {mechanism!r}. "
                f"Valid: {sorted(_VALID_MECHANISMS)}"
            )
        # inference_method=None is the fit-only / parameter-learning
        # construction (#109): the adapter is fit (and scored where supported),
        # never queried, so it carries an engine-less name (e.g. "pyro-empirical").
        if (inference_method is not None
                and inference_method not in _VALID_INFERENCE_METHODS):
            raise ValueError(
                f"Unknown inference_method {inference_method!r}. "
                f"Valid: {sorted(_VALID_INFERENCE_METHODS)}"
            )
        self.mechanism = mechanism
        self.inference_method = inference_method
        self.n_samples = int(n_samples)
        # None / "auto" resolve to cuda-if-available-else-cpu; concrete
        # strings pass through.  Stored as a plain str (not torch.device) so
        # that self.device.startswith("cuda") works for the lstsq driver
        # check below (gels on CUDA, gelsd on CPU).
        self.device: str = resolve_device(device)
        self.name = (
            f"pyro-{mechanism}-{inference_method}"
            if inference_method is not None else f"pyro-{mechanism}"
        )

        # State populated by fit()
        self._cpts: dict[str, torch.Tensor] = {}
        # _gaussian[node] = (beta, sigma, cont_parents, disc_parents, disc_cards)
        #   beta  — [1 + len(cont_pa) + sum(k-1 for k in disc_cards)]
        #           layout: [intercept | cont coefs | one-hot coefs (k-1 per disc pa)]
        #   sigma — scalar residual std, clamped ≥ 1e-3
        self._gaussian: dict[
            str,
            tuple[torch.Tensor, torch.Tensor, list[str], list[str], list[int]],
        ] = {}
        # _cpt_parents[node] — discrete-only parents used to build the CPT.
        # In hybrid BNs a discrete node may have continuous parents; those are
        # ignored in the CPT (P(node | discrete_parents) only).
        # _pyro_model() uses this list for CPT row-index construction so that
        # fit and inference use the same cardinalities.
        self._cpt_parents: dict[str, list[str]] = {}
        self._parents: dict[str, list[str]] = {}
        self._cards: dict[str, int] = {}
        self._topo: list[str] = []
        self.problem: BenchmarkProblem | None = None
        self._fitted: bool = False

    # -------------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------------

    def fit(self, problem: BenchmarkProblem, **kwargs: Any) -> None:
        """Fit on problem.train_data. State stored on self.

        kwargs:
            epochs (int): accepted for runner API compatibility; not used —
                pyro fitting is empirical (count-based CPTs + lstsq), not
                epoch-based gradient descent.
        """
        try:
            import pyro  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PyroAdapter needs pyro-ppl: pip install pyro-ppl"
            ) from exc

        import networkx as nx

        self.problem = problem
        g = nx.DiGraph()
        g.add_nodes_from(problem.variables)
        g.add_edges_from(problem.dag)
        self._topo = list(nx.topological_sort(g))
        self._parents = {n: list(g.predecessors(n)) for n in self._topo}

        # Fit in topological order so each discrete node's parents are
        # already registered in self._cards before _fit_discrete_cpt looks
        # them up.  problem.variables iterates in insertion order, which is
        # not guaranteed topological (e.g. alarm/sachs have children before
        # their parents) — using it here raised KeyError(parent). See #127.
        for node in self._topo:
            kind, k = problem.variables[node]
            if kind == "discrete":
                self._cards[node] = k
                self._fit_discrete_cpt(node, k, problem)
            else:
                self._fit_lg_leaf(node, problem)

        self._fitted = True

    def _fit_discrete_cpt(
        self,
        node: str,
        k: int,
        problem: BenchmarkProblem,
    ) -> None:
        """Build empirical CPT for a discrete node with Laplace smoothing (α=1).

        In hybrid BNs a discrete node may have continuous parents.  The CPT
        only handles discrete parents — continuous parents are filtered out.
        _cpt_parents[node] records the discrete-only parent list so that
        _pyro_model() uses the same parent subset at inference time.
        """
        x = problem.train_data[node].to(self.device).long().reshape(-1)
        disc_parents = [
            p for p in self._parents[node]
            if problem.variables[p][0] == "discrete"
        ]
        self._cpt_parents[node] = disc_parents

        if not disc_parents:
            counts = torch.zeros(k, device=self.device)
            counts.scatter_add_(0, x, torch.ones_like(x, dtype=torch.float))
            self._cpts[node] = ((counts + 1.0) / (counts.sum() + k)).unsqueeze(0)
            return

        pa_cards = [self._cards[p] for p in disc_parents]
        strides: list[int] = []
        stride = 1
        for c in reversed(pa_cards):
            strides.append(stride)
            stride *= c
        strides = list(reversed(strides))
        n_pa = stride

        pa_idx = torch.zeros_like(x)
        for d, p in enumerate(disc_parents):
            pa_idx = (
                pa_idx
                + problem.train_data[p].to(self.device).long().reshape(-1) * strides[d]
            )
        flat = pa_idx * k + x
        cnt = torch.zeros(n_pa * k, device=self.device)
        cnt.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
        cnt = cnt.reshape(n_pa, k) + 1.0
        self._cpts[node] = cnt / cnt.sum(-1, keepdim=True)

    def _fit_lg_leaf(self, node: str, problem: BenchmarkProblem) -> None:
        """Linear-Gaussian conditional fit for a continuous node.

        Fits y = β₀ + Σ βᵢ·contᵢ + Σ γⱼ·ohⱼ + ε via lstsq.
        Discrete parents are one-hot encoded using k−1 indicator columns
        (last category dropped to avoid collinearity with the intercept).
        Continuous parents enter directly as numeric features.

        Root nodes (no parents): marginal Normal(mean, std).

        lstsq driver: gelsd on CPU (handles rank-deficient X), gels on CUDA
        (full-rank QR — PyTorch constraint; gelsd is CPU-only).
        Validated as equivalent for paper-scale data in PR #102.

        beta layout: [intercept | cont_coefs | oh_coefs (k-1 per disc pa)]
        This layout must match _pyro_model()'s reconstruction.
        """
        y = problem.train_data[node].to(self.device).float().reshape(-1)
        cont_parents = [
            p for p in self._parents[node]
            if problem.variables[p][0] == "continuous"
        ]
        disc_parents = [
            p for p in self._parents[node]
            if problem.variables[p][0] == "discrete"
        ]
        disc_cards = [problem.variables[p][1] for p in disc_parents]

        if not cont_parents and not disc_parents:
            beta = y.mean().reshape(1)
            sigma = y.std().clamp_min(1e-3)
            self._gaussian[node] = (beta, sigma, [], [], [])
            return

        cols: list[torch.Tensor] = [torch.ones_like(y)]
        for p in cont_parents:
            cols.append(problem.train_data[p].to(self.device).float().reshape(-1))
        for p, k in zip(disc_parents, disc_cards):
            x_disc = (
                problem.train_data[p].to(self.device).long().reshape(-1).clamp(0, k - 1)
            )
            oh = torch.zeros(len(y), k, device=self.device)
            oh.scatter_(1, x_disc.unsqueeze(1), 1.0)
            cols.append(oh[:, :-1])  # k-1 indicator columns, last category dropped

        X = torch.cat([c.reshape(len(y), -1) for c in cols], dim=1)
        driver = "gels" if self.device.startswith("cuda") else "gelsd"
        beta = torch.linalg.lstsq(X, y, driver=driver).solution.reshape(-1)
        residuals = y - X @ beta
        sigma = residuals.std().clamp_min(1e-3)
        self._gaussian[node] = (beta, sigma, cont_parents, disc_parents, disc_cards)

    # -------------------------------------------------------------------------
    # Pyro model
    # -------------------------------------------------------------------------

    def _pyro_model(self):
        """Return a Pyro generative model function for ancestral sampling.

        The returned model() function captures the fitted CPTs/Gaussians as
        local aliases (snapshots of self attributes at call time), making it
        safe to pass to poutine.condition.  The pyro/dist imports live inside
        model() to avoid module-level pyro dependency.

        beta layout for continuous nodes must match _fit_lg_leaf():
          [intercept | cont_coefs | oh_coefs (k-1 per disc_pa in order)]
        """
        # Local aliases — captured by the inner model() closure
        cpts = self._cpts
        gauss = self._gaussian
        cpt_parents = self._cpt_parents
        cards = self._cards
        topo = self._topo

        def model():
            import pyro
            import pyro.distributions as dist

            s: dict = {}
            for node in topo:
                if node in cpts:
                    # Discrete node — use discrete-only parent list for CPT
                    # row indexing (mirrors _fit_discrete_cpt).
                    cpa = cpt_parents.get(node, [])
                    if cpa:
                        row, stride = 0, 1
                        for p in reversed(cpa):
                            v = s[p].long() if isinstance(s[p], torch.Tensor) else int(s[p])
                            row = row + v * stride
                            stride *= cards[p]
                        probs = cpts[node][row]
                    else:
                        probs = cpts[node][0]
                    s[node] = pyro.sample(node, dist.Categorical(probs))
                else:
                    # Continuous node — reconstruct μ from beta layout.
                    beta, sigma, cont_pa, disc_pa, disc_cards_local = gauss[node]
                    mu = beta[0]
                    for i, p in enumerate(cont_pa):
                        mu = mu + beta[i + 1] * s[p]
                    # One-hot contribution from discrete parents.
                    # beta[1+len(cont_pa) : 1+len(cont_pa)+(k-1)] per disc pa.
                    offset = 1 + len(cont_pa)
                    for p, k in zip(disc_pa, disc_cards_local):
                        v = s[p].long() if isinstance(s[p], torch.Tensor) else int(s[p])
                        v = max(0, min(int(v), k - 1))
                        # Last category → all indicators 0 (absorbed by intercept)
                        if v < k - 1:
                            mu = mu + beta[offset + v]
                        offset += k - 1
                    s[node] = pyro.sample(node, dist.Normal(mu, sigma))
            return s

        return model

    # -------------------------------------------------------------------------
    # Posterior sampling
    # -------------------------------------------------------------------------

    def _posterior_samples(
        self, q: Query, *, n_samples: int,
    ) -> torch.Tensor:
        """Draw n_samples importance-weighted posterior samples of q.targets[0].

        Returns a (n_samples,) float tensor on self.device.
        """
        try:
            import pyro.poutine as poutine
            from pyro.infer import EmpiricalMarginal, Importance
        except ImportError as exc:
            raise ImportError(
                "PyroAdapter needs pyro-ppl: pip install pyro-ppl"
            ) from exc

        target = q.targets[0]
        evidence: dict = {}
        for k, v in q.evidence.items():
            if v is None:
                # Phase 3 empty mode: leave this site unconditioned so
                # importance sampling marginalizes over it.
                continue
            val = v.item() if isinstance(v, torch.Tensor) else v
            # Discrete evidence → long tensor; continuous → float tensor
            evidence[k] = (
                torch.tensor(int(val), device=self.device)
                if k in self._cards
                else torch.tensor(float(val), device=self.device)
            )

        model = self._pyro_model()
        conditioned = poutine.condition(model, data=evidence)
        posterior = Importance(conditioned, num_samples=n_samples).run()
        marg = EmpiricalMarginal(posterior, sites=target)
        return torch.stack([marg() for _ in range(n_samples)]).float().reshape(n_samples)

    # -------------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------------

    def query(self, q: Query) -> Posterior:
        """Query the fitted model. This call is what gets timed.

        Returns:
            Posterior(probs=...)   for discrete targets — empirical histogram
                                   from n_samples IS particles, shape (K,).
            Posterior(samples=...) for continuous targets — full (n_samples,)
                                   sample tensor.  NOT the point estimate.

        The continuous-target return is a behaviour change from the old
        adapter, which returned marg.mean(0) (a scalar).  The full sample
        tensor is needed for W1 accuracy computation (v0.13 Posterior
        contract).

        Raises:
            RuntimeError: if called before fit().
        """
        if not self._fitted:
            raise RuntimeError(
                "Adapter not fitted. Call fit() before query()."
            )

        target = q.targets[0]
        marg = self._posterior_samples(q, n_samples=self.n_samples)

        if target in self._cards:
            # Discrete target: empirical histogram → normalised probs
            k = self._cards[target]
            counts = torch.zeros(k, device=self.device)
            for v in marg.long().reshape(-1).tolist():
                if 0 <= v < k:
                    counts[v] += 1
            probs = (counts / counts.sum().clamp_min(1e-12)).cpu()
            return Posterior(probs=probs)

        # Continuous target: full sample distribution (not collapsed to mean)
        return Posterior(samples=marg.cpu())

    def query_batch(self, queries: list[Query]) -> list[Posterior]:
        """Sequential default (PR 1, #148). Pyro stays sequential
        (design doc §3.3); no override planned."""
        return default_query_batch(self, queries)

    # -------------------------------------------------------------------------
    # Applicability
    # -------------------------------------------------------------------------

    def is_applicable(self, problem: BenchmarkProblem) -> bool:
        """Return True if this adapter can handle problem's family.

        Delegates to _BASELINE_APPLICABILITY using self.name as the key.
        Family is inferred from variable kinds:
          all discrete    → "discrete"
          all continuous  → "continuous_lg"   (approximation; see note)
          mixed           → "hybrid"

        Note: continuous_nongauss cannot be distinguished from continuous_lg
        by variable kinds alone.  pyro-empirical-importance excludes
        continuous_nongauss in the registry; this is a known Phase 1b
        approximation that Phase 1c will resolve per-adapter.
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

        Returns ``{node: probs[n_parent_configs, K]}`` for each discrete node, in
        the canonical layout the recovery metric compares cell-by-cell against
        the true model (parents sorted lex, configs row-major with the first
        parent slowest, classes ``0..K-1``, each row a distribution).

        READS the stored ``self._cpts`` — no re-estimation, no fill, no clamp.
        ``_fit_discrete_cpt`` builds the full declared grid with Laplace
        smoothing (alpha=1): ``(counts + 1) / (counts.sum() + K)``. That single
        ``+1`` structurally avoids BOTH pathologies the earlier adapters hit —
        the hard zeros that make pgmpy-mle's KL diverge, and the ``0/0 = NaN``
        on unseen parent configs that bit pomegranate (#218): an unseen config
        smooths to uniform ``1/K``, a zero-count class to a small positive mass.
        So no NaN-fill and no zero-clamp are needed here, and pyro's recovery KL
        is always finite (it joins the smoothed bucket with pgmpy-bayes / NBN).

        The stored ``self._cpts[node]`` is ``[n_parent_configs, K]`` with the
        config index flattened over ``self._cpt_parents[node]`` (dag-edge order,
        first parent slowest). Extraction unflattens that into per-parent axes,
        permutes them into canonical lex order, and reshapes back.

        Hybrid-scope note: pyro is mixed-applicable, and in a hybrid problem a
        discrete node's continuous parents are filtered out of
        ``self._cpt_parents`` by ``_fit_discrete_cpt`` (which conditions on
        discrete parents only). Recovery is confined by the measurement's family
        gate to FULLY-DISCRETE problems, where ``self._cpt_parents`` equals the
        full parent set — so the filtering is identity-equivalent in scope, and
        reading ``self._cpt_parents`` directly (no hybrid special-casing) is safe.

        Integer class indices ⇒ no state reindex. The only defensive check is a
        row-sum validation (everything else is structurally guaranteed by the
        Laplace fit); a row not summing to 1 raises -> status="error".
        """
        if not self._fitted or self.problem is None:
            raise RuntimeError(
                "Adapter not fitted. Call fit() before extract_learned_cpts()."
            )
        out: dict[str, torch.Tensor] = {}
        for node in self._topo:
            kind, k = self.problem.variables[node]
            if kind != "discrete":
                continue
            k = int(k)
            cpt_parents = self._cpt_parents[node]
            flat = self._cpts[node].detach().cpu().float()  # [n_configs, K]

            if not cpt_parents:
                cpt = flat.reshape(1, k)
            else:
                pa_cards = [int(self._cards[p]) for p in cpt_parents]
                raw = flat.reshape(*pa_cards, k)            # [*pa_cards(dag), K]
                canon = sorted(cpt_parents)
                perm = [cpt_parents.index(p) for p in canon] + [len(cpt_parents)]
                n_configs = 1
                for p in canon:
                    n_configs *= int(self._cards[p])
                cpt = raw.permute(*perm).reshape(n_configs, k)

            row_sums = cpt.sum(dim=-1)
            if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
                i = int((row_sums - 1.0).abs().argmax())
                raise ValueError(
                    f"node {node!r}: extracted CPT row {i} does not sum to 1 "
                    f"(sum={float(row_sums[i]):.6f})"
                )
            out[node] = cpt
        return out
