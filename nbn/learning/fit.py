from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

import torch

from nbn.learning.weighting import select, validate_weights, weighted_mean
from nbn.utils.batching import pack_parents

if TYPE_CHECKING:
    from nbn.core.network import NeuralBayesianNetwork

logger = logging.getLogger(__name__)


@dataclass
class TrainHistory:
    """Records per-node training metrics.

    ``node_log_likelihoods`` holds **in-sample** mean log-likelihoods —
    scored on the same rows the mechanism was fitted on.  ``fit`` performs no
    validation split and no early stopping, so these are a fit-quality trace,
    never a generalisation estimate.
    """

    node_log_likelihoods: Dict[str, List[float]] = field(default_factory=dict)
    wall_clock_s: float = 0.0
    device: str = "cpu"

    def mean_ll(self, node: str) -> float:
        vals = self.node_log_likelihoods.get(node, [])
        return sum(vals) / len(vals) if vals else float("nan")


def fit(
    model: NeuralBayesianNetwork,
    data: Dict[str, torch.Tensor],
    *,
    method: str = "local",
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    device: str | None = None,
    log_every: int = 10,
    weights: torch.Tensor | None = None,
    consolidate: bool = True,
    warm_start: bool = False,
    **kwargs: Any,
) -> TrainHistory:
    """Fit all node mechanisms to data.

    Parameters
    ----------
    model:
        The ``NeuralBayesianNetwork`` to fit.
    data:
        Dict mapping node name → data tensor of shape ``[N, D]`` or ``[N]``.
    method:
        ``"local"`` — calls ``mechanism.fit_local(x, parents, **kwargs)``
        for each node independently (parallelisable in the future).
        ``"joint"`` — single gradient pass over ``-sum_i log p(x_i | pa_i)``.
    epochs:
        Number of training epochs (used by neural mechanisms). ``None``
        (default) means each mechanism uses its own designed budget (e.g.
        flow 300, MDN 200, neural-categorical 100); an explicit value
        overrides that budget globally for every mechanism.
        For ``method="joint"`` there is no per-mechanism budget, so ``None``
        falls back to 100.
    batch_size:
        Mini-batch size for gradient-based methods. ``None`` (default) =
        per-mechanism default; explicit value overrides globally
        (``"joint"`` fallback: 4096).
    lr:
        Learning rate. ``None`` (default) = per-mechanism default; explicit
        value overrides globally (``"joint"`` fallback: 1e-3).
    device:
        Target device; data is moved here.
    log_every:
        Log progress every N nodes.
    weights:
        Optional ``[N]`` non-negative per-sample multiplicities aligned with
        the data rows.  They are *not* a probability distribution and need not
        sum to 1: ``w = [2, 1]`` means the first row counts twice, and fitting
        with integer weights equals fitting on the data with each row repeated
        that many times.  A weight of exactly 0 is equivalent to dropping the
        row.  Gradient-trained mechanisms reduce with a weighted **mean**
        (``sum w*nll / sum w``) so the effective step size does not scale with
        the weights' overall magnitude — see :mod:`nbn.learning.weighting`.

        Mechanisms that cannot honour weights raise *before* any node is
        fitted; see ``Mechanism.supports_weights``.
    consolidate:
        If True (default), neural mechanisms snapshot their post-fit EWC
        state (theta* + diagonal Fisher) so ``model.update()`` works later.
        Set False to skip the Fisher pass entirely for fit-only workloads
        (it costs up to ``sample_cap`` sequential backward passes per node);
        ``model.update()`` on such a model raises until refit with
        ``consolidate=True``.

        Callers running ``fit`` in a loop -- an EM outer loop, say -- pay that
        Fisher pass on *every* iteration.  It is almost always wanted only on
        the final fit, if at all; ``warm_start`` deliberately does not change
        this default, so pass ``consolidate=False`` explicitly.
    warm_start:
        If True, each mechanism continues from the parameters it already
        holds rather than rebuilding from a fresh initialisation.  Default
        False reproduces the historical behaviour exactly.

        This is what makes an iterative caller's second call a *refinement*
        rather than an independent refit — the premise an EM M-step needs.
        A fresh optimiser is built over the existing parameters (momentum
        restarts, the point is kept), data-derived standardisation buffers
        freeze, and an incompatible shape raises rather than silently
        rebuilding.  Mechanisms whose fit is the exact closed-form maximiser
        accept it as a documented no-op — including the *root* branches of
        MDN, neural-categorical and FlexCode, which must keep tracking the
        E-step.  Each mechanism reports ``warm_started`` in the metrics it
        returns; see :mod:`nbn.learning.warm_start` for the full contract.

        It is an explicit parameter rather than a ``**kwargs`` passenger on
        purpose: passed through ``kwargs`` it would be swallowed in silence by
        any mechanism that did not implement it, which is the same silent
        failure the ``supports_weights`` check exists to prevent.  No
        fail-fast scan is needed here, though, because every mechanism accepts
        it.

        ``method="joint"`` optimises the parameters that are already there and
        never rebuilds, so it is inherently warm; the flag is accepted and has
        no effect on that path.

    Returns
    -------
    TrainHistory
    """
    if device is None:
        device = str(model.device)
    dev = torch.device(device)
    model.to(dev)

    # Move data to device
    data_dev: Dict[str, torch.Tensor] = {
        k: v.to(dev) for k, v in data.items()
    }

    n_rows = next(iter(data_dev.values())).shape[0]
    w_vec = validate_weights(weights, n_rows, where="fit")
    if w_vec is not None:
        w_vec = w_vec.to(dev)
        # Fail before any node is fitted, not after half the network is done.
        # The check has to be explicit: every fit_local ends in **kwargs, so a
        # `weights=` a mechanism does not implement would be swallowed in
        # silence and produce an unweighted fit that looks like a converged
        # weighted one.  Name both the class and the node, because a network
        # has many of each and "kNN does not support weights" does not say
        # which one to swap.
        unsupported = [
            (node, type(model.mechanisms[node]).__name__)
            for node in model.dag.topological_order()
            if node in model.mechanisms
            and not getattr(model.mechanisms[node], "supports_weights", False)
        ]
        if unsupported:
            listed = ", ".join(f"{cls} at node '{node}'" for node, cls in unsupported)
            raise NotImplementedError(
                f"weights= was given, but these mechanisms do not support "
                f"per-sample weights: {listed}.  Swap them for a weighting "
                f"mechanism (ConditionalKDE rather than KNNConditional, say), "
                f"or fit without weights."
            )

    history = TrainHistory(device=device)
    t0 = time.perf_counter()

    if method == "local":
        for i, node in enumerate(model.dag.topological_order()):
            if node not in model.mechanisms:
                raise RuntimeError(f"No mechanism registered for node '{node}'.")
            mech = model.mechanisms[node]
            parents = model.dag.parents(node)
            pa_tensor = pack_parents(data_dev, parents)
            x = data_dev[node]

            # None = "no global override": the mechanism keeps its own
            # designed training budget (flow 300 epochs @ lr 5e-4, MDN 200,
            # neural-categorical 100, ...). A former setdefault() here always
            # fired (the keys were function defaults, never missing), silently
            # flattening every mechanism to one global budget.
            mech_kwargs = dict(kwargs)
            if epochs is not None:
                mech_kwargs["epochs"] = epochs
            if lr is not None:
                mech_kwargs["lr"] = lr
            if batch_size is not None:
                mech_kwargs["batch_size"] = batch_size
            mech_kwargs["consolidate"] = consolidate
            mech_kwargs["warm_start"] = warm_start
            if w_vec is not None:
                mech_kwargs["weights"] = w_vec

            # Bug 1a (#127): thread declared cardinalities so tabular
            # mechanisms span the full declared range rather than truncating
            # to observed-distinct values (which caused query-time factor-axis
            # mismatches on networks with rare states). Mechanisms that don't
            # need them ignore the kwargs. Only supplied when available so the
            # mechanism keeps its observed-data fallback otherwise.
            node_var = model.variables.get(node)
            if node_var is not None and node_var.cardinality is not None:
                mech_kwargs.setdefault("n_classes", int(node_var.cardinality))
            parent_vars = [model.variables.get(p) for p in parents]
            if parents and all(
                pv is not None and pv.cardinality is not None for pv in parent_vars
            ):
                mech_kwargs.setdefault(
                    "parent_cards", [int(pv.cardinality) for pv in parent_vars]
                )

            metrics = mech.fit_local(x, pa_tensor, **mech_kwargs)

            # In-sample (training) mean log-likelihood, for logging only.
            # This was labelled "held-out LL", which it has never been: it is
            # scored on the very rows just fitted.  ``fit`` has no validation
            # split and no early stopping — callers who need a held-out
            # estimate must hold data out themselves and score it with
            # ``NeuralBayesianNetwork.log_prob``.
            with torch.no_grad():
                lp = mech.log_prob(x, pa_tensor)
                mean_ll = float(lp.mean())
            history.node_log_likelihoods.setdefault(node, []).append(mean_ll)

            if (i + 1) % log_every == 0 or (i + 1) == len(model.dag.topological_order()):
                logger.info(
                    "Fitted %d/%d nodes. Last: '%s' ll=%.3f",
                    i + 1, len(model.dag.topological_order()), node, mean_ll
                )

    elif method == "joint":
        # Joint training has no per-mechanism budget to defer to; None
        # resolves to the historical joint defaults.
        epochs = 100 if epochs is None else epochs
        batch_size = 4096 if batch_size is None else batch_size
        lr = 1e-3 if lr is None else lr
        params = list(model.parameters())
        if not params:
            # torch would raise "optimizer got an empty parameter list", which
            # says nothing about the cause.  Most mechanisms build their
            # parameters inside fit_local from the data's shape
            # (LinearGaussianMechanism._weight, MDNMechanism.net and
            # NormalizingFlowMechanism._flow are all None until then), so a
            # never-fitted model genuinely has nothing to optimise jointly.
            raise RuntimeError(
                "method='joint' has no parameters to optimise: no mechanism "
                "has been fitted yet, and most build their parameters lazily "
                "inside fit_local from the data's shape.  Run "
                "model.fit(data) once with the default method='local' first, "
                "then refine with method='joint'."
            )
        opt = torch.optim.Adam(params, lr=lr)
        nodes = model.dag.topological_order()
        n = next(iter(data_dev.values())).shape[0]
        for ep in range(epochs):
            perm = torch.randperm(n, device=dev)
            epoch_loss = 0.0
            steps = 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                batch = {k: v[idx] for k, v in data_dev.items()}
                # Same idx as the batch: this loop owns the permutation, so
                # the weights are gathered here or they desynchronise.
                bw = select(w_vec, idx)
                loss = torch.tensor(0.0, device=dev)
                for node in nodes:
                    mech = model.mechanisms[node]
                    parents = model.dag.parents(node)
                    pa = pack_parents(batch, parents)
                    lp = mech.log_prob(batch[node], pa)
                    loss = loss - weighted_mean(lp, bw)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                epoch_loss += float(loss.item())
                steps += 1
            if (ep + 1) % log_every == 0:
                logger.info("Epoch %d/%d  loss=%.4f", ep + 1, epochs, epoch_loss / max(steps, 1))
        # Record the same per-node in-sample LL the "local" path records, so
        # ``TrainHistory.mean_ll`` is meaningful after either method.  The
        # joint path used to leave ``node_log_likelihoods`` empty, and
        # ``mean_ll`` silently returned NaN for every node — indistinguishable
        # from a genuinely degenerate fit.  One extra full-data forward pass,
        # matching what "local" already costs per node.
        with torch.no_grad():
            for node in nodes:
                mech = model.mechanisms[node]
                pa_tensor = pack_parents(data_dev, model.dag.parents(node))
                lp = mech.log_prob(data_dev[node], pa_tensor)
                history.node_log_likelihoods.setdefault(node, []).append(float(lp.mean()))
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'local' or 'joint'.")

    history.wall_clock_s = time.perf_counter() - t0
    return history
