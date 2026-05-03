from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

import torch

from nbn.utils.batching import pack_parents

if TYPE_CHECKING:
    from nbn.core.network import NeuralBayesianNetwork

logger = logging.getLogger(__name__)


@dataclass
class TrainHistory:
    """Records per-epoch training metrics."""

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
    epochs: int = 100,
    batch_size: int = 4096,
    lr: float = 1e-3,
    device: str | None = None,
    log_every: int = 10,
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
        Number of training epochs (used by neural mechanisms).
    batch_size:
        Mini-batch size for gradient-based methods.
    lr:
        Learning rate.
    device:
        Target device; data is moved here.
    log_every:
        Log progress every N nodes.

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

    history = TrainHistory(device=device)
    t0 = time.perf_counter()

    if method == "local":
        for i, node in enumerate(model.dag.topological_order()):
            mech = model.mechanisms.get(node)
            if mech is None:
                raise RuntimeError(f"No mechanism registered for node '{node}'.")
            parents = model.dag.parents(node)
            pa_tensor = pack_parents(data_dev, parents)
            x = data_dev[node]

            mech_kwargs = dict(kwargs)
            mech_kwargs.setdefault("epochs", epochs)
            mech_kwargs.setdefault("lr", lr)
            mech_kwargs.setdefault("batch_size", batch_size)

            metrics = mech.fit_local(x, pa_tensor, **mech_kwargs)

            # Compute held-out LL on full data for logging
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
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        nodes = model.dag.topological_order()
        n = next(iter(data_dev.values())).shape[0]
        for ep in range(epochs):
            perm = torch.randperm(n, device=dev)
            epoch_loss = 0.0
            steps = 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                batch = {k: v[idx] for k, v in data_dev.items()}
                loss = torch.tensor(0.0, device=dev)
                for node in nodes:
                    mech = model.mechanisms[node]
                    parents = model.dag.parents(node)
                    pa = pack_parents(batch, parents)
                    lp = mech.log_prob(batch[node], pa)
                    loss = loss - lp.mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                epoch_loss += float(loss.item())
                steps += 1
            if (ep + 1) % log_every == 0:
                logger.info("Epoch %d/%d  loss=%.4f", ep + 1, epochs, epoch_loss / max(steps, 1))
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'local' or 'joint'.")

    history.wall_clock_s = time.perf_counter() - t0
    return history
