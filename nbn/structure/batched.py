"""Train the MLPs of several candidate parent sets of one node together.

All candidates of a node read the **union** of their parents.  Candidate
``c`` sees it through a fixed 0/1 mask ``m_c``; its weights are one slice of
stacked tensors ``[C, ...]``, so one forward pass evaluates every candidate on
the batch.  The loss is the **sum** of the per-candidate losses and the
parameters are disjoint, so the gradient of candidate ``c`` is the gradient of
its own loss; Adam is element-wise, so each candidate follows the trajectory
it would follow alone with the same initial values and the same batch order
(the ``C = 1`` run of this trainer is the reference).

Masking cannot leak: the first-layer weights of a candidate's unused inputs are
zero and multiplied by the mask in every forward pass, so they get no gradient
and no input value of an unused parent reaches the candidate.  The first layer
is initialised with each candidate's own fan-in (``nn.Linear``'s default,
uniform in ``±1/sqrt(fan_in)``), so a candidate starts where a separately
built network of its own input size would.

After training, :meth:`StackedMLP.export` returns candidate ``c`` as an
ordinary ``nn.Sequential`` over its own parents (the layout of the NBN
mechanisms), so it can be dropped into a standard mechanism.
"""
from __future__ import annotations

import math
from typing import List, Literal, Sequence

import torch
import torch.nn as nn

Activation = Literal["relu", "silu", "tanh", "gelu", "elu"]
_ACTIVATIONS = {"relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh, "gelu": nn.GELU, "elu": nn.ELU}


class StackedMLP(nn.Module):
    """``C`` MLPs with the same hidden sizes, stacked along a leading axis.

    Parameters
    ----------
    masks:
        ``[C, D_in]`` 0/1 masks: which union inputs each candidate reads.
    hidden:
        Hidden layer widths (shared by all candidates).
    out_dim:
        Output width.
    activation:
        Name of the activation between layers.
    generators:
        One CPU generator per candidate for the initial weights (so a
        candidate's start does not depend on the others).
    input_orders:
        Per candidate, its union columns **in the candidate's own order**.
        The first-layer rows are drawn in that order and scattered to the
        union positions, so a candidate starts exactly where a network built
        on its own inputs would, whatever the union and its siblings.
        Default: the active columns in union order.
    """

    def __init__(
        self,
        masks: torch.Tensor,
        hidden: Sequence[int],
        out_dim: int,
        activation: Activation = "relu",
        generators: Sequence[torch.Generator] | None = None,
        input_orders: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}; got {activation!r}.")
        masks = masks.to(torch.float32)
        n_cand, d_in = masks.shape
        self.activation = activation
        self.act = _ACTIVATIONS[activation]()
        self.register_buffer("mask", masks)
        sizes = [d_in, *[int(h) for h in hidden], int(out_dim)]
        if generators is None:
            generators = [torch.Generator().manual_seed(c) for c in range(n_cand)]
        if len(generators) != n_cand:
            raise ValueError("one generator per candidate is needed.")
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        layer_w: List[List[torch.Tensor]] = [[] for _ in sizes[:-1]]
        layer_b: List[List[torch.Tensor]] = [[] for _ in sizes[:-1]]
        if input_orders is None:
            input_orders = [torch.nonzero(masks[c]).reshape(-1).tolist() for c in range(n_cand)]
        for c in range(n_cand):
            order = list(input_orders[c])
            if sorted(order) != torch.nonzero(masks[c]).reshape(-1).tolist():
                raise ValueError(f"input_orders[{c}] does not match the mask's active columns.")
            fan_ins = [len(order), *sizes[1:-1]]
            for li, (d_a, d_b) in enumerate(zip(sizes[:-1], sizes[1:])):
                fan_in = fan_ins[li]
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                if li == 0:
                    # Draw only the active rows, so a candidate's start depends on its
                    # own inputs and generator, never on the union or its siblings.
                    w = torch.zeros(d_a, d_b)
                    if order:
                        w[torch.tensor(order)] = (torch.rand(fan_in, d_b, generator=generators[c]) * 2 - 1) * bound
                else:
                    w = (torch.rand(d_a, d_b, generator=generators[c]) * 2 - 1) * bound
                b = (torch.rand(d_b, generator=generators[c]) * 2 - 1) * bound
                layer_w[li].append(w)
                layer_b[li].append(b)
        for ws, bs in zip(layer_w, layer_b):
            self.weights.append(nn.Parameter(torch.stack(ws)))
            self.biases.append(nn.Parameter(torch.stack(bs)))

    @property
    def n_candidates(self) -> int:
        return int(self.mask.shape[0])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z`` ``[N, D_in]`` (union inputs) -> ``[C, N, out]``."""
        h = z.unsqueeze(0).expand(self.n_candidates, -1, -1)
        n_layers = len(self.weights)
        for li in range(n_layers):
            w = self.weights[li]
            if li == 0:
                w = w * self.mask.unsqueeze(-1)       # unused inputs: no value, no gradient
            h = torch.baddbmm(self.biases[li].unsqueeze(1), h, w)
            if li < n_layers - 1:
                h = self.act(h)
        return h

    def export(self, c: int, order: Sequence[int] | None = None) -> nn.Sequential:
        """Candidate ``c`` as an ``nn.Sequential`` over its own inputs.

        Inputs are in union order, or in ``order`` (a permutation of the active
        union columns) when given.  The layers are assembled on CPU from
        contiguous copies and then moved to the parameters' device: strided
        copies (row selection, transposes) gave wrong weights on some MPS builds.
        """
        device = self.weights[0].device
        active = torch.nonzero(self.mask[c].detach().cpu()).reshape(-1).tolist()
        if order is None:
            order = active
        elif sorted(order) != active:
            raise ValueError(f"order {list(order)} is not a permutation of the active columns {active}.")
        layers: List[nn.Module] = []
        n_layers = len(self.weights)
        for li in range(n_layers):
            w = self.weights[li][c].detach().cpu().clone()
            if li == 0:
                w = w.index_select(0, torch.tensor(list(order), dtype=torch.long))
            weight = w.t().contiguous()
            bias = self.biases[li][c].detach().cpu().clone().contiguous()
            lin = nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                lin.weight.copy_(weight)
                lin.bias.copy_(bias)
            layers.append(lin)
            if li < n_layers - 1:
                layers.append(_ACTIVATIONS[self.activation]())
        return nn.Sequential(*layers).to(device)


def train_stacked(
    net: StackedMLP,
    z: torch.Tensor,
    y: torch.Tensor,
    loss: Literal["mse", "cross_entropy"],
    weights: torch.Tensor | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int | None = None,
    order_generator: torch.Generator | None = None,
) -> List[float]:
    """Minimise the sum of the candidates' (weighted-mean) losses with one Adam.

    ``z`` ``[N, D_in]``; ``y`` ``[N, D_out]`` for ``"mse"`` (squared error summed
    over output dims, as ``AdditiveNoiseMechanism``) or ``[N]`` class indices for
    ``"cross_entropy"`` (as ``NeuralCategoricalMechanism``).  Memory: one
    step holds ``[C, batch, hidden]`` activations, ``C`` times a single fit;
    pass ``batch_size`` for many rows.  The batch order
    comes from ``order_generator`` (CPU), shared by every candidate.  Returns
    each candidate's (weighted) mean loss on **all** rows after training.
    """
    from nbn.utils.device import accum_dtype
    if order_generator is None:
        order_generator = torch.Generator().manual_seed(0)
    n = z.shape[0]
    size = n if batch_size is None else int(batch_size)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(int(epochs)):
        order = torch.randperm(n, generator=order_generator).to(z.device)
        for start in range(0, n, size):
            idx = order[start:start + size]
            out = net(z[idx])                                     # [C, B, out]
            w = None if weights is None else weights[idx]
            if loss == "mse":
                err = (out - y[idx].unsqueeze(0)).pow(2).sum(-1)   # [C, B]
            else:
                c, b, k = out.shape
                err = nn.functional.cross_entropy(
                    out.reshape(c * b, k), y[idx].repeat(c), reduction="none").reshape(c, b)
            per_cand = _per_candidate_mean(err, w, accum_dtype(err.device))
            optimiser.zero_grad()
            per_cand.sum().backward()
            optimiser.step()
    # Final full-data loss, in chunks so the [C, N, hidden] activations stay bounded.
    errs = []
    with torch.no_grad():
        for start in range(0, n, 65_536):
            sl = slice(start, start + 65_536)
            out = net(z[sl])
            if loss == "mse":
                errs.append((out - y[sl].unsqueeze(0)).pow(2).sum(-1))
            else:
                c, b, k = out.shape
                errs.append(nn.functional.cross_entropy(
                    out.reshape(c * b, k), y[sl].repeat(c), reduction="none").reshape(c, b))
        final = _per_candidate_mean(torch.cat(errs, -1), weights, accum_dtype(z.device))
    return [float(v) for v in final.cpu()]


def _per_candidate_mean(err: torch.Tensor, w: torch.Tensor | None, acc: torch.dtype) -> torch.Tensor:
    """``[C]`` weighted means of ``err`` ``[C, B]``, as ``nbn.learning.weighting.weighted_mean``.

    Accumulated in the widest float of the device and cast back, so the
    gradient keeps the parameters' dtype.
    """
    if w is None:
        return err.mean(-1)
    from nbn.learning.weighting import _MIN_TOTAL_WEIGHT

    e = err.to(acc)
    ww = w.to(device=err.device, dtype=acc)
    # An all-zero mini-batch contributes 0 (not 0/0), as weighted_mean does.
    return ((e * ww).sum(-1) / ww.sum().clamp_min(_MIN_TOTAL_WEIGHT)).to(err.dtype)
