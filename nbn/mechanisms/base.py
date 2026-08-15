from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.distributions import Distribution


class Mechanism(nn.Module, ABC):
    """Abstract base class for a learnable conditional distribution P(X | pa(X)).

    Design contract
    ---------------
    * ``forward(parents)`` returns a ``torch.distributions.Distribution`` whose
      ``batch_shape`` starts with ``[B]`` (or ``[B, S]`` when parents is
      ``[B, S, D_pa]``).
    * ``log_prob(x, parents)`` → ``[B]`` or ``[B, S]``.
    * ``sample(parents, n)`` → ``[B, S, D_x]``.
    * Every parameter is an ``nn.Parameter`` — autograd, ``.to(device)``,
      ``torch.compile`` all work for free.

    Shape conventions
    -----------------
    parents: ``[B, D_pa]`` (2-D) or ``[B, S, D_pa]`` (3-D with particle dim).
    x:       ``[B, D_x]``  (2-D) or ``[B, S, D_x]`` (3-D).

    Snapshotting and restoring parameters
    -------------------------------------
    To take a step and be able to reject it — a monotone generalized-EM loop
    backtracking an M-step that decreased the objective — use torch's own
    ``state_dict`` / ``load_state_dict``.  There is no NBN-specific API for
    this and none is needed, but **the snapshot must be copied**::

        snap = copy.deepcopy(mech.state_dict())   # or {k: v.clone() for ...}
        ...                                       # optimiser step
        mech.load_state_dict(snap)                # reverts exactly

    ``state_dict()`` returns tensors that *share storage* with the live
    parameters, and optimisers update parameters in place, so an uncopied
    snapshot is mutated by the very step it is supposed to undo.  Nothing
    raises; the parameters simply fail to revert, and a backtracking M-step
    would silently accept every step it meant to reject.

    The round trip covers parameters *and* buffers (``LinearGaussianMechanism``'s
    normal equations, ``MDNMechanism``'s standardisation statistics).  It does
    not cover fit-time structural attributes such as ``_parent_cards`` or
    ``_n_classes`` — those describe the CPD's shape, are set once at fit time,
    and are not altered by an optimiser step.  Pinned by
    ``tests/unit/test_parameter_snapshot_contract.py``.
    """

    is_discrete: bool = False
    output_dim: int = 1
    # Whether this mechanism supports incremental, no-rehearsal updates via
    # ``update_local``.  Mechanisms that persist sufficient statistics inside
    # ``fit_local`` (e.g. categorical counts, linear-Gaussian normal equations)
    # set this True; ``nbn.update.orchestrate`` skips any node where it is False.
    supports_update: bool = False
    # Whether ``fit_local`` honours a per-sample ``weights`` vector.  Checked
    # by ``nbn.learning.fit`` *before* any node is fitted, so an unsupported
    # mechanism fails fast instead of after half the network is done.
    #
    # It has to be an explicit capability rather than a convention: every
    # ``fit_local`` ends in ``**kwargs``, so a ``weights=`` a mechanism does
    # not implement would be silently swallowed — yielding an unweighted fit
    # that looks like a converged weighted one, which is the worst outcome
    # available here.
    supports_weights: bool = False

    @abstractmethod
    def forward(
        self, parents: torch.Tensor | None
    ) -> Distribution:
        """Return the conditional distribution P(X | parents).

        Parameters
        ----------
        parents:
            Conditioning context of shape ``[B, D_pa]`` or ``[B, S, D_pa]``.
            ``None`` for root nodes (no parents).

        Returns
        -------
        Distribution
            A batched distribution with batch_shape starting with ``[B]``.
        """

    def log_prob(
        self, x: torch.Tensor, parents: torch.Tensor | None
    ) -> torch.Tensor:
        """Log conditional probability log P(x | parents).

        Parameters
        ----------
        x: shape ``[B, D_x]`` or ``[B, S, D_x]``.
        parents: shape ``[B, D_pa]`` or ``[B, S, D_pa]``, or ``None``.

        Returns
        -------
        torch.Tensor of shape ``[B]`` or ``[B, S]``.
        """
        return self.forward(parents).log_prob(x)

    def sample(
        self, parents: torch.Tensor | None, n: int = 1
    ) -> torch.Tensor:
        """Draw ``n`` samples per row.

        Parameters
        ----------
        parents: shape ``[B, D_pa]``, or ``None`` for root nodes.
        n: number of samples per batch element.

        Returns
        -------
        torch.Tensor of shape ``[B, n, D_x]``.
        """
        dist = self.forward(parents)
        # dist.batch_shape = [B]; sample n → [n, B, D_x] → permute to [B, n, D_x]
        s = dist.sample((n,))
        if s.dim() == 2:
            # [n, B] → [B, n, 1]
            return s.T.unsqueeze(-1)
        if s.dim() == 3:
            # [n, B, D_x] → [B, n, D_x]
            return s.permute(1, 0, 2)
        return s

    def rsample(
        self, parents: torch.Tensor | None, n: int = 1
    ) -> torch.Tensor:
        """Reparameterized sample (differentiable).  Raises if not supported."""
        dist = self.forward(parents)
        if not dist.has_rsample:
            raise RuntimeError(
                f"{type(self).__name__} does not support reparameterized sampling"
            )
        s = dist.rsample((n,))
        if s.dim() == 2:
            return s.T.unsqueeze(-1)
        if s.dim() == 3:
            return s.permute(1, 0, 2)
        return s

    @abstractmethod
    def fit_local(
        self, x: torch.Tensor, parents: torch.Tensor | None, **kwargs
    ) -> dict:
        """Closed-form or small-loop local MLE.  Returns a dict of metrics.

        Implementations that set ``supports_weights = True`` accept a
        ``weights`` keyword: a non-negative ``[N]`` tensor of per-sample
        multiplicities aligned with ``x``.  The contract is replication
        equivalence — fitting with integer weights must equal fitting on the
        data with each row repeated that many times — and a weight of exactly
        0 must be indistinguishable from dropping the row.  See
        :mod:`nbn.learning.weighting`.
        """

    def update_local(
        self, x: torch.Tensor, parents: torch.Tensor | None, **kwargs
    ) -> dict:
        """Fold new data into the already-fitted CPD (no rehearsal).

        Default implementation raises — only mechanisms that persist the
        sufficient statistics they need inside ``fit_local`` (and set
        ``supports_update = True``) override this.  ``nbn.update.orchestrate``
        never calls this on a mechanism whose ``supports_update`` is False, so
        this signals a direct mis-call rather than a routing bug.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support incremental update"
        )

    @property
    def is_fitted(self) -> bool:
        """True iff ``fit_local`` has produced a usable CPD.

        Default ``False``.  Each concrete mechanism that supports
        fitting overrides this with a check appropriate for its
        internal state (e.g. ``self._logits is not None`` for
        ``CategoricalTableMechanism``; presence of ``_root_logits``
        or a fitted ``net`` for ``NeuralCategoricalMechanism``).

        Used by the variable-elimination engine
        (``nbn/inference/tensor_ve.py:_extract_factors``) to surface
        a clear ``RuntimeError`` rather than letting an unfitted
        mechanism propagate as an opaque ``AssertionError`` from
        deeper inside ``tabulate()``.  See v0.8 issue #26.
        """
        return False

    def tabulate(
        self, parent_cards: list[int] | None = None
    ) -> torch.Tensor:
        """Return the tabulated CPD as a tensor.

        Shape ``[*parent_cards, K]`` for non-root discrete mechanisms;
        ``[K]`` for root.  Returned values are in logit space — apply
        ``softmax`` along the last axis to get a normalised CPD.  Some
        mechanisms (e.g. ``CategoricalTableMechanism``) store log-probs
        internally; others (``NeuralCategoricalMechanism``) return
        unnormalised logits.  Both are equivalent under softmax
        shift-invariance.

        ``parent_cards`` is the cardinality of each parent in DAG
        order; required for mechanisms that enumerate via
        ``forward()`` (e.g. neural categorical), and may be ignored
        by mechanisms that already store the tabulation
        (e.g. categorical table).

        Default raises ``NotImplementedError`` — only meaningful for
        discrete mechanisms.  Used by:

        * The benchmarking metric site
          (benchmarking/measurements/accuracy_timing.py)
        * The variable-elimination engine
          (``nbn/inference/tensor_ve.py:_extract_factors``)

        Both previously read ``mech._logits`` directly, which broke
        on mechanisms whose CPD is computed per-call rather than
        stored.  See v0.8 issues #59 and #26.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.tabulate() is only defined for "
            f"discrete mechanisms."
        )
