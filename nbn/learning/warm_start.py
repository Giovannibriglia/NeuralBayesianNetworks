"""Warm-starting ``fit_local``: what it means, and the compatibility guards.

By default every gradient-trained mechanism rebuilds its network with a fresh
random initialisation and a fresh optimiser on *every* ``fit_local`` call.  A
caller that invokes ``fit_local`` iteratively therefore does not get successive
refinement — it gets an independent refit each time.  ``warm_start=True`` says
"continue from the parameters you already have" instead.

The motivating use is the M-step of an EM loop, which must *increase*
``Q(theta | theta_old)` starting from ``theta_old``.  Under a fresh
initialisation that premise is simply false, and everything built on it — a
backtracking line search, a monotonicity guard — inverts: retrying at a
smaller learning rate does not produce a gentler step, it produces a worse
fresh fit, monotonically in the reduction.

Semantics
---------
**Optimiser state is not carried over.**  Each call builds a fresh
``torch.optim.Adam`` over the *existing* parameters: the point is preserved,
the momentum restarts.  The decisive reason is that optimiser moments are not
in ``state_dict()``.  A caller running the supported backtracking idiom
(``copy.deepcopy(state_dict())`` → step → ``load_state_dict`` on rejection,
see :mod:`tests.unit.test_parameter_snapshot_contract`) reverts the
*parameters* but could not revert the moments, so a rejected step would leave
its momentum alive and silently corrupt the very guard warm-starting exists to
repair.  Two supporting reasons: optimiser state survives neither
``torch.save``/``load`` nor ``intervene()``'s deepcopy, so persisting it would
create state that exists in one process only; and the M-step objective changes
between calls (the sample weights change), so the moments are stale by
construction.

**Data-derived standardisation buffers freeze.**  ``MDNMechanism``'s
``_pa_mean``/``_pa_std``, ``FlexCodeMechanism``'s the same plus its
``_y_min``/``_y_max`` z-space bounds, are *not* recomputed under a warm start.
The network's weights were trained against the map those buffers define;
recomputing them applies every learned weight to a shifted, rescaled input, so
the warm start would be warm in name only — and the damage would be largest
exactly when the weights moved most.  FlexCode's bounds are the more
load-bearing half: its MLP predicts orthogonal-basis coefficients in the
z-space ``(_y_min, _y_max)`` fixes, so changing them *reinterprets* every
learned coefficient rather than merely rescaling an input.

The cost is that the scaling can drift from the data.  It is bounded in the
motivating use: across EM M-steps the rows are identical and only the weights
change.  A caller who wants re-standardisation passes ``warm_start=False``.

**Shape incompatibility raises.**  Never a silent rebuild — a caller who
believes it is continuing and is not, with nothing in the output to say so, is
the failure this whole feature exists to remove.  See :func:`check_shapes` and
:func:`check_branch`.

**A never-fitted mechanism cold-builds** rather than raising: there is nothing
to discard, and ``for it in range(T): fit(..., warm_start=True)`` is the
natural call shape.  To keep that observable rather than silent, every
``fit_local`` reports ``warm_started: bool`` in its returned metrics dict.  It
is ``True`` only when parameters were actually carried over.

Which mechanisms warm-start — a *branch*-level property
--------------------------------------------------------
Only four of the twelve concrete mechanisms have an initialisation to continue
from at all: ``MDNMechanism``, ``NormalizingFlowMechanism``,
``NeuralCategoricalMechanism`` and ``FlexCodeMechanism``.  Everything else
computes the exact maximiser of the local (weighted) objective in closed form,
or stores the training sample itself, so "continue from theta_old" and
"recompute" give the same answer: there ``warm_start=True`` is an accepted,
documented no-op, reported as ``warm_started: False``.

Crucially this is a property of the *branch*, not the class.  The **root**
branches of MDN, neural-categorical and FlexCode are closed form — a root MDN
is initialised analytically from the data moments and never enters a training
loop at all.  Freezing those under a warm start would stop them responding to
the E-step: a silent, permanent M-step failure on exactly the nodes nobody
inspects.  They recompute.  The bitwise ``epochs=0`` contract still holds for
them, because recomputing from unchanged data is deterministic.

``Mechanism.warm_start_is_noop`` describes the non-root branch; root branches
are always no-ops.

Note on the aliasing trap
-------------------------
Warm-starting makes ``test_parameter_snapshot_contract``'s deepcopy
requirement reach further.  A cold refit allocates *new* parameter tensors, so
an uncopied ``snap = mech.state_dict()`` accidentally survived a subsequent
``fit_local``.  Under ``warm_start=True`` the objects persist and the fit
mutates them in place, so the snapshot is now clobbered by the very call it
was taken to undo.  ``copy.deepcopy`` was previously load-bearing only across
optimiser steps; it is now load-bearing across ``fit_local`` calls too.

Note on ``consolidate``
-----------------------
``fit``/``fit_local`` default to ``consolidate=True``, which snapshots EWC
state (theta* plus a diagonal Fisher) after every fit — up to ``sample_cap``
*sequential* per-sample backward passes.  ``warm_start`` does not change that,
deliberately.  A caller running ``fit_local`` in a loop pays that cost on
every iteration and almost certainly wants ``consolidate=False``, re-enabling
it only for the final fit if ``model.update()`` will be used later.
"""
from __future__ import annotations

from typing import Mapping, Tuple


def check_shapes(where: str, dims: Mapping[str, Tuple[int, int]]) -> None:
    """Raise unless every ``name: (existing, requested)`` pair agrees.

    Parameters
    ----------
    where:
        Caller name, quoted in the error message.
    dims:
        Maps a dimension's name to ``(value the existing fit has, value this
        call supplies)``.

    Raises
    ------
    ValueError
        Naming the mechanism, the dimension, and *both* values — a message
        that says only "shape mismatch" leaves the caller to guess which of
        several dimensions moved and in which direction.
    """
    for name, (have, want) in dims.items():
        if int(have) != int(want):
            raise ValueError(
                f"{where}: warm_start=True, but the existing fit has "
                f"{name}={int(have)} and this call supplies {name}={int(want)}. "
                f"Warm-starting reuses the existing parameters, which are "
                f"shaped for {name}={int(have)}, so there is nothing to "
                f"continue from here.  Pass warm_start=False to refit from a "
                f"fresh initialisation."
            )


def check_branch(where: str, *, was_root: bool, is_root: bool) -> None:
    """Raise if a warm start would cross the root / non-root branch boundary.

    A root fit and a conditioned fit produce different parameters entirely
    (analytic moments or a marginal table versus a conditioner network), so
    one cannot continue the other.  Silently rebuilding is the failure mode
    this guard exists to prevent, and a bare shape check would not catch it:
    ``d_pa`` moving 0 → 3 reads as a dimension change, but 3 → 0 would slip
    through a check written only over the parents' width.
    """
    if bool(was_root) != bool(is_root):
        had, now = (
            ("no parents (root)", "parents") if was_root
            else ("parents", "no parents (root)")
        )
        raise ValueError(
            f"{where}: warm_start=True, but the existing fit was made with "
            f"{had} and this call supplies {now}.  Those are different sets of "
            f"parameters, not a continuation of one another.  Pass "
            f"warm_start=False to refit."
        )
