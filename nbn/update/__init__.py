"""Incremental, no-rehearsal CPD updates for a fitted ``NeuralBayesianNetwork``.

Public entry point is ``model.update(new_data, ...)`` (see
:meth:`nbn.core.network.NeuralBayesianNetwork.update`).  Each mechanism that
supports it persists its own sufficient statistics inside ``fit_local`` and
folds new data in via ``update_local`` — no ``attach``/``consolidate`` call is
needed.  See :mod:`nbn.update.base` for the three constraints and the
posterior-as-prior principle.
"""
from nbn.update.base import ForgettingConfig, UpdateHistory
from nbn.update.orchestrate import update

__all__ = ["update", "UpdateHistory", "ForgettingConfig"]
