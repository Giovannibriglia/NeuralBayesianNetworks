"""Turn episodes into one-step transition data (the two slices of an MDP).

A Markov decision process is a two-slice network: the variables at ``t``
(state and action) are parents of the state variables at ``t + 1``.  This
helper flattens episodes into rows ``(s_t, a_t, s_{t+1})`` and keeps the
episode id and the time index, which the diagnostics and the trajectory-aware
probability sets need.
"""
from __future__ import annotations

from typing import Dict, Mapping

import torch


def transitions_from_episodes(
    states: Mapping[str, torch.Tensor],
    actions: Mapping[str, torch.Tensor],
    next_suffix: str = "'",
) -> Dict[str, torch.Tensor]:
    """Flatten episodes into transition rows.

    Parameters
    ----------
    states:
        ``{name: [E, T + 1]}`` (or ``[E, T + 1, D]``) state trajectories.
    actions:
        ``{name: [E, T]}`` (or ``[E, T, D]``) actions taken at ``t = 0 .. T-1``.
    next_suffix:
        Suffix of the next-state variables (``"health"`` → ``"health'"``).

    Returns
    -------
    ``{name: [E*T, ...], name + suffix: [E*T, ...], action: [E*T, ...],
    "episode": [E*T], "time": [E*T]}``.
    """
    first = next(iter(states.values()))
    n_episodes, horizon = first.shape[0], first.shape[1] - 1
    out: Dict[str, torch.Tensor] = {}
    for name, traj in states.items():
        if traj.shape[:2] != (n_episodes, horizon + 1):
            raise ValueError(f"State '{name}' has shape {tuple(traj.shape)}; expected [E, T+1, ...].")
        out[name] = traj[:, :-1].reshape(n_episodes * horizon, *traj.shape[2:])
        out[name + next_suffix] = traj[:, 1:].reshape(n_episodes * horizon, *traj.shape[2:])
    for name, acts in actions.items():
        if acts.shape[:2] != (n_episodes, horizon):
            raise ValueError(f"Action '{name}' has shape {tuple(acts.shape)}; expected [E, T, ...].")
        out[name] = acts.reshape(n_episodes * horizon, *acts.shape[2:])
    device = first.device
    out["episode"] = torch.arange(n_episodes, device=device).repeat_interleave(horizon)
    out["time"] = torch.arange(horizon, device=device).repeat(n_episodes)
    return out
