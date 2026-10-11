"""Ground-truth simulators for the SCM tests (known noise, known confounding)."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from nbn.scm import DiscreteAdditiveNoiseMechanism, StructuralCausalModel

N_HEALTH, N_FATIGUE, N_ACTION = 4, 2, 2
TRUE_Q = torch.tensor([0.15, 0.70, 0.15])  # offsets -1, 0, +1


def rehab_location(health: torch.Tensor, fatigue: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Training helps when rested and hurts when tired."""
    return (health + action * (1 - fatigue) - action * fatigue).clamp(0, N_HEALTH - 1)


@dataclass
class RehabData:
    data: dict
    true_table: torch.Tensor  # [R, K] true P(health' | health, fatigue, action)


def rehab_iid(n: int, q: torch.Tensor = TRUE_Q, generator: torch.Generator | None = None) -> RehabData:
    """i.i.d. rows of ``health' = clip(g(health, fatigue, action) + U)``."""
    h = torch.randint(0, N_HEALTH, (n,), generator=generator)
    f = torch.randint(0, N_FATIGUE, (n,), generator=generator)
    a = torch.randint(0, N_ACTION, (n,), generator=generator)
    u = torch.multinomial(q, n, replacement=True, generator=generator) - 1
    h_next = (rehab_location(h, f, a) + u).clamp(0, N_HEALTH - 1)
    data = {"h": h.float(), "f": f.float(), "a": a.float(), "h'": h_next.float()}
    return RehabData(data, rehab_true_table(q))


def rehab_true_table(q: torch.Tensor = TRUE_Q) -> torch.Tensor:
    grid = torch.cartesian_prod(
        torch.arange(N_HEALTH), torch.arange(N_FATIGUE), torch.arange(N_ACTION)
    )
    loc = rehab_location(grid[:, 0], grid[:, 1], grid[:, 2])
    table = torch.zeros(grid.shape[0], N_HEALTH, dtype=torch.float64)
    for offset, prob in zip((-1, 0, 1), q.double()):
        table[torch.arange(grid.shape[0]), (loc + offset).clamp(0, N_HEALTH - 1)] += prob
    return table


def rehab_model(device: str = "cpu", max_offset: int = 1) -> StructuralCausalModel:
    model = StructuralCausalModel(
        [("h", "h'"), ("f", "h'"), ("a", "h'")],
        {"h": ("discrete", N_HEALTH), "f": ("discrete", N_FATIGUE),
         "a": ("discrete", N_ACTION), "h'": ("discrete", N_HEALTH)},
        device=device,
    )
    model.set_mechanism("h'", DiscreteAdditiveNoiseMechanism(max_offset=max_offset))
    model.auto_mechanisms()
    return model


def confounded_regimes(
    n_per_regime: int, generator: torch.Generator | None = None
) -> dict:
    """Three loggers on one environment with a hidden strength ``W``.

    ``W`` raises the training effect.  Regime 0 is random, regime 1 uses only
    the observed state, regime 2 (the "doctor") trains mostly the strong
    patients — it sees ``W``, so its data are confounded.
    """
    rows = []
    for regime in range(3):
        n = n_per_regime
        h = torch.randint(0, N_HEALTH, (n,), generator=generator)
        w = (torch.rand(n, generator=generator) < 0.4).long()  # strong
        if regime == 0:
            a = (torch.rand(n, generator=generator) < 0.5).long()
        elif regime == 1:
            a = (h < 2).long()
        else:
            a = (torch.rand(n, generator=generator) < torch.where(w == 1, 0.9, 0.1)).long()
        u = torch.multinomial(TRUE_Q, n, replacement=True, generator=generator) - 1
        h_next = (h + a * (2 * w - 1) + u).clamp(0, N_HEALTH - 1)
        rows.append((h, a, h_next, torch.full((n,), regime)))
    h, a, h_next, c = (torch.cat(t) for t in zip(*rows))
    return {"h": h.float(), "a": a.float(), "h'": h_next.float(), "c": c.float()}


def random_chain_episodes(
    n_episodes: int, horizon: int, generator: torch.Generator | None = None
) -> tuple[dict, torch.Tensor]:
    """Episodes of a 3-state, 2-action chain under a uniform random policy.

    Returns ``(transitions, true_table)`` with the table indexed by
    ``(s, a)`` rows in C order.
    """
    n_states, n_actions = 3, 2
    kernel = torch.tensor([
        [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]],
        [[0.3, 0.4, 0.3], [0.05, 0.15, 0.8]],
        [[0.5, 0.0, 0.5], [0.2, 0.2, 0.6]],
    ], dtype=torch.float64)  # [S, A, S']
    s = torch.zeros(n_episodes, horizon + 1, dtype=torch.long)
    a = torch.randint(0, n_actions, (n_episodes, horizon), generator=generator)
    s[:, 0] = torch.randint(0, n_states, (n_episodes,), generator=generator)
    for t in range(horizon):
        probs = kernel[s[:, t], a[:, t]]
        s[:, t + 1] = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    from nbn.scm import transitions_from_episodes

    data = transitions_from_episodes({"s": s.float()}, {"a": a.float()})
    return data, kernel.reshape(n_states * n_actions, n_states)
