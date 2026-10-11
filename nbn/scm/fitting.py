"""Pure tensor functions for discrete additive (offset) noise.

The model is ``x = clip(g + U, 0, K - 1)`` with an integer location ``g`` per
parent configuration and one categorical law ``q`` over the offsets
``-m .. m``.  Everything here works on count tables and is batched over any
leading dimensions, so the same code fits one model or a few hundred bootstrap
replicates at once.  Nothing here knows about mechanisms or networks.

Locations range over ``-m .. K - 1 + m``: under clipping, a location beyond
the value range is a different law (it piles mass on the edge), not a copy of
an edge location.  Location *index* ``l`` stands for ``g = l - m``.

Shapes: ``K`` child values, ``M = 2 m + 1`` offsets, ``G = K + 2 m``
locations, ``R`` parent configurations, ``...`` optional batch dimensions.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def offset_transfer(n_classes: int, offsets: torch.Tensor) -> torch.Tensor:
    """``A[l, k, m] = 1`` iff ``clip(l - m_max + offsets[m], 0, K - 1) == k``; ``[G, K, M]``.

    Row ``l`` is the location ``g = l - m_max``; the child law there is the
    linear image ``A[l] @ q``.
    """
    max_offset = int(offsets.abs().max()) if offsets.numel() else 0
    locations = torch.arange(-max_offset, n_classes + max_offset, device=offsets.device)
    values = (locations.unsqueeze(-1) + offsets.unsqueeze(0)).clamp(0, n_classes - 1)  # [G, M]
    return F.one_hot(values, n_classes).permute(0, 2, 1)


def child_law(transfer: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``P(x = k | location index l)``, shape ``[..., G, K]``."""
    return torch.einsum("gkm,...m->...gk", transfer.to(q.dtype), q)


def rows_table(law: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
    """Per-row table ``[..., R, K]`` from a child law ``[..., G, K]`` and location indices ``[..., R]``."""
    index = locations.unsqueeze(-1).expand(*locations.shape, law.shape[-1])
    return torch.gather(law, -2, index)


def counts_by_location(counts: torch.Tensor, locations: torch.Tensor, n_locations: int) -> torch.Tensor:
    """Sum the rows of ``counts [..., R, K]`` that share a location: ``[..., G, K]``."""
    assign = F.one_hot(locations, n_locations).to(counts.dtype)  # [..., R, G]
    return torch.einsum("...rg,...rk->...gk", assign, counts)


def log_likelihood(counts: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """``sum N log P`` over the last two dimensions, with ``0 log 0 = 0``."""
    return torch.xlogy(counts, table).sum(dim=(-2, -1))


def fit_offset_law(
    transfers: Sequence[torch.Tensor],
    counts: Sequence[torch.Tensor],
    n_iter: int = 200,
    rtol: float = 1e-10,
    max_row_moves: int = 32,
    start: tuple[Sequence[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Maximum likelihood for the locations of every member and one shared ``q``.

    The likelihood ``sum_members sum_{r,k} N[r,k] log (A[g_r] q)_k`` has local
    optima that coordinate ascent cannot leave, because the escape needs a
    location and ``q`` to move together.  So the fit runs in batched stages
    and keeps the best run for every batch element:

    1. *multi-start*: a triangular law peaked at 0, then one start
       concentrated on each offset (ties go to the earlier start);
    2. *shift polish*: the winner's law shifted by ``s = ±1, ±2`` (a location
       moved by ``s`` with the law moved by ``-s``);
    3. *row-move polish*, repeated while it improves: for each of the
       ``max_row_moves`` busiest rows of every member and each feasible
       location, move that row there, refit ``q`` with the locations fixed
       (a concave problem), then resume coordinate ascent.

    With ``start=(locations, q)`` the stages are skipped: coordinate ascent
    runs from that point and from the triangular law, and the better run is
    kept.  The parametric bootstrap uses this — its replicates are simulated
    from the observed fit, so that fit is the right basin — which keeps a
    batch of hundreds of replicates cheap.

    The search is a heuristic: it matched brute-force enumeration in the tests,
    but no optimality guarantee is claimed.  The result is canonicalised (see
    :func:`canonicalise`), so it is reproducible across devices and runs.
    Rows without data keep the middle location (a flagged extrapolation).
    Cells no location can reach are ignored by the EM step;
    :func:`unreachable_cells` detects them.

    Parameters
    ----------
    transfers:
        ``[G_i, K_i, M]`` per member (same offsets for all).
    counts:
        ``[..., R_i, K_i]`` per member, same batch dimensions for all.
    max_row_moves:
        Rows per member tried in stage 3 (0 disables it).
    start:
        Optional warm start ``(location indices per member, q)``, broadcastable
        to the batch.

    Returns
    -------
    (locations, q, mean_log_likelihood):
        location indices ``[..., R_i]`` (long) per member, ``[..., M]``, ``[...]``.
    """
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}.")
    transfers = [a.to(counts[0].dtype) for a in transfers]
    n_offsets = transfers[0].shape[-1]
    batch_shape = counts[0].shape[:-2]
    triangle = _multi_starts(n_offsets, counts[0])[:1]

    if start is not None:
        start_locations = [loc.expand(*batch_shape, loc.shape[-1]).clone() for loc in start[0]]
        start_q = start[1].to(counts[0]).expand(*batch_shape, n_offsets)
        best = _coordinate_ascent(transfers, counts, start_q, n_iter, rtol, locations=start_locations)
        best = _pick(best, _best_of(transfers, counts, triangle, n_iter, rtol))
        return canonicalise(transfers, counts, *best)

    best = _best_of(transfers, counts, _multi_starts(n_offsets, counts[0]), n_iter, rtol)
    shifts = [_shift_law(best[1], s) for s in (-2, -1, 1, 2) if abs(s) < n_offsets]
    if shifts:
        best = _pick(best, _best_of(transfers, counts, torch.stack(shifts), n_iter, rtol,
                                    per_element=True))
    for _ in range(3 if max_row_moves > 0 else 0):
        moved = _row_moves(transfers, counts, best, n_iter, rtol, max_row_moves)
        if moved is None or not bool((moved[2] > best[2] + rtol * best[2].abs().clamp_min(1.0)).any()):
            break
        best = _pick(best, moved)
    return canonicalise(transfers, counts, *best)


def canonicalise(transfers, counts, locations, q, objective):
    """Pick one representative of the fits with exactly the same likelihood.

    ``(g + s, q moved by -s)`` gives the same law for every row whose shifted
    location stays in range, so the split between location and noise is not
    identified.  The canonical fit

    1. moves every saturated row (several locations give its exact law) to
       the equivalent location closest to the value range ``0 .. K - 1``, so
       no row blocks the next step;
    2. shifts the whole group so the median of ``q`` sits as close to offset 0
       as the offset and location ranges allow;
    3. repeats step 1 for the new ``q``.

    No step changes any row's law, so the likelihood is unchanged.  Rows
    without data keep their location.  Batched over ``[...]``.
    """
    max_offset = (q.shape[-1] - 1) // 2
    tol = 1e3 * torch.finfo(q.dtype).eps  # "zero" and "equal" up to rounding
    seen = [c.sum(-1) > 0 for c in counts]
    locations = _snap_saturated(transfers, seen, locations, q, max_offset, tol)
    locations, q = _centre_law(transfers, seen, locations, q, max_offset, tol)
    locations = _snap_saturated(transfers, seen, locations, q, max_offset, tol)
    return locations, q, objective


def _snap_saturated(transfers, seen, locations, q, max_offset: int, tol: float):
    """Each seen row to the location with the same law that is nearest the value range."""
    snapped = []
    for a, loc, s in zip(transfers, locations, seen):
        n_locations, n_classes = a.shape[0], a.shape[1]
        law = child_law(a, q)                                              # [..., G, K]
        same = (law.unsqueeze(-2) - law.unsqueeze(-3)).abs().amax(-1) <= tol  # [..., G, G]
        index = torch.arange(n_locations, device=q.device)
        g_values = index - max_offset
        outside = (g_values.clamp(0, n_classes - 1) - g_values).abs()       # [G]
        rows_same = torch.gather(same, -2, loc.unsqueeze(-1).expand(*loc.shape, n_locations))
        cost = outside * n_locations + (index - loc.unsqueeze(-1)).abs()
        best = torch.where(rows_same, cost.to(q.dtype), torch.inf).argmin(-1)
        snapped.append(torch.where(s, best, loc))
    return snapped


def _centre_law(transfers, seen, locations, q, max_offset: int, tol: float):
    """One global shift ``s``: ``q'[j] = q[j + s]``, ``g' = g + s`` (seen rows), median of ``q'`` near 0."""
    n_offsets = q.shape[-1]
    offsets = torch.arange(-max_offset, max_offset + 1, device=q.device)
    support = q > tol
    lowest = torch.where(support, offsets, max_offset).amin(-1)            # [...]
    highest = torch.where(support, offsets, -max_offset).amax(-1)
    median = offsets[(q.cumsum(-1) < 0.5).sum(-1).clamp_max(n_offsets - 1)]
    shifts = torch.arange(-2 * max_offset, 2 * max_offset + 1, device=q.device)  # [S]
    feasible = ((lowest.unsqueeze(-1) - shifts >= -max_offset)
                & (highest.unsqueeze(-1) - shifts <= max_offset))
    for a, loc, s in zip(transfers, locations, seen):
        lo_loc = torch.where(s, loc, a.shape[0]).amin(-1)
        hi_loc = torch.where(s, loc, -1).amax(-1)
        feasible &= (lo_loc.unsqueeze(-1) + shifts >= 0) & (hi_loc.unsqueeze(-1) + shifts <= a.shape[0] - 1)
    distance = (median.unsqueeze(-1) - shifts).abs().to(q.dtype)
    shift = shifts[torch.where(feasible, distance, torch.inf).argmin(-1)]  # [...]
    index = torch.arange(n_offsets, device=q.device) + shift.unsqueeze(-1)
    inside = (index >= 0) & (index < n_offsets)
    q = torch.where(inside, torch.gather(q, -1, index.clamp(0, n_offsets - 1)), torch.zeros_like(q))
    q = q / q.sum(-1, keepdim=True)
    locations = [torch.where(s, loc + shift.unsqueeze(-1), loc) for loc, s in zip(locations, seen)]
    return locations, q


def unreachable_cells(transfer: torch.Tensor, counts: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
    """Cells ``[R, K]`` with data that no offset can produce from their row's location."""
    reachable = transfer[locations].sum(-1) > 0                          # [R, K]
    return (counts > 0) & ~reachable


def _best_of(transfers, counts, starts, n_iter, rtol, per_element: bool = False):
    """Run coordinate ascent from every start (a leading batch dimension) and keep the best.

    ``starts`` is ``[S, M]`` (same starts for every batch element) or, with
    ``per_element``, ``[S, ..., M]``.
    """
    n_starts = starts.shape[0]
    batch_shape = counts[0].shape[:-2]
    stacked = [c.unsqueeze(0).expand(n_starts, *c.shape) for c in counts]
    q0 = starts if per_element else starts.view(n_starts, *([1] * len(batch_shape)), -1).expand(
        n_starts, *batch_shape, -1
    )
    result = _coordinate_ascent(transfers, stacked, q0, n_iter, rtol)
    return _select(result, result[2].argmax(0))


def _select(result, winner: torch.Tensor):
    """Pick run ``winner[...]`` along the leading dimension of every tensor in ``result``."""
    def pick(t: torch.Tensor) -> torch.Tensor:
        index = winner.view(1, *winner.shape, *([1] * (t.dim() - 1 - winner.dim())))
        return torch.gather(t, 0, index.expand(1, *t.shape[1:])).squeeze(0)

    locations, q, objective = result
    return [pick(loc) for loc in locations], pick(q), pick(objective)


def _pick(first, second):
    """Element-wise better of two ``(locations, q, objective)`` results."""
    better = second[2] > first[2]
    choose = lambda a, b: torch.where(  # noqa: E731
        better.view(*better.shape, *([1] * (a.dim() - better.dim()))), b, a
    )
    locations = [choose(a, b) for a, b in zip(first[0], second[0])]
    return locations, choose(first[1], second[1]), torch.where(better, second[2], first[2])


def _coordinate_ascent(transfers, counts, q, n_iter: int, rtol: float, locations=None):
    """Alternate the best location per observed row and one EM step for ``q``.

    Neither step decreases the objective; stops when every batch element
    improves by less than ``rtol`` (relative).
    """
    seen = [c.sum(-1) > 0 for c in counts]
    if locations is None:
        locations = [
            torch.full(s.shape, a.shape[0] // 2, dtype=torch.long, device=s.device)
            for s, a in zip(seen, transfers)
        ]
    locations = list(locations)
    total = sum(c.sum(dim=(-2, -1)) for c in counts).clamp_min(1.0)
    previous = None
    for _ in range(n_iter):
        for i, (a, c, s) in enumerate(zip(transfers, counts, seen)):
            log_law = torch.log(child_law(a, q).clamp_min(_tiny(q)))         # [..., G, K]
            score = torch.einsum("...rk,...gk->...rg", c, log_law)          # [..., R, G]
            locations[i] = torch.where(s, score.argmax(-1), locations[i])
        q = _em_step(transfers, counts, locations, q)
        objective = _objective(transfers, counts, locations, q) / total
        if previous is not None and bool(
            ((objective - previous) <= rtol * objective.abs().clamp_min(1.0)).all()
        ):
            break
        previous = objective
    return locations, q, objective


def _objective(transfers, counts, locations, q) -> torch.Tensor:
    return sum(
        log_likelihood(c, rows_table(child_law(a, q), loc).clamp_min(_tiny(q)))
        for a, c, loc in zip(transfers, counts, locations)
    )


def _row_moves(transfers, counts, best, n_iter: int, rtol: float, max_rows: int):
    """Stage 3: try every location for the busiest rows, all candidates in one batch."""
    best_locations, best_q, _ = best
    batch_dims = counts[0].dim() - 2
    candidates = []  # (member, row, location index)
    for i, (a, c) in enumerate(zip(transfers, counts)):
        pooled = c.reshape(-1, *c.shape[-2:]).sum(0)                        # [R, K] over the batch
        row_mass = pooled.sum(-1)
        busiest = torch.argsort(row_mass, descending=True)[:max_rows]
        busiest = busiest[row_mass[busiest] > 0]
        # Only locations from which every value observed in the row is reachable.
        reachable = a.sum(-1) > 0                                           # [G, K]
        observed = pooled[busiest] > 0                                      # [B_rows, K]
        feasible = ~(observed.unsqueeze(1) & ~reachable.unsqueeze(0)).any(-1)  # [B_rows, G]
        candidates += [(i, int(r), int(g)) for r, row in zip(busiest, feasible)
                       for g in row.nonzero().flatten()]
    if not candidates:
        return None
    n_cand = len(candidates)
    member = torch.tensor([m for m, _, _ in candidates])
    rows = torch.tensor([r for _, r, _ in candidates], device=best_q.device)
    targets = torch.tensor([g for _, _, g in candidates], device=best_q.device)
    locations = []
    for i, loc in enumerate(best_locations):
        stacked = loc.unsqueeze(0).expand(n_cand, *loc.shape).clone()     # [C, ..., R]
        mine = (member == i).nonzero().squeeze(-1).to(best_q.device)
        if mine.numel():
            view = stacked.movedim(-1, 1)                                   # [C, R, ...]
            view[mine, rows[mine]] = targets[mine].view(-1, *([1] * batch_dims)).expand(
                -1, *loc.shape[:-1]
            )
        locations.append(stacked)
    stacked_counts = [c.unsqueeze(0).expand(n_cand, *c.shape) for c in counts]
    q = best_q.unsqueeze(0).expand(n_cand, *best_q.shape)
    for _ in range(50):  # q for the moved locations (concave given the locations)
        q = _em_step(transfers, stacked_counts, locations, q)
    result = _coordinate_ascent(transfers, stacked_counts, q, n_iter, rtol, locations=locations)
    return _select(result, result[2].argmax(0))


def _em_step(transfers, counts, locations, q: torch.Tensor) -> torch.Tensor:
    """One EM update of the shared offset law, aggregated by location."""
    expected = torch.zeros_like(q)
    for a, c, loc in zip(transfers, counts, locations):
        by_loc = counts_by_location(c, loc, a.shape[0])                   # [..., K, K]
        law = child_law(a, q)
        # Expected offset counts: N[g,k] * q_m * A[g,k,m] / P[g,k]; unreachable cells drop out.
        ratio = torch.where(law > 0, by_loc / law.clamp_min(_tiny(q)), torch.zeros_like(law))
        expected = expected + torch.einsum("...gk,gkm->...m", ratio, a) * q
    return expected / expected.sum(-1, keepdim=True).clamp_min(_tiny(q))


def _multi_starts(n_offsets: int, like: torch.Tensor) -> torch.Tensor:
    """``[M + 1, M]`` starting laws: a triangle peaked at 0 first, then one per offset."""
    max_offset = (n_offsets - 1) // 2
    concentrated = torch.full((n_offsets, n_offsets), 0.1 / max(n_offsets - 1, 1),
                              device=like.device, dtype=like.dtype)
    concentrated.fill_diagonal_(0.9 if n_offsets > 1 else 1.0)
    u = torch.arange(-max_offset, max_offset + 1, device=like.device, dtype=like.dtype)
    triangle = max_offset + 1 - u.abs()
    return torch.cat([(triangle / triangle.sum()).unsqueeze(0), concentrated])


def _shift_law(q: torch.Tensor, shift: int) -> torch.Tensor:
    """``q`` moved by ``shift`` offsets (mass pushed past the edge piles up there), smoothed."""
    moved = torch.zeros_like(q)
    n = q.shape[-1]
    target = (torch.arange(n, device=q.device) + shift).clamp(0, n - 1)
    moved.index_add_(-1, target, q)
    moved = moved + 1e-3 / n
    return moved / moved.sum(-1, keepdim=True)


def _tiny(t: torch.Tensor) -> float:
    return torch.finfo(t.dtype).tiny
