"""Probability-set containers: simplex tightening, L1 boxes, sample boxes."""
from __future__ import annotations

import pytest
import torch
from torch.distributions import Dirichlet

from nbn.uncertainty import IntervalSet, L1BallSet, SampleSet, tighten_box_to_simplex


def test_tightening_keeps_every_distribution_of_the_box():
    torch.manual_seed(0)
    lower = torch.tensor([[0.1, 0.2, 0.0]])
    upper = torch.tensor([[0.5, 0.9, 0.3]])
    new_lower, new_upper = tighten_box_to_simplex(lower, upper)
    draws = Dirichlet(torch.ones(3)).sample((200_000,))
    in_box = ((draws >= lower) & (draws <= upper)).all(-1)
    kept = draws[in_box]
    assert bool(((kept >= new_lower - 1e-7) & (kept <= new_upper + 1e-7)).all())
    # ... and it actually tightens: p_1 >= 1 - 0.5 - 0.3 = 0.2 (unchanged), p_0 <= 1 - 0.2 - 0 = 0.8.
    assert float(new_upper[0, 1]) == pytest.approx(0.9)
    assert float(new_lower[0, 1]) == pytest.approx(0.2)


def test_interval_set_contains_and_width():
    s = IntervalSet(level=0.95, guarantee="pac", unseen=torch.tensor([False]),
                    lower=torch.tensor([[0.2, 0.3]]), upper=torch.tensor([[0.7, 0.8]]))
    assert s.contains(torch.tensor([[0.5, 0.5]]))
    assert not s.contains(torch.tensor([[0.1, 0.9]]))
    assert torch.allclose(s.width(), torch.tensor([[0.5, 0.5]]))


def test_l1_ball_box_moves_each_entry_by_half_the_radius():
    s = L1BallSet(level=0.95, guarantee="pac", unseen=torch.tensor([False]),
                  center=torch.tensor([[0.5, 0.3, 0.2]]), radius=torch.tensor([0.2]))
    lower, upper = s.bounds()
    assert torch.allclose(lower, torch.tensor([[0.4, 0.2, 0.1]]))
    assert torch.allclose(upper, torch.tensor([[0.6, 0.4, 0.3]]))
    assert s.contains(torch.tensor([[0.6, 0.2, 0.2]]))       # L1 distance 0.2
    assert not s.contains(torch.tensor([[0.7, 0.1, 0.2]]))   # L1 distance 0.4


def test_sample_set_box_is_simultaneous_when_asked():
    torch.manual_seed(0)
    samples = torch.rand(4000, 3, 2)
    joint = SampleSet(level=0.9, guarantee="bayesian", unseen=torch.zeros(3, dtype=torch.bool),
                      samples=samples, simultaneous=True)
    marginal = SampleSet(level=0.9, guarantee="bayesian", unseen=torch.zeros(3, dtype=torch.bool),
                         samples=samples, simultaneous=False)
    assert bool((joint.width() >= marginal.width()).all())
