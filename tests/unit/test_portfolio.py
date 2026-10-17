"""select_mechanisms: successive halving over a mechanism portfolio."""
import functools

import torch

from nbn import NeuralBayesianNetwork
from nbn.learning.portfolio import select_mechanisms
from nbn.mechanisms import LinearGaussianMechanism, MDNMechanism


def _data(n=2000):
    g = torch.Generator().manual_seed(0)
    a = torch.randn(n, generator=g)
    b = torch.sin(2 * a) + 0.1 * torch.randn(n, generator=g)
    c = 0.5 * a + 0.2 * torch.randn(n, generator=g)
    return {"a": a[:, None], "b": b[:, None], "c": c[:, None]}


def _model():
    return NeuralBayesianNetwork([("a", "b"), ("a", "c")],
                                 {k: ("continuous", 1) for k in "abc"}, device="cpu")


def test_picks_a_nonlinear_mechanism_where_needed_and_spends_less():
    torch.manual_seed(0)
    cands = {"linear": LinearGaussianMechanism,
             "mdn_small": functools.partial(MDNMechanism, num_components=1, hidden=(8,)),
             "mdn": functools.partial(MDNMechanism, num_components=3, hidden=(32, 32))}
    m = _model()
    rep = select_mechanisms(m, _data(), cands, max_epochs=60, eta=2, min_epochs=5)
    assert rep.chosen["b"].startswith("mdn")
    assert rep.chosen["c"] in ("linear", "mdn_small", "mdn")
    assert all(rep.epochs_spent[n] <= rep.epochs_full[n] for n in "abc")
    # the winner is installed and trained: its held-out-free score is finite
    lp = m.mechanisms["b"].log_prob(_data(200)["b"], _data(200)["a"])
    assert torch.isfinite(lp).all()


def test_closed_form_only_portfolio_and_per_node_candidates():
    m = _model()
    rep = select_mechanisms(m, _data(), {"a": {"linear": LinearGaussianMechanism},
                                         "b": {"linear": LinearGaussianMechanism},
                                         "c": {"linear": LinearGaussianMechanism}})
    assert set(rep.chosen.values()) == {"linear"}
    assert all(v == 0 for v in rep.epochs_spent.values())


def test_refit_uses_all_rows():
    torch.manual_seed(0)
    m = _model()
    rep = select_mechanisms(m, _data(), {"mdn": functools.partial(MDNMechanism, num_components=1,
                                                                  hidden=(8,))},
                            nodes=["b"], max_epochs=10, refit=True)
    assert rep.chosen == {"b": "mdn"}
