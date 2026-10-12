"""Regression tests for #304 (0-d DeterministicMechanism value) and #305
(training gradients left on parameters after fit / update)."""
import pytest
import torch

from nbn import (
    CategoricalTableMechanism,
    DeterministicMechanism,
    LinearGaussianMechanism,
    MDNMechanism,
    NeuralBayesianNetwork,
)


# --- #304 -----------------------------------------------------------------------------
@pytest.mark.parametrize("value", [torch.tensor(1.0), 1.0, 1])
def test_scalar_value_is_read_as_one_dimensional(value):
    m = NeuralBayesianNetwork([("A", "B")], variables={"A": ("discrete", 2), "B": ("discrete", 2)},
                              device="cpu")
    m.set_mechanism("A", CategoricalTableMechanism())
    m.set_mechanism("B", DeterministicMechanism(value, cardinality=2))
    m.fit({"A": torch.randint(0, 2, (100,)), "B": torch.ones(100, dtype=torch.long)})
    samples = m.sample(5)
    assert torch.all(samples["B"].reshape(-1) == 1)


def test_bad_value_shapes_are_rejected():
    with pytest.raises(ValueError, match=r"\[D_x\] or \[B, D_x\]"):
        DeterministicMechanism(torch.zeros(2, 3, 4))
    with pytest.raises(ValueError, match="one value per row"):
        DeterministicMechanism(torch.tensor([0.0, 1.0]), cardinality=2)


def test_valid_shapes_unchanged():
    assert DeterministicMechanism(torch.tensor([0.5, 2.0]))._fixed_value.shape == (2,)
    assert DeterministicMechanism(torch.zeros(3, 2))._fixed_value.shape == (3, 2)
    assert DeterministicMechanism(torch.tensor([1.0]), cardinality=2)._fixed_value.shape == (1,)


# --- #305 -----------------------------------------------------------------------------
def _hybrid_model():
    torch.manual_seed(0)
    n = NeuralBayesianNetwork([("X", "Y")], variables={"X": ("continuous", 1), "Y": ("continuous", 1)},
                              device="cpu")
    n.set_mechanism("X", LinearGaussianMechanism())
    n.set_mechanism("Y", MDNMechanism(num_components=2))
    return n


def _data(n=400):
    x = torch.randn(n, 1)
    return {"X": x, "Y": 2 * x + 0.1 * torch.randn(n, 1)}


def _grads_left(model):
    return [name for name, p in model.named_parameters() if p.grad is not None]


@pytest.mark.parametrize("method", ["local", "joint"])
def test_fit_releases_training_gradients(method):
    model = _hybrid_model()
    model.fit(_data(), epochs=5)
    if method == "joint":
        # The library warns that a later update() re-solves the linear-Gaussian
        # node from its local statistics; expected here.
        with pytest.warns(UserWarning, match="re-solves"):
            model.fit(_data(), epochs=2, method="joint")
    assert _grads_left(model) == []


def test_update_releases_gradients():
    model = _hybrid_model()
    model.fit(_data(), epochs=5)
    model.update(_data(200))
    assert _grads_left(model) == []


def test_backward_after_fit_starts_from_clean_gradients():
    model = _hybrid_model()
    model.fit(_data(), epochs=5)
    batch = _data(50)
    model.log_prob(batch).sum().backward()
    first = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
    for p in model.parameters():
        p.grad = None
    model.log_prob(batch).sum().backward()
    for k, p in model.named_parameters():
        if k in first:
            torch.testing.assert_close(p.grad, first[k])
