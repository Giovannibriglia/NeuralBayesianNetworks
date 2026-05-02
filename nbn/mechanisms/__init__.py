from nbn.mechanisms.base import Mechanism
from nbn.mechanisms.categorical_table import CategoricalTableMechanism
from nbn.mechanisms.linear_gaussian import LinearGaussianMechanism
from nbn.mechanisms.mdn import MDNMechanism
from nbn.mechanisms.neural_categorical import NeuralCategoricalMechanism
from nbn.mechanisms.deterministic import DeterministicMechanism

__all__ = [
    "Mechanism",
    "CategoricalTableMechanism",
    "LinearGaussianMechanism",
    "MDNMechanism",
    "NeuralCategoricalMechanism",
    "DeterministicMechanism",
]

# Optional: normalizing flow mechanism (requires zuko)
try:
    from nbn.mechanisms.normalizing_flow import NormalizingFlowMechanism, ConditionalFlowMechanism
    __all__ += ["NormalizingFlowMechanism", "ConditionalFlowMechanism"]
except ImportError:
    pass
