"""NeuralBayesianNetworks (NBN) — PyTorch-native Bayesian Networks with neural mechanisms."""

from nbn.core.network import NeuralBayesianNetwork
from nbn.core.dag import DAG
from nbn.core.variables import Variable, DiscreteVariable, ContinuousVariable
from nbn.core.query import NBNQuery
from nbn.mechanisms import (
    Mechanism,
    CategoricalTableMechanism,
    LinearGaussianMechanism,
    MDNMechanism,
    NeuralCategoricalMechanism,
    DeterministicMechanism,
)
from nbn.inference import (
    InferenceEngine,
    LikelihoodWeightingEngine,
    TensorVariableElimination,
    HybridRouter,
)
from nbn.learning import fit, TrainHistory
from nbn.sampling import ancestral_sample
from nbn.utils.seed import seed_all

try:
    from nbn._version import version as __version__
except ImportError:
    __version__ = "0.0.0+dev"

__all__ = [
    # Core
    "NeuralBayesianNetwork",
    "DAG",
    "Variable", "DiscreteVariable", "ContinuousVariable",
    "NBNQuery",
    # Mechanisms
    "Mechanism",
    "CategoricalTableMechanism",
    "LinearGaussianMechanism",
    "MDNMechanism",
    "NeuralCategoricalMechanism",
    "DeterministicMechanism",
    # Inference
    "InferenceEngine",
    "LikelihoodWeightingEngine",
    "TensorVariableElimination",
    "HybridRouter",
    # Learning
    "fit",
    "TrainHistory",
    # Sampling
    "ancestral_sample",
    # Utils
    "seed_all",
    # Version
    "__version__",
]
