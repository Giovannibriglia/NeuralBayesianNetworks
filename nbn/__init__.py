"""NeuralBayesianNetworks (NBN) — PyTorch-native Bayesian Networks with neural mechanisms."""

from nbn.core.dag import DAG
from nbn.core.network import NeuralBayesianNetwork
from nbn.core.query import NBNQuery
from nbn.core.variables import ContinuousVariable, DiscreteVariable, Variable
from nbn.inference import (
    HybridRouter,
    InferenceEngine,
    LikelihoodWeightingEngine,
    TensorVariableElimination,
)
from nbn.learning import TrainHistory, fit
from nbn.mechanisms import (
    CategoricalTableMechanism,
    DeterministicMechanism,
    LinearGaussianMechanism,
    MDNMechanism,
    Mechanism,
    NeuralCategoricalMechanism,
)
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


# --- non-parametric mechanisms (added by apply_nonparametric_mechanisms) ---
from nbn.mechanisms import (  # noqa: E402
    ConditionalKDEMechanism,
    KNNConditionalMechanism,
    SmoothedEmpiricalCategoricalMechanism,
    FlexCodeMechanism,
)

if "__all__" not in globals():
    __all__ = []
__all__ += [
    "ConditionalKDEMechanism",
    "KNNConditionalMechanism",
    "SmoothedEmpiricalCategoricalMechanism",
    "FlexCodeMechanism",
]
