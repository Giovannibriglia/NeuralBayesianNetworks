from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Union


@dataclass
class Variable:
    """Describes the type and dimensionality of a DAG node's value.

    Parameters
    ----------
    name: str
        Node name (matches the DAG node label).
    kind: "discrete" | "continuous" | "mixed"
        Variable type.
    dim: int
        Output dimensionality (1 for univariate).
    cardinality: int | None
        Number of classes for discrete variables; ``None`` for continuous.
    """

    name: str
    kind: Literal["discrete", "continuous", "mixed"]
    dim: int = 1
    cardinality: int | None = None

    @classmethod
    def from_spec(
        cls,
        name: str,
        spec: Union[Tuple[Literal["discrete", "continuous", "mixed"], int], Variable],
    ) -> Variable:
        if isinstance(spec, Variable):
            return spec
        kind, dim = spec
        cardinality = dim if kind == "discrete" else None
        return cls(name=name, kind=kind, dim=dim, cardinality=cardinality)

    @property
    def is_discrete(self) -> bool:
        return self.kind == "discrete"

    @property
    def is_continuous(self) -> bool:
        return self.kind == "continuous"


@dataclass
class DiscreteVariable(Variable):
    """Convenience subclass for discrete variables."""

    def __init__(self, name: str, cardinality: int) -> None:
        super().__init__(name=name, kind="discrete", dim=1, cardinality=cardinality)


@dataclass
class ContinuousVariable(Variable):
    """Convenience subclass for continuous variables."""

    def __init__(self, name: str, dim: int = 1) -> None:
        super().__init__(name=name, kind="continuous", dim=dim, cardinality=None)
