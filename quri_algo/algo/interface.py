from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Protocol, Mapping, Any
from quri_parts.backend.units import TimeValue
from quri_parts.core.estimator import Estimate
from quri_parts.circuit import ImmutableQuantumCircuit
from sympy import Expr


@dataclass
class Analysis:
    latency: TimeValue
    circuit_execution_count: int
    circuit_fidelities: Mapping[ImmutableQuantumCircuit, float]


@dataclass
class AlgorithmResult:
    """Algorithm result with a resource analysis"""
    estimate: Estimate[Any]
    analysis: Analysis


class Algorithm(ABC):
    name: str
    """The name of the algorithm"""

    @property
    @abstractmethod
    def run_time_scaling(self, *args: Any, **kwargs: Any) -> Expr:
        """The run-time scaling given in big-O notation as a sympy expression"""
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> AlgorithmResult:
        """The call method which runs the algorithm and returns a result"""
        pass

    def __str__(self) -> str:
        """Basic information about the algorithm should be returned"""
        return self.name + " with run-time scaling " + str(self.run_time_scaling)


class QuantumMixin(Protocol):
    @abstractmethod
    def estimate(self, *args: Any, **kwargs: Any) -> Estimate[Any]:
        pass

    @abstractmethod
    def analyze(self, *args: Any, **kwargs: Any) -> Analysis:
        pass

class QuantumAlgorithm(Algorithm, QuantumMixin):
     def __call__(self, *args: Any, **kwargs: Any) -> AlgorithmResult:
        return AlgorithmResult(self.estimate(*args, **kwargs), self.analyze(*args, **kwargs))
    