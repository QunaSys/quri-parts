from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Protocol, Mapping, Any
from quri_parts.backend.units import TimeValue
from quri_parts.circuit import ImmutableQuantumCircuit
from sympy import Expr


@dataclass
class AlgorithmResult:
    """Algorithm result"""

    algorithm: "Algorithm"
    runtime: float
    
    @property
    def name(self) -> str:
        """The name of the algorithm"""
        return self.algorithm.name


@dataclass
class Analysis:
    latency: TimeValue
    circuit_execution_count: int
    circuit_fidelities: Mapping[ImmutableQuantumCircuit, float]
    circuit_footprint: int


@dataclass
class QuantumAlgorithmResult(AlgorithmResult):
    """Algorithm result with a resource analysis"""

    analysis: Analysis


class Algorithm(ABC):
    name: str
    """The name of the algorithm"""

    @abstractmethod
    def run(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[
        str, Any
    ]:  # instead of Any: Estimate | HamiltonianEigensystem | OptimizerState | QuantumCircuit
        """The estimate returned by the algorithm"""
        pass

    @property
    @abstractmethod
    def run_time_scaling(self, *args: Any, **kwargs: Any) -> Expr:
        """The run-time scaling given in big-O notation as a sympy expression"""
        pass

    @abstractmethod
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> AlgorithmResult | QuantumAlgorithmResult:
        """The call method which runs the algorithm and returns a result"""
        pass

    def __str__(self) -> str:
        """Basic information about the algorithm should be returned"""
        return self.name + " with run-time scaling " + str(self.run_time_scaling)


class QuantumMixin(Protocol):
    @abstractmethod
    def analyze(self, *args: Any, **kwargs: Any) -> Analysis:
        """The resource analysis of the algorithm"""
        pass


class QuantumAlgorithm(Algorithm, QuantumMixin):
    def __call__(self, *args: Any, **kwargs: Any) -> QuantumAlgorithmResult:
        return QuantumAlgorithmResult(
            self.run(*args, **kwargs), self.analyze(*args, **kwargs)
        )


