from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, TypeVar, Generic, runtime_checkable

from quri_parts.backend.units import TimeUnit, TimeValue
from quri_parts.circuit import ImmutableQuantumCircuit
from quri_parts.core.estimator import Estimatable, Estimate
from quri_parts.core.sampling import MeasurementCounts
from quri_parts.core.state import CircuitQuantumState
from quri_parts.circuit import NonParametricQuantumCircuit
from quri_vm.vm import VM
from sympy import Expr

class AnalyzeResult(Protocol):
    """Analyze result protocol for consistency with the VM interface."""
    qubit_count: int
    gate_count: int
    depth: int
    latency: TimeValue | None = None
    fidelity: float | None = None


class VM(Protocol):
    """VM interface protocol."""
    @abstractmethod
    def estimate(
        self,
        estimatable: Estimatable,
        state: CircuitQuantumState,
    ) -> Estimate[complex]:
        pass
    
    @abstractmethod
    def sample(
        self,
        circuit: NonParametricQuantumCircuit,
        shots: int,
    ) -> MeasurementCounts:
        pass
    
    @abstractmethod
    def transpile(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        pass

    @abstractmethod
    def analyze(self, circuit: NonParametricQuantumCircuit) -> AnalyzeResult:
        pass


T = TypeVar("T")

class EvaluatorHooks(Generic[T], ABC):
    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> T:
        """Evaluate the algorithm."""
        pass

class EvaluatorHooksAnalysis(EvaluatorHooks[AnalyzeResult]):
    """Evaluator hooks for the analysis of the algorithm."""
    
    def analyze(self, vm: VM, *args: Any, **kwargs: Any) -> AnalyzeResult:
        """Analyze the algorithm."""
        return vm.analyze(*args, **kwargs)

@dataclass
class AlgorithmResult(ABC):
    """Algorithm result."""

    algorithm: "Algorithm"
    runtime: float

    @property
    def name(self) -> str:
        """The name of the algorithm."""
        return self.algorithm.name



@dataclass
class Analysis(ABC):
    """Analysis of the algorithm."""
    circuit_latency: Mapping[ImmutableQuantumCircuit, TimeValue | None]
    circuit_execution_count: Mapping[ImmutableQuantumCircuit, int]
    circuit_fidelities: Mapping[ImmutableQuantumCircuit, float | None]
    circuit_footprint: Mapping[ImmutableQuantumCircuit, int]

    @property
    def total_latency_sequential(self) -> TimeValue:
        """Total latency of the circuit, assuming it is not parallelizable."""
        latency_in_ns = 0.0
        for c, l in self.circuit_latency.items():
            if l is None:
                continue
            n = self.circuit_execution_count[c]
            latency_in_ns += l.in_ns() * n
        return TimeValue(latency_in_ns, TimeUnit.NANOSECOND)

    @property
    @abstractmethod
    def total_latency_parallel(self) -> TimeValue:
        """Total latency of the circuit, assuming parallelizable execution."""
        latency_in_ns = 0.0
        for c, l in self.circuit_latency.items():
            if l is None:
                continue
            n = self.circuit_execution_count[c]
            latency_in_ns = max(latency_in_ns, l.in_ns() * n)
        return TimeValue(latency_in_ns, TimeUnit.NANOSECOND)

    @property
    def total_footprint_sequential(self) -> int:
        """Total footprint of the circuit, assuming it is not
        parallelizable."""
        return max(self.circuit_footprint.values())

    @property
    def total_footprint_parallel(self) -> int:
        """Total footprint of the circuit, assuming parallelizable
        execution."""
        return sum(self.circuit_footprint.values())


@dataclass
class QuantumAlgorithmResult(AlgorithmResult, ABC):
    """Algorithm result with a resource analysis."""

    analysis: Analysis


class Algorithm(ABC):
    name: str
    """The name of the algorithm."""

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the algorithm itself."""
        pass

    @property
    @abstractmethod
    def run_time_scaling(self, *args: Any, **kwargs: Any) -> Expr:
        """The run-time scaling given in big-O notation as a sympy
        expression."""
        pass

    @abstractmethod
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> AlgorithmResult | QuantumAlgorithmResult:
        """Run the algorithm and return a result."""
        pass

    def __str__(self) -> str:
        """Basic information about the algorithm should be returned."""
        return self.name + " with run-time scaling " + str(self.run_time_scaling)


class QuantumMixin(Protocol):
    @abstractmethod
    def analyze(self, *args: Any, **kwargs: Any) -> Analysis:
        """The resource analysis of the algorithm."""
        pass


class QuantumAlgorithm(Algorithm, QuantumMixin, ABC):
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> QuantumAlgorithmResult:
        """Runs the algorithm and the algorithm analysis and return a
        result."""
        pass
