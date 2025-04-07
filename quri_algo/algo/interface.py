from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence, Tuple, TypeAlias, TypeVar, runtime_checkable

from quri_parts.backend.units import TimeValue
from quri_parts.circuit import ImmutableQuantumCircuit, NonParametricQuantumCircuit
from quri_parts.core.estimator import Estimatable, Estimate
from quri_parts.core.sampling import MeasurementCounts
from quri_parts.core.state import CircuitQuantumState
from sympy import Expr

LoweringLevel = Enum(
    "LoweringLevel",
    [
        ("LogicalCircuit", 0),
        ("ArchLogicalCircuit", 1),
        ("ArchInstruction", 2),
        ("DeviceInstruction", 3),
    ],
)


class AnalyzeResult(Protocol):
    """Analyze result protocol for consistency with the VM interface."""

    lowering_level: LoweringLevel
    qubit_count: int
    gate_count: int
    depth: int
    latency: TimeValue | None = None
    fidelity: float | None = None


@runtime_checkable
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


@dataclass
class AlgorithmResult(ABC):
    """Algorithm result."""

    algorithm: "Algorithm"
    runtime: float

    @property
    def name(self) -> str:
        """The name of the algorithm."""
        return self.algorithm.name


T = TypeVar("T")
CircuitMapping: TypeAlias = Sequence[
    Tuple[ImmutableQuantumCircuit, T]
]  # Would be better to use a dict, but circuits aren't hashable


@dataclass
class Analysis(ABC):
    """Analysis of the algorithm."""

    lowering_level: LoweringLevel
    circuit_gate_count: CircuitMapping[int]
    circuit_depth: CircuitMapping[int]
    circuit_latency: CircuitMapping[TimeValue | None]
    circuit_execution_count: CircuitMapping[int]
    circuit_fidelities: CircuitMapping[float | None]
    circuit_qubit_count: CircuitMapping[int]

    @property
    @abstractmethod
    def total_latency(self, *args: Any, **kwargs: Any) -> TimeValue:
        pass

    @property
    @abstractmethod
    def qubit_count(self, *args: Any, **kwargs: Any) -> int:
        pass


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
