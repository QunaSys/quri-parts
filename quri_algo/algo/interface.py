from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
)

from quri_parts.algo.optimizer import OptimizerState
from quri_parts.backend.units import TimeValue
from quri_parts.circuit import ImmutableQuantumCircuit, NonParametricQuantumCircuit

from .utils import timer

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


Analyzer: TypeAlias = Callable[
    [NonParametricQuantumCircuit], AnalyzeResult
]  # Provides an interface for QURI VM


@dataclass
class AlgorithmResult(ABC):
    """Algorithm result."""

    algorithm: "Algorithm"
    elapsed_time: Optional[float]

    @property
    def name(self) -> str:
        """The name of the algorithm."""
        return self.algorithm.name


@dataclass
class VariationalAlgorithmResultMixin(ABC):
    optimizer_history: Sequence[OptimizerState]

    @property
    def optimizer_result(self) -> OptimizerState:
        return self.optimizer_history[-1]

    @property
    def cost_function_value(self) -> float:
        return self.optimizer_result.cost

    @property
    def cost_function_history(self) -> Sequence[float]:
        return [optimizer_state.cost for optimizer_state in self.optimizer_history]


T = TypeVar("T")


class CircuitMapping(Mapping[ImmutableQuantumCircuit, T]):
    """Map an immutable quantum circuit to a value."""

    _vals_dictionary: dict[int, T]
    _keys_dictionary: dict[int, ImmutableQuantumCircuit]

    def _circuit_hash(self, circuit: ImmutableQuantumCircuit) -> int:
        """Hash the circuit gates and qubit count."""
        return hash((circuit.qubit_count, circuit.gates))

    def __init__(
        self, circuit_vals: Sequence[tuple[ImmutableQuantumCircuit, T]]
    ) -> None:
        self._vals_dictionary = {}
        self._keys_dictionary = {}
        for circuit, val in circuit_vals:
            self._vals_dictionary[self._circuit_hash(circuit)] = val
            self._keys_dictionary[self._circuit_hash(circuit)] = circuit

    def __getitem__(self, circuit: ImmutableQuantumCircuit) -> T:
        return self._vals_dictionary.__getitem__(self._circuit_hash(circuit))

    def __setitem__(self, circuit: ImmutableQuantumCircuit, value: T) -> None:
        self._vals_dictionary.__setitem__(self._circuit_hash(circuit), value)
        self._keys_dictionary.__setitem__(self._circuit_hash(circuit), circuit)

    def __delitem__(self, circuit: ImmutableQuantumCircuit) -> None:
        self._vals_dictionary.__delitem__(self._circuit_hash(circuit))
        self._keys_dictionary.__delitem__(self._circuit_hash(circuit))

    def __iter__(self) -> Iterator[ImmutableQuantumCircuit]:
        return self._keys_dictionary.values().__iter__()

    def __len__(self) -> int:
        return len(self._vals_dictionary)

    def __contains__(self, key: object) -> bool:
        return True if key in self._keys_dictionary.values() else False


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
    """Base class for all algorithms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the algorithm."""
        pass

    def __init__(self) -> None:
        self._elapsed_time: Optional[float] = None

    @property
    def elapsed_time(self) -> Optional[float]:
        t = self._elapsed_time
        return t

    @elapsed_time.setter
    def elapsed_time(self, t: float) -> None:
        self._elapsed_time = t

    @elapsed_time.deleter
    def elapsed_time(self) -> None:
        self._elapsed_time = None

    @timer
    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the algorithm itself."""
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> AlgorithmResult:
        """Run the algorithm and return a result."""
        pass

    def __str__(self) -> str:
        """Basic information about the algorithm should be returned."""
        return self.name


class QuantumMixin(Protocol):
    @abstractmethod
    def analyze(self, *args: Any, **kwargs: Any) -> Analysis:
        """The quantum resource analysis of the algorithm."""
        pass


class QuantumAlgorithm(Algorithm, QuantumMixin, ABC):
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> QuantumAlgorithmResult:
        """Runs the algorithm and the algorithm analysis and return a
        result."""
        pass
