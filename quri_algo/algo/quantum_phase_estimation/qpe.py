from typing import Any, Collection, Final, Mapping, Optional, Sequence

from quri_parts.backend.units import TimeUnit, TimeValue
from quri_parts.circuit import ImmutableQuantumCircuit, QuantumCircuit
from quri_parts.core.sampling import MeasurementCounts, Sampler
from quri_parts.core.sampling.default_sampler import DEFAULT_SAMPLER
from quri_parts.core.state import CircuitQuantumState
from quri_parts.qsub.compile import compile_sub
from quri_parts.qsub.eval import QURIPartsEvaluatorHooks
from quri_parts.qsub.evaluate import Evaluator
from quri_parts.qsub.lib import std
from quri_parts.qsub.machineinst import MachineSub
from quri_parts.qsub.op import Op
from quri_parts.qsub.resolve import resolve_sub

from quri_algo.algo.interface import (
    AlgorithmResult,
    Analysis,
    Analyzer,
    AnalyzeResult,
    EnergyEstimationResultMixin,
    LoweringLevel,
    QuantumAlgorithm,
)
from quri_algo.algo.utils.mappings import CircuitMapping
from quri_algo.algo.utils.timer import timer
from quri_algo.problem import QubitHamiltonian
from quri_algo.qsub.time_evolution.interface import TimeEvolutionOpFactory
from quri_algo.qsub.utils import QPEListUk


def get_qpe_msub(
    ancilla_count: int,
    hamiltonian: QubitHamiltonian,
    TimeEvoOpFactory: TimeEvolutionOpFactory,
    evolution_time: float,
    *args: Any,
    primitives: Sequence[Op] = (std.X, std.T, std.Tdag, std.CNOT, std.Toffoli)
) -> MachineSub:
    time_evolution_ops = [
        TimeEvoOpFactory(hamiltonian, 2**k * evolution_time, *args)
        for k in range(ancilla_count)
    ]
    qpe_sub = resolve_sub(QPEListUk(ancilla_count, time_evolution_ops))
    assert qpe_sub
    qpe_msub = compile_sub(qpe_sub, primitives)
    return qpe_msub


class QPEResult(AlgorithmResult, EnergyEstimationResultMixin):
    """QPE result."""

    def __init__(
        self,
        algorithm: "QPE",
        eigen_values: Collection[float],
        elapsed_time: Optional[float] = None,
    ):
        AlgorithmResult.__init__(self, algorithm, elapsed_time)
        EnergyEstimationResultMixin.__init__(self, eigen_values)


class QPEAnalysis(Analysis):
    """QPE resource analysis"""

    def __init__(
        self,
        lowering_level: LoweringLevel,
        circuit_gate_count: Mapping[ImmutableQuantumCircuit, int],
        circuit_depth: Mapping[ImmutableQuantumCircuit, int],
        circuit_latency: Mapping[ImmutableQuantumCircuit, TimeValue | None],
        circuit_execution_count: Mapping[ImmutableQuantumCircuit, int],
        circuit_fidelities: Mapping[ImmutableQuantumCircuit, float | None],
        circuit_qubit_count: Mapping[ImmutableQuantumCircuit, int],
    ) -> None:
        super().__init__(
            lowering_level,
            circuit_gate_count,
            circuit_depth,
            circuit_latency,
            circuit_execution_count,
            circuit_fidelities,
            circuit_qubit_count,
        )

    @property
    def total_latency(self) -> TimeValue:
        """Total latency of the circuit."""
        assert len(self.circuit_latency) == 1
        latency = list(self.circuit_latency.values())[0]
        assert latency
        assert len(self.circuit_execution_count) == 1
        latency_in_ns = latency.in_ns() * list(self.circuit_execution_count.values())[0]
        return TimeValue(latency_in_ns, TimeUnit.NANOSECOND)

    @property
    def max_physical_qubit_count(self) -> int:
        """Total number of physical qubits of the circuit."""
        assert len(self.circuit_qubit_count) == 1
        return list(self.circuit_qubit_count.values())[0]

    @classmethod
    def from_circuit_list(
        cls,
        circuit_analysis: AnalyzeResult,
        total_circuit_executions: int,
        circuit_list: Sequence[ImmutableQuantumCircuit],
    ) -> "QPEAnalysis":
        return cls(
            lowering_level=circuit_analysis.lowering_level,
            circuit_depth=CircuitMapping(
                [(c, circuit_analysis.depth) for c in circuit_list]
            ),
            circuit_gate_count=CircuitMapping(
                [(c, circuit_analysis.gate_count) for c in circuit_list]
            ),
            circuit_latency=CircuitMapping(
                [(c, circuit_analysis.latency) for c in circuit_list]
            ),
            circuit_execution_count=CircuitMapping(
                [(c, total_circuit_executions) for c in circuit_list]
            ),
            circuit_fidelities=CircuitMapping(
                [(c, circuit_analysis.fidelity) for c in circuit_list]
            ),
            circuit_qubit_count=CircuitMapping(
                [(c, circuit_analysis.qubit_count) for c in circuit_list]
            ),
        )


class QPE(QuantumAlgorithm):
    """Quantum Phase Estimation (QPE)

    This class provides an implementation of QPE.
    """

    _name: Final[str] = "Quantum Phase Estimation (QPE)"

    @property
    def name(self) -> str:
        return self._name

    def __init__(
        self,
        time_evolution_op_factory: TimeEvolutionOpFactory,
        sampler: Sampler = DEFAULT_SAMPLER,
    ):
        self._time_evolution_op_factory = time_evolution_op_factory
        self._qp_generator = Evaluator(QURIPartsEvaluatorHooks())
        self._sampler = sampler

    @property
    def time_evolution_op_factory(self) -> TimeEvolutionOpFactory:
        return self._time_evolution_op_factory

    @property
    def qp_generator(self) -> Evaluator[QuantumCircuit]:
        return self._qp_generator

    def sample(
        self, circuit: ImmutableQuantumCircuit, n_shots: int
    ) -> MeasurementCounts:
        return self._sampler(circuit, n_shots)

    @timer
    def run(
        self,
        trial_State: CircuitQuantumState,
        n_shots: int,
        ancilla_count: int,
        hamiltonian: QubitHamiltonian,
        evolution_time: float,
        *args: Any,
        primitives: Sequence[Op] = (std.X, std.T, std.Tdag, std.CNOT, std.Toffoli)
    ) -> QPEResult:
        qpe_msub = get_qpe_msub(
            ancilla_count,
            hamiltonian,
            self.time_evolution_op_factory,
            evolution_time,
            *args,
            primitives=primitives
        )

        qp_circuit = self.qp_generator.run(qpe_msub)
        total_circuit = trial_State.circuit + qp_circuit
        counts = dict(self.sample(total_circuit, n_shots))
        eigen_values = {
            b / (evolution_time * 2**ancilla_count) for b, _ in counts.items()
        }

        return QPEResult(self, eigen_values)

    def analyze(
        self,
        trial_State: CircuitQuantumState,
        n_shots: int,
        ancilla_count: int,
        analyzer: Analyzer,
        hamiltonian: QubitHamiltonian,
        evolution_time: float,
        *args: Any,
        primitives: Sequence[Op] = (std.X, std.T, std.Tdag, std.CNOT, std.Toffoli)
    ) -> QPEAnalysis:
        qpe_msub = get_qpe_msub(
            ancilla_count,
            hamiltonian,
            self.time_evolution_op_factory,
            evolution_time,
            *args,
            primitives=primitives
        )

        qp_circuit: ImmutableQuantumCircuit = self.qp_generator.run(qpe_msub)
        total_circuit = trial_State.circuit + qp_circuit
        analysis = analyzer(total_circuit)

        return QPEAnalysis.from_circuit_list(analysis, n_shots, [qp_circuit])
