# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

from quri_parts.algo.optimizer import Optimizer, OptimizerState, Params
from quri_parts.backend.units import TimeUnit, TimeValue
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit

from quri_algo.algo.compiler.base_classes import (
    CompilationResult,
    QuantumCompilerGeneric,
)
from quri_algo.algo.interface import Analysis, Analyzer, CircuitMapping, timer
from quri_algo.circuit.interface import CircuitFactory
from quri_algo.core.cost_functions.base_classes import CostFunction
from quri_algo.core.cost_functions.utils import prepare_circuit_hilbert_schmidt_test


@dataclass
class QAQCAnalysis(Analysis):
    concurrency: int = 1

    @property
    def total_latency(self) -> TimeValue:
        """Total latency of the circuit."""
        latency_in_ns = 0.0
        for c, l in self.circuit_latency.items():
            if l is None:
                continue
            n = (self.circuit_execution_count[c] - 1) // self.concurrency + 1
            latency_in_ns += l.in_ns() * n
        return TimeValue(latency_in_ns, TimeUnit.NANOSECOND)

    @property
    def qubit_count(self) -> int:
        """Total number of physical qubits of the circuit."""
        qubit_counts = [c * self.concurrency for c in self.circuit_qubit_count.values()]
        return max(qubit_counts)


class QAQC(QuantumCompilerGeneric):
    """Quantum-Assisted Quantum Compilation (QAQC)

    This class provides an implementation of QAQC https://quantum-
    journal.org/papers/q-2019-05-13-140/. The optimize method returns a
    compiled circuit along with the converged state of the optimizer. It
    is also possible to use the __call__ method to return the compiled
    circuit only.
    """

    name = Literal["Quantum-Assisted Quantum Compilation (QAQC)"]

    def __init__(self, cost_fn: CostFunction, optimizer: Optimizer, analyzer: Optional[Analyzer] = None):
        super().__init__(cost_fn, optimizer, analyzer)

    @property
    def cost_function(self) -> CostFunction:
        return self._cost_fn

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def analyzer(self) -> Optional[Analyzer]:
        return self._analyzer

    @timer
    def run(
        self,
        circuit_factory: CircuitFactory,
        ansatz: LinearMappedUnboundParametricQuantumCircuit,
        init_params: Optional[Params] = None,
        *args: Any,
        **kwargs: Any,
    ) -> CompilationResult:
        """This function initiates the variational optimization.

        Arguments:
        circuit_factory - Circuit factory which generates the target circuit for the compilation
        ansatz - Parametric ansatz circuit that is used to approximate the target circuit
        init_params - initial variational parameters for the optimization

        variable arguments if any are passed to the circuit_factory's __call__ method

        Return value:
        Tuple of compiled circuit and the state of the optimizer
        """
        optimized_circuit, optimizer_history = self._optimize(
            circuit_factory, ansatz, init_params, *args, **kwargs
        )

        return CompilationResult(
            algorithm=self,
            elapsed_time=None,
            optimizer_history=optimizer_history,
            optimized_circuit=optimized_circuit,
        )

    def analyze(
        self,
        circuit_factory: CircuitFactory,
        ansatz: LinearMappedUnboundParametricQuantumCircuit,
        optimizer_history: Sequence[OptimizerState],
        circuit_execution_multiplier: Optional[int] = None,
        concurrency: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> Analysis:
        """This function analyses the quantum resources required to run the
        quantum compilation.

        Arguments:
        circuit_factory - Circuit factory which generates the target circuit for the compilation
        ansatz - Parametric ansatz circuit that is used to approximate the target circuit
        optimizer_state - state of the optimizer after the optimization
        circuit_execution_multiplier - multiplier for the number of circuit executions, e.g. number of samples for each estimate

        variable arguments if any are passed to the circuit_factory's __call__ method

        Return value:
        Tuple of compiled circuit and the state of the optimizer
        """
        assert self.analyzer, "Needs an analyzer to analyze the circuit"

        circuit_list = []
        for optimizer_state in optimizer_history:
            target_circuit = circuit_factory(*args, **kwargs)
            combined_circuit = prepare_circuit_hilbert_schmidt_test(
                target_circuit,
                ansatz.bind_parameters(list(optimizer_state.params.tolist())),
            ).circuit
            circuit_list.append(combined_circuit)

        total_circuit_executions = (
            ansatz.parameter_count * 2 * optimizer_state.gradcalls
            + optimizer_state.funcalls
        )
        if circuit_execution_multiplier is not None:
            total_circuit_executions *= circuit_execution_multiplier
        circuit_analysis = self.analyzer(
            combined_circuit
        )  # QAQC analysis can be assumed to result identically for each circuit
        analysis = QAQCAnalysis(
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
            # The analysis is the same for all circuits
            concurrency=concurrency,
        )

        return analysis
