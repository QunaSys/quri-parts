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
from enum import Enum
from typing import Any, Optional

from quri_parts.algo.optimizer import Optimizer, OptimizerState, Params
from quri_parts.backend.units import TimeUnit, TimeValue
from quri_parts.circuit import (
    LinearMappedUnboundParametricQuantumCircuit,
    NonParametricQuantumCircuit,
)
from sympy import Expr

from quri_algo.algo.compiler.base_classes import QuantumCompilerGeneric
from quri_algo.algo.interface import VM, Analysis
from quri_algo.circuit.interface import CircuitFactory
from quri_algo.core.cost_functions.base_classes import CostFunction
from quri_algo.core.cost_functions.utils import prepare_circuit_hilbert_schmidt_test

ExecutionMode = Enum("ExecutionMode", [("Sequential", 0), ("Parallel", 1)])


@dataclass
class QAQCAnalysis(Analysis):
    execution_mode: ExecutionMode = ExecutionMode.Sequential

    def total_latency_sequential(self) -> TimeValue:
        """Total latency of the circuit, assuming it is not parallelizable."""
        latency_in_ns = 0.0
        for c, l in self.circuit_latency.items():
            if l is None:
                continue
            n = self.circuit_execution_count[c]
            latency_in_ns += l.in_ns() * n
        return TimeValue(latency_in_ns, TimeUnit.NANOSECOND)

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
    def total_latency(self) -> TimeValue:
        if self.execution_mode == ExecutionMode.Sequential:
            return self.total_latency_sequential()
        elif self.execution_mode == ExecutionMode.Parallel:
            return self.total_latency_parallel()
        else:
            raise TypeError("Execution mode undefined")

    def qubit_count_sequential(self) -> int:
        """Total footprint of the circuit, assuming it is not
        parallelizable."""
        return max(self.circuit_qubit_count.values())

    def qubit_count_parallel(self) -> int:
        """Total footprint of the circuit, assuming parallelizable
        execution."""
        return sum(self.circuit_qubit_count.values())

    @property
    def qubit_count(self) -> int:
        if self.execution_mode == ExecutionMode.Sequential:
            return self.qubit_count_sequential()
        elif self.execution_mode == ExecutionMode.Parallel:
            return self.qubit_count_parallel()
        else:
            raise TypeError("Execution mode undefined")


class QAQC(QuantumCompilerGeneric):
    """Quantum-Assisted Quantum Compilation (QAQC)

    This class provides an implementation of QAQC https://quantum-
    journal.org/papers/q-2019-05-13-140/. The optimize method returns a
    compiled circuit along with the converged state of the optimizer. It
    is also possible to use the __call__ method to return the compiled
    circuit only.
    """

    def __init__(self, cost_fn: CostFunction, optimizer: Optimizer):
        super().__init__(cost_fn, optimizer)

    def run_time_scaling(self) -> Expr:
        pass

    @property
    def cost_function(self) -> CostFunction:
        return self._cost_fn

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def vm(self) -> Optional[VM]:
        return self._vm

    def run(
        self,
        circuit_factory: CircuitFactory,
        ansatz: LinearMappedUnboundParametricQuantumCircuit,
        init_params: Optional[Params] = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[NonParametricQuantumCircuit, OptimizerState]:
        """This function initiates the variational optimization.

        Arguments:
        circuit_factory - Circuit factory which generates the target circuit for the compilation
        ansatz - Parametric ansatz circuit that is used to approximate the target circuit
        init_params - initial variational parameters for the optimization

        variable arguments if any are passed to the circuit_factory's __call__ method

        Return value:
        Tuple of compiled circuit and the state of the optimizer
        """
        return self._optimize(circuit_factory, ansatz, init_params, *args, **kwargs)

    def analyze(
        self,
        circuit_factory: CircuitFactory,
        ansatz: LinearMappedUnboundParametricQuantumCircuit,
        optimizer_state: OptimizerState,
        circuit_execution_multiplier: Optional[int] = None,
        circuit_execution_mode: ExecutionMode = ExecutionMode.Sequential,
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
        assert isinstance(self.vm, VM), "VM must be set to perform analysis"

        total_circuit_executions = (
            ansatz.parameter_count * 2 * optimizer_state.gradcalls
            + optimizer_state.funcalls
        )
        if circuit_execution_multiplier is not None:
            total_circuit_executions *= circuit_execution_multiplier
        target_circuit = circuit_factory(*args, **kwargs)
        combined_circuit = prepare_circuit_hilbert_schmidt_test(
            target_circuit,
            ansatz.bind_parameters(list(optimizer_state.params.tolist())),
        ).circuit
        vm_analysis = self.vm.analyze(combined_circuit)
        analysis = QAQCAnalysis(
            lowering_level=vm_analysis.lowering_level,
            circuit_depth={combined_circuit: vm_analysis.depth},
            circuit_gate_count={combined_circuit: vm_analysis.gate_count},
            circuit_latency={combined_circuit: vm_analysis.latency},
            circuit_execution_count={combined_circuit: total_circuit_executions},
            circuit_fidelities={combined_circuit: vm_analysis.fidelity},
            circuit_qubit_count={combined_circuit: vm_analysis.qubit_count},
            execution_mode=circuit_execution_mode,
        )

        return analysis
