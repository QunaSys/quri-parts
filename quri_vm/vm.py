# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.backend.device import DeviceProperty
from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.core.estimator import Estimatable, Estimate
from quri_parts.core.sampling import MeasurementCounts
from quri_parts.core.state import CircuitQuantumState, quantum_state

from .simulator import LogicalCircuitSimulator, QulacsSimulator
from .vm_backend import AnalyzeResult, LoweringLevel, VMBackend, _DevicePropertyBackend


class VM:
    @staticmethod
    def from_device_prop(prop: DeviceProperty) -> "VM":
        return VM(vm_backend=_DevicePropertyBackend(prop))

    def __init__(
        self,
        vm_backend: VMBackend | None = None,
        lowering_level: LoweringLevel | None = None,
        logical_circuit_simulator: LogicalCircuitSimulator = QulacsSimulator(),
    ):
        self._logical_circuit_simulator = logical_circuit_simulator

        if vm_backend is None:
            lowering_level = LoweringLevel.LogicalCircuit
        elif lowering_level is None:
            lowering_level = max(vm_backend.lowering_levels)

        self._set_vm_backend(vm_backend, lowering_level)

    def _set_vm_backend(
        self, vm_backend: VMBackend | None, lowering_level: LoweringLevel
    ) -> None:
        if vm_backend is None and lowering_level != LoweringLevel.LogicalCircuit:
            raise ValueError(
                "When VMBackend is not specified, only LoweringLevel.LogicalCircuit"
                " is supported."
            )
        if vm_backend is not None and lowering_level not in {
            LoweringLevel.LogicalCircuit
        } | set(vm_backend.lowering_levels):
            raise ValueError(
                "Required LoweringLevel is not supported by given VMBackend:"
                f" {lowering_level}"
            )
        self._backend = vm_backend
        self._lowering_level = lowering_level

    def estimate(
        self,
        estimatable: Estimatable,
        state: CircuitQuantumState,
    ) -> Estimate[complex]:
        match self._lowering_level:
            case LoweringLevel.LogicalCircuit:
                return self._logical_circuit_simulator.estimate(estimatable, state)
            case LoweringLevel.ArchLogicalCircuit:
                circuit = self.transpile(state.circuit)
                return self._logical_circuit_simulator.estimate(
                    estimatable,
                    quantum_state(state.qubit_count, circuit=circuit),
                    self._backend.noise_model if self._backend is not None else None,
                )
            case _:
                raise ValueError(
                    "Eestimator is not available for the required LoweringLevel."
                    " Please use sampling estimater instead."
                )

    def sample(
        self,
        circuit: NonParametricQuantumCircuit,
        shots: int,
    ) -> MeasurementCounts:
        match self._lowering_level:
            case LoweringLevel.LogicalCircuit:
                return self._logical_circuit_simulator.sample(circuit, shots)
            case LoweringLevel.ArchLogicalCircuit:
                if self._backend is not None:
                    circuit = self.transpile(circuit)
                    return self._logical_circuit_simulator.sample(circuit, shots)
            case _:
                if self._backend is not None:
                    return self._backend.sample(circuit, shots, self._lowering_level)
        raise ValueError("Cannot execute for the required LoweringLevel.")

    def transpile(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        if self._lowering_level == LoweringLevel.LogicalCircuit:
            return circuit
        elif self._backend is not None:
            return self._backend.transpile(circuit, self._lowering_level)
        else:
            raise ValueError(
                "Transpiler is not supported for the required LoweringLevel."
            )

    def analyze(self, circuit: NonParametricQuantumCircuit) -> AnalyzeResult:
        if self._lowering_level == LoweringLevel.LogicalCircuit:
            return AnalyzeResult(
                lowering_level=self._lowering_level,
                qubit_count=circuit.qubit_count,
                gate_count=len(circuit.gates),
                depth=circuit.depth,
                fidelity=1.0,
            )
        elif self._backend is not None:
            return self._backend.analyze(circuit, self._lowering_level)
        else:
            raise ValueError("Cannot analyze for the required LoweringLevel.")
