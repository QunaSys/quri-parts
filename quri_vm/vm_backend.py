# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

from quri_parts.backend import cost_estimator
from quri_parts.backend.device import DeviceProperty
from quri_parts.backend.units import TimeValue
from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.noise import NoiseModel
from quri_parts.core.sampling import MeasurementCounts


class LoweringLevel(IntEnum):
    LogicalCircuit = 0
    ArchLogicalCircuit = 1
    ArchInstruction = 2
    DeviceInstruction = 3


@dataclass
class AnalyzeResult:
    lowering_level: LoweringLevel
    qubit_count: int
    gate_count: int
    depth: int
    latency: TimeValue | None = None
    fidelity: float | None = None


class VMBackend(ABC):
    @abstractmethod
    def sample(
        self,
        circuit: NonParametricQuantumCircuit,
        shots: int,
        lowering_level: LoweringLevel | None = None,
    ) -> MeasurementCounts: ...

    @abstractmethod
    def analyze(
        self,
        circuit: NonParametricQuantumCircuit,
        lowering_level: LoweringLevel | None = None,
    ) -> AnalyzeResult: ...

    @abstractmethod
    def transpile(
        self,
        circuit: NonParametricQuantumCircuit,
        lowering_level: LoweringLevel | None = None,
    ) -> NonParametricQuantumCircuit: ...

    @property
    @abstractmethod
    def noise_model(self) -> NoiseModel | None: ...

    @property
    @abstractmethod
    def lowering_levels(self) -> Sequence[LoweringLevel]: ...

    def _check_select_lowering_level(
        self, lowering_level: LoweringLevel | None
    ) -> LoweringLevel:
        if lowering_level is None:
            return max(self.lowering_levels)
        elif lowering_level not in self.lowering_levels:
            raise ValueError(f"Unsupported LoweringLevel: {lowering_level}")
        else:
            return lowering_level


class _DevicePropertyBackend(VMBackend):
    def __init__(self, device: DeviceProperty) -> None:
        self._device = device

    def sample(
        self,
        circuit: NonParametricQuantumCircuit,
        shots: int,
        lowering_level: LoweringLevel | None = None,
    ) -> MeasurementCounts:
        raise ValueError("This backend does not support sampling.")

    def analyze(
        self,
        circuit: NonParametricQuantumCircuit,
        lowering_level: LoweringLevel | None = None,
    ) -> AnalyzeResult:
        lowering_level = self._check_select_lowering_level(lowering_level)

        if (
            lowering_level == LoweringLevel.ArchLogicalCircuit
            and self._device.analyze_transpiler is not None
        ):
            circuit = self._device.analyze_transpiler(circuit)
        else:
            circuit = self.transpile(circuit, lowering_level)
        return AnalyzeResult(
            lowering_level=lowering_level,
            qubit_count=circuit.qubit_count,
            gate_count=len(circuit.gates),
            depth=circuit.depth,
            latency=cost_estimator.estimate_circuit_latency(circuit, self._device),
            fidelity=cost_estimator.estimate_circuit_fidelity(circuit, self._device),
        )

    def transpile(
        self,
        circuit: NonParametricQuantumCircuit,
        lowering_level: LoweringLevel | None = None,
    ) -> NonParametricQuantumCircuit:
        self._check_select_lowering_level(lowering_level)

        if self._device.transpiler is not None:
            return self._device.transpiler(circuit)
        else:
            return circuit

    @property
    def noise_model(self) -> NoiseModel | None:
        return self._device.noise_model

    @property
    def lowering_levels(self) -> Sequence[LoweringLevel]:
        return [LoweringLevel.ArchLogicalCircuit]


# class ActualDeviceBackend(VMBackend): ...
