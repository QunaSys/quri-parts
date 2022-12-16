# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable

from quri_parts.circuit import NonParametricQuantumCircuit, QuantumCircuit
from quri_parts.core.sampling import MeasurementCounts
from quri_parts.core.sampling.qubit_remapping import (
    create_qubit_remapping_concurrent_sampler,
    create_qubit_remapping_sampler,
)

qubit_remapping = {
    0: 4,
    1: 2,
    2: 0,
}

original_counts = {
    0b000: 100,
    0b001: 200,
    0b010: 300,
    0b011: 400,
    0b100: 500,
    0b101: 600,
    0b110: 700,
    0b111: 800,
}

mapped_counts = {
    # 0b000
    0b00000: 20,
    0b00010: 10,
    0b01000: 70,
    # 0b001
    0b10000: 50,
    0b11010: 150,
    # 0b010
    0b00100: 300,
    # 0b011
    0b10100: 400,
    # 0b100
    0b00001: 500,
    # 0b101
    0b10001: 600,
    # 0b110
    0b00101: 700,
    # 0b111
    0b10101: 800,
}


def test_create_qubit_remapping_sampler() -> None:
    def sampler(
        circuit: NonParametricQuantumCircuit, n_shots: int
    ) -> MeasurementCounts:
        return mapped_counts

    remapping_sampler = create_qubit_remapping_sampler(sampler, qubit_remapping)

    measurement_counts = remapping_sampler(QuantumCircuit(3), 3600)

    assert measurement_counts == original_counts


def test_create_qubit_remapping_concurrent_sampler() -> None:
    def sampler(
        circuit_and_shots: Iterable[tuple[NonParametricQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        return [mapped_counts] * len(list(circuit_and_shots))

    remapping_sampler = create_qubit_remapping_concurrent_sampler(
        sampler, qubit_remapping
    )

    measurement_counts = remapping_sampler([(QuantumCircuit(3), 3600)] * 2)

    assert list(measurement_counts) == [original_counts] * 2
