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

from quri_parts.circuit import QuantumCircuit
from quri_parts.core.measurement import (
    CommutablePauliSetMeasurement,
    CommutablePauliSetMeasurementFactory,
)
from quri_parts.core.operator import PAULI_IDENTITY, CommutablePauliSet, Operator
from quri_parts.core.sampling import PauliSamplingShotsAllocator
from quri_parts.core.state import CircuitQuantumState


def distribute_shots_among_pauli_sets(
    operator: Operator,
    measurements: Iterable[CommutablePauliSetMeasurement],
    shots_allocator: PauliSamplingShotsAllocator,
    total_shots: int,
) -> dict[CommutablePauliSet, int]:
    pauli_sets = {m.pauli_set for m in measurements}
    shot_allocs = shots_allocator(operator, pauli_sets, total_shots)
    return {pauli_set: n_shots for pauli_set, n_shots in shot_allocs}


def get_constant_seperated_measurement_group(
    operator: Operator, measurement_factory: CommutablePauliSetMeasurementFactory
) -> tuple[Iterable[CommutablePauliSetMeasurement], complex]:
    const = 0.0 + 0.0j
    measurement = []
    for m in measurement_factory(operator):
        if m.pauli_set == {PAULI_IDENTITY}:
            const += operator[PAULI_IDENTITY]
        else:
            measurement.append(m)
    return measurement, const


def remove_zero_shot_group(
    measurement_groups: Iterable[CommutablePauliSetMeasurement],
    shots_map: dict[CommutablePauliSet, int],
) -> Iterable[CommutablePauliSetMeasurement]:
    return [m for m in measurement_groups if shots_map[m.pauli_set] > 0]


def get_circuits_and_shot_pairs(
    state: CircuitQuantumState,
    measurement_groups: Iterable[CommutablePauliSetMeasurement],
    shots_map: dict[CommutablePauliSet, int],
) -> Iterable[tuple[QuantumCircuit, int]]:
    return [
        (state.circuit + m.measurement_circuit, shots_map[m.pauli_set])
        for m in measurement_groups
    ]
