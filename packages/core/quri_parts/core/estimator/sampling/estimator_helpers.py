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
from typing import Callable

from typing_extensions import TypeAlias

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.core.measurement import CommutablePauliSetMeasurement
from quri_parts.core.operator import PAULI_IDENTITY, CommutablePauliSet, Operator
from quri_parts.core.sampling import PauliSamplingShotsAllocator
from quri_parts.core.state import CircuitQuantumState

CircuitShotPairPreparationFunction: TypeAlias = Callable[
    [
        CircuitQuantumState,
        Iterable[CommutablePauliSetMeasurement],
        dict[CommutablePauliSet, int],
    ],
    Iterable[tuple[NonParametricQuantumCircuit, int]],
]


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
    operator: Operator, measurement_groups: Iterable[CommutablePauliSetMeasurement]
) -> tuple[Iterable[CommutablePauliSetMeasurement], complex]:
    """Seperate the identity part from the operator."""
    const = 0.0 + 0.0j
    measurement = []
    for m in measurement_groups:
        if m.pauli_set == {PAULI_IDENTITY}:
            const += operator[PAULI_IDENTITY]
        else:
            measurement.append(m)
    return measurement, const


def circuit_shot_pairs_preparation_fn(
    state: CircuitQuantumState,
    measurement_groups: Iterable[CommutablePauliSetMeasurement],
    shots_map: dict[CommutablePauliSet, int],
) -> Iterable[tuple[NonParametricQuantumCircuit, int]]:
    """A function that concatenates the measurement circuit after the circuit
    quantum state."""
    return [
        (
            state.circuit + m.measurement_circuit,
            n_shots,
        )
        for m in measurement_groups
        if (n_shots := shots_map[m.pauli_set]) > 0
    ]
