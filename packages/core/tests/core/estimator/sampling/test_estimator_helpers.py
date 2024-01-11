# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import QuantumCircuit
from quri_parts.core.estimator.sampling.estimator_helpers import (
    circuit_shot_pairs_preparation_fn,
    distribute_shots_among_pauli_sets,
)
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.sampling.shots_allocator import (
    create_equipartition_shots_allocator,
)
from quri_parts.core.state import GeneralCircuitQuantumState


def test_distribute_shots_among_pauli_sets() -> None:
    operator = Operator({pauli_label("X0 X1"): 1, pauli_label("Y0 X1"): 1})
    groups = bitwise_commuting_pauli_measurement(operator)
    shots_allocator = create_equipartition_shots_allocator()
    expected_distribution = {
        frozenset({pauli_label("X0 X1")}): 500,
        frozenset({pauli_label("Y0 X1")}): 500,
    }

    distribution = distribute_shots_among_pauli_sets(
        operator, groups, shots_allocator, total_shots=1000
    )
    assert distribution == expected_distribution


def test_circuit_shot_pairs_preparation_fn() -> None:
    circuit = QuantumCircuit(2)
    circuit.add_H_gate(0)
    circuit.add_CNOT_gate(0, 1)
    state = GeneralCircuitQuantumState(2, circuit)

    operator = Operator({pauli_label("X0 X1"): 1, pauli_label("Y0 X1"): 1})
    groups = bitwise_commuting_pauli_measurement(operator)
    distribution = {
        frozenset({pauli_label("X0 X1")}): 500,
        frozenset({pauli_label("Y0 X1")}): 500,
    }

    pairs = circuit_shot_pairs_preparation_fn(state, groups, distribution)
    assert len(pairs) == 2
    for p, g in zip(pairs, groups):
        assert p == (circuit.combine(g.measurement_circuit), 500)
