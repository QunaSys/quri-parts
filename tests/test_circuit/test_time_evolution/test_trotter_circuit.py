# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import QuantumCircuit
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label

from quri_algo.circuit.time_evolution.trotter_time_evo import (
    TrotterControlledTimeEvolutionCircuitFactory,
    TrotterTimeEvolutionCircuitFactory,
)
from quri_algo.problem import QubitHamiltonianInput


def test_trotter_factory() -> None:
    operator = Operator(
        {pauli_label("X0 X1"): 2, pauli_label("Y0 Y1"): 2, PAULI_IDENTITY: 2}
    )
    problem = QubitHamiltonianInput(2, operator)
    circuit_factory = TrotterTimeEvolutionCircuitFactory(problem, 1)
    circuit = circuit_factory(1.0)
    expected_circuit = QuantumCircuit(2)
    expected_circuit.add_PauliRotation_gate([0, 1], [1, 1], 4.0)
    expected_circuit.add_PauliRotation_gate([0, 1], [2, 2], 4.0)
    assert circuit == expected_circuit

    n_trotter = 5
    circuit_factory = TrotterTimeEvolutionCircuitFactory(problem, n_trotter)
    circuit = circuit_factory(1.0)
    expected_circuit = QuantumCircuit(2)
    for _ in range(n_trotter):
        expected_circuit.add_PauliRotation_gate([0, 1], [1, 1], 4.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([0, 1], [2, 2], 4.0 / n_trotter)
    assert circuit == expected_circuit


def test_controlled_trotter_factory() -> None:
    operator = Operator(
        {pauli_label("X0 X1"): 2, pauli_label("Y0 Y1"): 2, PAULI_IDENTITY: 2}
    )
    problem = QubitHamiltonianInput(2, operator)
    circuit_factory = TrotterControlledTimeEvolutionCircuitFactory(problem, 1)
    circuit = circuit_factory(1.0)
    expected_circuit = QuantumCircuit(3)
    expected_circuit.add_PauliRotation_gate([1, 2], [1, 1], 2.0)
    expected_circuit.add_PauliRotation_gate([0, 1, 2], [3, 1, 1], -2.0)
    expected_circuit.add_PauliRotation_gate([1, 2], [2, 2], 2.0)
    expected_circuit.add_PauliRotation_gate([0, 1, 2], [3, 2, 2], -2.0)
    expected_circuit.add_RZ_gate(0, -2.0)
    assert circuit == expected_circuit

    n_trotter = 5
    circuit_factory = TrotterControlledTimeEvolutionCircuitFactory(problem, n_trotter)
    circuit = circuit_factory(1.0)
    expected_circuit = QuantumCircuit(3)
    for _ in range(n_trotter):
        expected_circuit.add_PauliRotation_gate([1, 2], [1, 1], 2.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([0, 1, 2], [3, 1, 1], -2.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([1, 2], [2, 2], 2.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([0, 1, 2], [3, 2, 2], -2.0 / n_trotter)
        expected_circuit.add_RZ_gate(0, -2.0 / n_trotter)
    assert circuit == expected_circuit
