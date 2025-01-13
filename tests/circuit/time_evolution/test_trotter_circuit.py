# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from quri_parts.circuit import QuantumCircuit
from quri_parts.core.operator import PAULI_IDENTITY, Operator, PauliLabel, pauli_label

from quri_algo.circuit.time_evolution.trotter_time_evo import (
    FixedIntervalTrotterTimeEvolution,
    TrotterControlledTimeEvolutionCircuitFactory,
    TrotterTimeEvolutionCircuitFactory,
    get_trotter_time_evolution_operator,
)
from quri_algo.problem import QubitHamiltonianInput


def test_get_trotter_time_evolution_operator() -> None:
    operator = Operator(
        {pauli_label("X0 X1"): 2, pauli_label("Y0 Y1"): 3, PAULI_IDENTITY: 2}
    )

    # Trotter order 1
    trotter_unbound_circuit = get_trotter_time_evolution_operator(
        operator, n_state_qubits=2, n_trotter=1
    )
    trotter_circuit = trotter_unbound_circuit.bind_parameters([2.0])
    reconstructed_op = Operator({})
    for g in trotter_circuit.gates:
        reconstructed_op.add_term(
            PauliLabel.from_index_and_pauli_list(g.target_indices, g.pauli_ids),
            0.5 * g.params[0],
        )
    assert reconstructed_op == 2.0 * (operator - Operator({PAULI_IDENTITY: 2}))

    # Trotter order 2
    trotter_unbound_circuit = get_trotter_time_evolution_operator(
        operator, n_state_qubits=2, n_trotter=1, trotter_order=2
    )
    trotter_circuit = trotter_unbound_circuit.bind_parameters([2.0])
    reconstructed_op = Operator({})
    for g in trotter_circuit.gates:
        reconstructed_op.add_term(
            PauliLabel.from_index_and_pauli_list(g.target_indices, g.pauli_ids),
            0.5 * g.params[0],
        )
    assert reconstructed_op == 2 * (operator - Operator({PAULI_IDENTITY: 2}))


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

    trotter_order = 2
    n_trotter = 3
    circuit_factory = TrotterTimeEvolutionCircuitFactory(
        problem, n_trotter, trotter_order
    )
    circuit = circuit_factory(1.0)

    expected_circuit = QuantumCircuit(2)
    for _ in range(n_trotter):
        expected_circuit.add_PauliRotation_gate([0, 1], [1, 1], 2.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([0, 1], [2, 2], 2.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([0, 1], [2, 2], 2.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([0, 1], [1, 1], 2.0 / n_trotter)

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


def test_fixed_interval_trotter() -> None:
    operator = Operator(
        {pauli_label("X0 X1"): 2, pauli_label("Y0 Y1"): 2, PAULI_IDENTITY: 2}
    )
    problem = QubitHamiltonianInput(2, operator)
    time_step = 0.1
    circuit_factory = FixedIntervalTrotterTimeEvolution(
        problem, time_step, trotter_order=2
    )
    evolution_time = 0.5
    circuit = circuit_factory(evolution_time)

    single_step_circuit = QuantumCircuit(2)
    single_step_circuit.add_PauliRotation_gate([0, 1], [1, 1], 2 * time_step)
    single_step_circuit.add_PauliRotation_gate([0, 1], [2, 2], 2 * time_step)
    single_step_circuit.add_PauliRotation_gate([0, 1], [2, 2], 2 * time_step)
    single_step_circuit.add_PauliRotation_gate([0, 1], [1, 1], 2 * time_step)

    expected = QuantumCircuit(2)
    for _ in range(int(evolution_time / time_step)):
        expected.extend(single_step_circuit)

    assert expected == circuit

    evolution_time = 0.43
    with pytest.raises(
        ValueError,
        match=f"Evolution time {evolution_time} is not an integer muliple of time step {time_step}."
    ):
        circuit_factory(0.43)
