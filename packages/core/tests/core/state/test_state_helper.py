# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import H, QuantumCircuit, Z
from quri_parts.core.state import (
    ComputationalBasisState,
    GeneralCircuitQuantumState,
    QuantumStateVector,
    quantum_state,
)


def a_state() -> GeneralCircuitQuantumState:
    s = GeneralCircuitQuantumState(2)
    s = s.with_gates_applied([H(0), Z(1)])
    return s


def test_quantum_state_without_circuit() -> None:
    n_qubits = 2
    state = quantum_state(n_qubits)

    assert isinstance(state, GeneralCircuitQuantumState)
    assert state.qubit_count == n_qubits


def test_quantum_state_with_circuit() -> None:
    n_qubits = 2
    circuit = QuantumCircuit(n_qubits)
    circuit.add_H_gate(1)
    state = quantum_state(n_qubits, circuit=circuit)

    assert isinstance(state, GeneralCircuitQuantumState)
    assert state.qubit_count == n_qubits
    assert len(state.circuit.gates) == 1


def test_quantum_state_with_vector() -> None:
    n_qubits = 2
    state = quantum_state(n_qubits, [1.0, 0, 0, 0])

    assert isinstance(state, QuantumStateVector)
    assert state.qubit_count == n_qubits


def test_quantum_state_with_vector_and_circuit() -> None:
    n_qubits = 2
    circuit = QuantumCircuit(n_qubits)
    circuit.add_H_gate(1)
    state = quantum_state(n_qubits, [1.0, 0, 0, 0], circuit=circuit)

    assert isinstance(state, QuantumStateVector)
    assert state.qubit_count == n_qubits
    assert len(state.circuit.gates) == 1


def test_quantum_state_with_cb() -> None:
    n_qubits = 4

    state = quantum_state(n_qubits, bits=0b0011)

    assert isinstance(state, ComputationalBasisState)


# def test_apply_circuit_empty() -> None:
#     state = a_state()
#     state2 = a_state()
#     circuit = QuantumCircuit(2)
#     combined_state = apply_circuit(state, circuit)

#     assert isinstance(combined_state, GeneralCircuitQuantumState)
#     assert combined_state._circuit == state2._circuit

# def test_apply_circuit() -> None:
#     state = a_state()
#     state2 = state.with_gates_applied([H(1)])
#     circuit = QuantumCircuit(2)
#     circuit.add_H_gate(1)
#     combined_state = apply_circuit(state, circuit)

#     assert isinstance(combined_state, GeneralCircuitQuantumState)
#     assert state._circuit == state2._circuit
