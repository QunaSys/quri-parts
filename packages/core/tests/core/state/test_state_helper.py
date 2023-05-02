# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import (
    RX,
    RY,
    H,
    QuantumCircuit,
    UnboundParametricQuantumCircuit,
    Z,
)
from quri_parts.core.state import (
    ComputationalBasisState,
    GeneralCircuitQuantumState,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
    QuantumStateVector,
    apply_circuit,
    quantum_state,
)


def a_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.add_H_gate(1)
    return circuit


def b_circuit() -> UnboundParametricQuantumCircuit:
    circuit = UnboundParametricQuantumCircuit(2)
    circuit.add_H_gate(0)
    circuit.add_Z_gate(1)
    circuit.add_ParametricRX_gate(0)
    return circuit


def a_state() -> GeneralCircuitQuantumState:
    s = GeneralCircuitQuantumState(2)
    s = s.with_gates_applied([H(0), Z(1)])
    return s


def b_state() -> ParametricCircuitQuantumState:
    return ParametricCircuitQuantumState(2, b_circuit())


def a_state_vector() -> QuantumStateVector:
    return QuantumStateVector(2, vector=[1.0, 0, 0, 0])


def b_state_vector() -> ParametricQuantumStateVector:
    return ParametricQuantumStateVector(2, vector=[1.0, 0, 0, 0], circuit=b_circuit())


def test_quantum_state_init() -> None:
    n_qubits = 2
    state = quantum_state(n_qubits)

    assert isinstance(state, GeneralCircuitQuantumState)
    assert state.qubit_count == n_qubits


def test_quantum_state_with_vector() -> None:
    n_qubits = 2
    state = quantum_state(n_qubits, [1.0, 0, 0, 0])

    assert isinstance(state, QuantumStateVector)
    assert state.qubit_count == n_qubits


def test_quantum_state_with_circuit() -> None:
    n_qubits = 2
    circuit = a_circuit()
    state = quantum_state(n_qubits, circuit=circuit)

    assert isinstance(state, GeneralCircuitQuantumState)
    assert state.qubit_count == n_qubits
    assert state.circuit == circuit


def test_quantum_state_with_parametric_circuit() -> None:
    n_qubits = 2
    circuit = b_circuit()
    state = quantum_state(n_qubits, circuit=circuit)

    assert isinstance(state, ParametricCircuitQuantumState)
    assert state.qubit_count == n_qubits
    assert state.parametric_circuit == circuit


def test_quantum_state_with_cb() -> None:
    n_qubits = 2
    circuit = QuantumCircuit(2)
    circuit.add_X_gate(0)
    state = quantum_state(n_qubits, bits=0b01)

    assert isinstance(state, ComputationalBasisState)
    assert state.circuit == circuit


def test_quantum_state_with_vector_and_circuit() -> None:
    n_qubits = 2
    circuit = a_circuit()
    state = quantum_state(n_qubits, [1.0, 0, 0, 0], circuit=circuit)
    exp_state = a_state_vector()

    assert isinstance(state, QuantumStateVector)
    assert state.qubit_count == n_qubits
    assert (state.vector == exp_state.vector).all()
    assert state.circuit == circuit


def test_quantum_state_with_vector_and_parametric_circuit() -> None:
    n_qubits = 2
    circuit = b_circuit()
    state = quantum_state(n_qubits, [1.0, 0, 0, 0], circuit=circuit)

    assert isinstance(state, ParametricQuantumStateVector)
    assert state.qubit_count == n_qubits
    assert state.parametric_circuit == circuit


def test_quantum_state_with_circuit_and_bits() -> None:
    n_qubits = 2
    circuit = a_circuit()
    state = quantum_state(n_qubits, circuit=circuit, bits=0b01)
    exp_circuit = QuantumCircuit(2)
    exp_circuit.add_X_gate(0)
    exp_circuit.add_H_gate(1)

    assert isinstance(state, GeneralCircuitQuantumState)
    assert state.qubit_count == n_qubits
    assert state.circuit == exp_circuit


def test_quantum_state_with_parametric_circuit_and_bits() -> None:
    n_qubits = 2
    circuit = b_circuit()
    state = quantum_state(n_qubits, circuit=circuit, bits=0b01)

    assert isinstance(state, ParametricCircuitQuantumState)
    assert state.qubit_count == n_qubits
    assert len(state.parametric_circuit.gates) == 4
    assert state.parametric_circuit.gates[0].name == 'X'
    assert state.parametric_circuit.gates[0].target_indices == (0,)


def test_apply_circuit_empty() -> None:
    state = a_state()
    circuit = QuantumCircuit(2)
    combined_state = apply_circuit(state, circuit)

    assert isinstance(combined_state, GeneralCircuitQuantumState)


def test_apply_circuit_with_general_circuit() -> None:
    state = a_state()
    circuit = a_circuit()
    combined_state = apply_circuit(state, circuit)
    exp_state = state.with_gates_applied([H(1)])

    assert isinstance(combined_state, GeneralCircuitQuantumState)
    assert combined_state.circuit == exp_state.circuit


def test_apply_circuit_with_parametric_circuit() -> None:
    state = a_state()
    circuit = b_circuit()
    combined_state = apply_circuit(state, circuit)

    assert isinstance(combined_state, ParametricCircuitQuantumState)
    assert len(combined_state.parametric_circuit.gates) == 5


def test_apply_circuit_with_state_vector_and_general_circuit() -> None:
    state = a_state_vector()
    circuit = a_circuit()
    combined_state = apply_circuit(state, circuit)

    assert isinstance(combined_state, QuantumStateVector)
    assert combined_state.circuit == circuit


def test_apply_circuit_with_state_vector_and_parametric_circuit() -> None:
    state = a_state_vector()
    circuit = b_circuit()
    combined_state = apply_circuit(state, circuit)

    assert isinstance(combined_state, ParametricQuantumStateVector)
    assert combined_state.parametric_circuit == circuit


def test_apply_circuit_with_parametric_state_and_general_circuit():
    state = b_state()
    circuit = a_circuit()
    combined_state = apply_circuit(state, circuit)
    exp_state = state.with_gates_applied(circuit)

    assert isinstance(combined_state, ParametricCircuitQuantumState)
    for actual, exp in zip(combined_state.parametric_circuit.gates, exp_state.parametric_circuit.gates):
        assert actual.name == exp.name
        assert actual.target_indices == exp.target_indices


def test_apply_circuit_with_parametric_state_and_parametric_circuit():
    state = b_state()
    circuit = b_circuit()
    combined_state = apply_circuit(state, circuit)
    exp_state = state.with_gates_applied(circuit)

    assert isinstance(combined_state, ParametricCircuitQuantumState)
    for actual, exp in zip(combined_state.parametric_circuit.gates, exp_state.parametric_circuit.gates):
        assert actual.name == exp.name
        assert actual.target_indices == exp.target_indices

def test_apply_circuit_with_parametric_state_vector_and_general_circuit():
    state = b_state_vector()
    circuit = a_circuit()
    combined_state = apply_circuit(state, circuit)
    exp_state = state.with_gates_applied(circuit)

    assert isinstance(state, ParametricQuantumStateVector)
    for actual, exp in zip(combined_state.parametric_circuit.gates, exp_state.parametric_circuit.gates):
        assert actual.name == exp.name
        assert actual.target_indices == exp.target_indices


def test_apply_circuit_with_parametric_state_vector_and_parametric_circuit():
    state = b_state_vector()
    circuit = b_circuit()
    combined_state = apply_circuit(state, circuit)
    exp_state = state.with_gates_applied(circuit)

    assert isinstance(state, ParametricQuantumStateVector)
    for actual, exp in zip(combined_state.parametric_circuit.gates, exp_state.parametric_circuit.gates):
        assert actual.name == exp.name
        assert actual.target_indices == exp.target_indices
