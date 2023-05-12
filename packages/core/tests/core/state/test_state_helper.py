# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

import pytest

from quri_parts.circuit import H, QuantumCircuit, UnboundParametricQuantumCircuit, Z
from quri_parts.core.state import (
    ComputationalBasisState,
    GeneralCircuitQuantumState,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
    QuantumStateVector,
    apply_circuit,
    quantum_state,
)
from quri_parts.core.state.state_helper import (
    _circuit_quantum_state,
    _quantum_state_vector,
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


def test_circuit_quantum_state() -> None:
    n_qubits = 2
    qc = QuantumCircuit(n_qubits)
    qc.add_H_gate(1)
    state1 = _circuit_quantum_state(n_qubits, qc)
    assert isinstance(state1, GeneralCircuitQuantumState)

    param_qc = UnboundParametricQuantumCircuit(n_qubits)
    param_qc.add_ParametricRX_gate(0)
    state2 = _circuit_quantum_state(n_qubits, param_qc)
    assert isinstance(state2, ParametricCircuitQuantumState)


def test_quantum_state_vector() -> None:
    n_qubits = 2
    state1 = _quantum_state_vector(n_qubits, [1.0, 0, 0, 0], None)
    assert isinstance(state1, QuantumStateVector)

    qc = QuantumCircuit(n_qubits)
    qc.add_H_gate(1)
    state2 = _quantum_state_vector(n_qubits, [1.0, 0, 0, 0], qc)
    assert isinstance(state2, QuantumStateVector)

    param_qc = UnboundParametricQuantumCircuit(n_qubits)
    param_qc.add_ParametricRX_gate(0)
    state3 = _quantum_state_vector(n_qubits, [1.0, 0, 0, 0], param_qc)
    assert isinstance(state3, ParametricQuantumStateVector)


@patch("quri_parts.core.state.state_helper._quantum_state_vector")
@patch("quri_parts.core.state.state_helper._circuit_quantum_state")
def test_quantum_state(_circuit_quantum_state_mock, _quantum_state_vector_mock) -> None:
    n_qubits = 2
    _circuit_quantum_state_mock.return_value = GeneralCircuitQuantumState(n_qubits)
    _quantum_state_vector_mock.return_value = QuantumStateVector(n_qubits)

    state1 = quantum_state(n_qubits, bits=0b01)
    assert isinstance(state1, ComputationalBasisState)

    circuit = QuantumCircuit(0)
    state2 = quantum_state(n_qubits, bits=0b01, circuit=circuit)
    assert isinstance(state2, GeneralCircuitQuantumState)
    assert _circuit_quantum_state_mock.call_count == 1

    state3 = quantum_state(n_qubits, vector=[1.0, 0, 0, 0])
    assert isinstance(state3, QuantumStateVector)
    assert _quantum_state_vector_mock.call_count == 1

    with pytest.raises(ValueError):
        quantum_state(n_qubits, vector=[1.0, 0, 0, 0], bits=0b01)


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


def test_apply_circuit_with_parametric_state_and_general_circuit() -> None:
    state = b_state()
    circuit = a_circuit()
    combined_state = apply_circuit(state, circuit)

    assert isinstance(combined_state, ParametricCircuitQuantumState)
    assert len(combined_state.parametric_circuit.gates) == 4


def test_apply_circuit_with_parametric_state_and_parametric_circuit() -> None:
    state = b_state()
    circuit = b_circuit()
    combined_state = apply_circuit(state, circuit)

    assert isinstance(combined_state, ParametricCircuitQuantumState)
    assert len(combined_state.parametric_circuit.gates) == 6


def test_apply_circuit_with_parametric_state_vector_and_general_circuit() -> None:
    state = b_state_vector()
    circuit = a_circuit()
    combined_state = apply_circuit(state, circuit)

    assert isinstance(combined_state, ParametricQuantumStateVector)
    assert len(combined_state.parametric_circuit.gates) == 4


def test_apply_circuit_with_parametric_state_vector_and_parametric_circuit() -> None:
    state = b_state_vector()
    circuit = b_circuit()
    combined_state = apply_circuit(state, circuit)

    assert isinstance(combined_state, ParametricQuantumStateVector)
    assert len(combined_state.parametric_circuit.gates) == 6
