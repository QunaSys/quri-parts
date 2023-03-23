# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.chem.utils.orbital_rotation import add_orbital_rotation_gate
from quri_parts.circuit import CONST, LinearMappedUnboundParametricQuantumCircuit


def test_add_orbital_rotation_gate() -> None:
    qubit_count = 4
    qubit_indices = (0, 1, 2, 3)
    circuit = LinearMappedUnboundParametricQuantumCircuit(qubit_count)
    phi = circuit.add_parameter("phi")
    add_orbital_rotation_gate(circuit, qubit_indices, phi)
    expected_circuit = LinearMappedUnboundParametricQuantumCircuit(4)
    _phi = expected_circuit.add_parameter("phi")
    expected_circuit.add_H_gate(qubit_indices[3])
    expected_circuit.add_H_gate(qubit_indices[2])
    expected_circuit.add_CNOT_gate(qubit_indices[3], qubit_indices[1])
    expected_circuit.add_CNOT_gate(qubit_indices[2], qubit_indices[0])
    expected_circuit.add_ParametricRY_gate(qubit_indices[3], {_phi: 0.5})
    expected_circuit.add_ParametricRY_gate(qubit_indices[2], {_phi: 0.5})
    expected_circuit.add_ParametricRY_gate(qubit_indices[1], {_phi: 0.5})
    expected_circuit.add_ParametricRY_gate(qubit_indices[0], {_phi: 0.5})
    expected_circuit.add_CNOT_gate(qubit_indices[3], qubit_indices[1])
    expected_circuit.add_CNOT_gate(qubit_indices[2], qubit_indices[0])
    expected_circuit.add_H_gate(qubit_indices[3])
    expected_circuit.add_H_gate(qubit_indices[2])
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit

    circuit = LinearMappedUnboundParametricQuantumCircuit(qubit_count)
    phi = circuit.add_parameter("phi")
    add_orbital_rotation_gate(circuit, qubit_indices, {phi: -2.0})
    expected_circuit = LinearMappedUnboundParametricQuantumCircuit(4)
    _phi = expected_circuit.add_parameter("phi")
    expected_circuit.add_H_gate(qubit_indices[3])
    expected_circuit.add_H_gate(qubit_indices[2])
    expected_circuit.add_CNOT_gate(qubit_indices[3], qubit_indices[1])
    expected_circuit.add_CNOT_gate(qubit_indices[2], qubit_indices[0])
    expected_circuit.add_ParametricRY_gate(qubit_indices[3], {_phi: -1.0})
    expected_circuit.add_ParametricRY_gate(qubit_indices[2], {_phi: -1.0})
    expected_circuit.add_ParametricRY_gate(qubit_indices[1], {_phi: -1.0})
    expected_circuit.add_ParametricRY_gate(qubit_indices[0], {_phi: -1.0})
    expected_circuit.add_CNOT_gate(qubit_indices[3], qubit_indices[1])
    expected_circuit.add_CNOT_gate(qubit_indices[2], qubit_indices[0])
    expected_circuit.add_H_gate(qubit_indices[3])
    expected_circuit.add_H_gate(qubit_indices[2])
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit

    circuit = LinearMappedUnboundParametricQuantumCircuit(qubit_count)
    add_orbital_rotation_gate(circuit, qubit_indices, {CONST: 0.5})
    expected_circuit = LinearMappedUnboundParametricQuantumCircuit(4)
    expected_circuit.add_H_gate(qubit_indices[3])
    expected_circuit.add_H_gate(qubit_indices[2])
    expected_circuit.add_CNOT_gate(qubit_indices[3], qubit_indices[1])
    expected_circuit.add_CNOT_gate(qubit_indices[2], qubit_indices[0])
    expected_circuit.add_ParametricRY_gate(qubit_indices[3], {CONST: 0.25})
    expected_circuit.add_ParametricRY_gate(qubit_indices[2], {CONST: 0.25})
    expected_circuit.add_ParametricRY_gate(qubit_indices[1], {CONST: 0.25})
    expected_circuit.add_ParametricRY_gate(qubit_indices[0], {CONST: 0.25})
    expected_circuit.add_CNOT_gate(qubit_indices[3], qubit_indices[1])
    expected_circuit.add_CNOT_gate(qubit_indices[2], qubit_indices[0])
    expected_circuit.add_H_gate(qubit_indices[3])
    expected_circuit.add_H_gate(qubit_indices[2])
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit
