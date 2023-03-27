# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from quri_parts.circuit.utils.controlled_rotations import add_controlled_RY_gate


def test_add_controlled_RY_gate() -> None:
    qubit_count = 4
    excitation = (0, 2)
    circuit = LinearMappedUnboundParametricQuantumCircuit(qubit_count)
    theta = circuit.add_parameter("theta")
    add_controlled_RY_gate(circuit, excitation[1], excitation[0], theta)
    expected_circuit = LinearMappedUnboundParametricQuantumCircuit(qubit_count)
    exp_theta = expected_circuit.add_parameter("theta")
    expected_circuit.add_ParametricRY_gate(excitation[0], {exp_theta: 0.5})
    expected_circuit.add_CNOT_gate(excitation[1], excitation[0])
    expected_circuit.add_ParametricRY_gate(excitation[0], {exp_theta: -0.5})
    expected_circuit.add_CNOT_gate(excitation[1], excitation[0])
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit

    qubit_count = 4
    excitation = (1, 3)
    circuit = LinearMappedUnboundParametricQuantumCircuit(qubit_count)
    theta = circuit.add_parameter("theta")
    add_controlled_RY_gate(circuit, excitation[1], excitation[0], {theta: 0.5})
    expected_circuit = LinearMappedUnboundParametricQuantumCircuit(qubit_count)
    exp_theta = expected_circuit.add_parameter("theta")
    expected_circuit.add_ParametricRY_gate(excitation[0], {exp_theta: 0.25})
    expected_circuit.add_CNOT_gate(excitation[1], excitation[0])
    expected_circuit.add_ParametricRY_gate(excitation[0], {exp_theta: -0.25})
    expected_circuit.add_CNOT_gate(excitation[1], excitation[0])
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit

    qubit_count = 6
    excitation = (1, 3)
    circuit = LinearMappedUnboundParametricQuantumCircuit(qubit_count)
    theta = circuit.add_parameter("theta")
    add_controlled_RY_gate(circuit, excitation[1], excitation[0], {theta: -0.5})
    expected_circuit = LinearMappedUnboundParametricQuantumCircuit(qubit_count)
    exp_theta = expected_circuit.add_parameter("theta")
    expected_circuit.add_ParametricRY_gate(excitation[0], {exp_theta: -0.25})
    expected_circuit.add_CNOT_gate(excitation[1], excitation[0])
    expected_circuit.add_ParametricRY_gate(excitation[0], {exp_theta: 0.25})
    expected_circuit.add_CNOT_gate(excitation[1], excitation[0])
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit
