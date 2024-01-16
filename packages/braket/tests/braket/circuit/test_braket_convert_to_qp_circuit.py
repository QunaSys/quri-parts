# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Callable, Mapping, Type

from braket.circuits import Circuit, Gate, Instruction
from scipy.stats import unitary_group

from quri_parts.braket.circuit import circuit_from_braket, gate_from_braket
from quri_parts.circuit import QuantumCircuit, QuantumGate, gates

single_qubit_gate_mapping: Mapping[Callable[[int], QuantumGate], Gate] = {
    gates.Identity: Gate.I(),
    gates.X: Gate.X(),
    gates.Y: Gate.Y(),
    gates.Z: Gate.Z(),
    gates.H: Gate.H(),
    gates.S: Gate.S(),
    gates.Sdag: Gate.Si(),
    gates.T: Gate.T(),
    gates.Tdag: Gate.Ti(),
}


def test_single_qubit_gate_conversion() -> None:
    for qp_factory, braket_gate in single_qubit_gate_mapping.items():
        expected = qp_factory(7)
        braket_instruction = Instruction(braket_gate, 7)
        converted = gate_from_braket(braket_instruction)
        assert expected == converted


two_qubit_gate_mapping: Mapping[Callable[[int, int], QuantumGate], Gate] = {
    gates.CNOT: Gate.CNot(),
    gates.CZ: Gate.CZ(),
    gates.SWAP: Gate.Swap(),
}


def test_two_qubit_gate_conversion() -> None:
    for qp_factory, braket_gate in two_qubit_gate_mapping.items():
        expected = qp_factory(11, 7)
        braket_instruction = Instruction(braket_gate, [11, 7])
        converted = gate_from_braket(braket_instruction)
        assert expected == converted


def test_three_qubit_gate_conversion() -> None:
    expected = gates.TOFFOLI(11, 7, 5)
    braket_instruction = Instruction(Gate.CCNot(), [11, 7, 5])
    converted = gate_from_braket(braket_instruction)
    assert expected == converted


rotation_gate_mapping: Mapping[Callable[[int, float], QuantumGate], Type[Gate]] = {
    gates.RX: Gate.Rx,
    gates.RY: Gate.Ry,
    gates.RZ: Gate.Rz,
}


def test_rotation_gate_conversion() -> None:
    for qp_factory, braket_type in rotation_gate_mapping.items():
        expected = qp_factory(7, 0.125)
        braket_instruction = Instruction(braket_type(angle=0.125), 7)
        converted = gate_from_braket(braket_instruction)
        assert converted == expected


def test_unitary_gate_conversion() -> None:
    matrix = unitary_group.rvs(8)
    expected = gates.UnitaryMatrix([0, 4, 7], matrix)
    braket_gate = Gate.Unitary(matrix)
    converted = gate_from_braket(Instruction(braket_gate, [7, 4, 0]))
    assert converted == expected


def test_circuit_conversion() -> None:
    braket_circuit = Circuit()
    braket_circuit.x(1).h(2).cnot(0, 1).rx(0, 0.125)

    qp_circuit = QuantumCircuit(3)
    qp_circuit.add_X_gate(1)
    qp_circuit.add_H_gate(2)
    qp_circuit.add_CNOT_gate(0, 1)
    qp_circuit.add_RX_gate(0, 0.125)

    assert qp_circuit == circuit_from_braket(braket_circuit)
