# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping
from typing import Callable, Type, cast

import numpy as np
import qiskit.circuit.library as qgate

# from qiskit.circuit import Instruction
from qiskit.circuit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.compiler import transpile
from qiskit.extensions import UnitaryGate
from qiskit.opflow import I, X, Y, Z

from quri_parts.circuit import (
    LinearMappedUnboundParametricQuantumCircuit,
    QuantumCircuit,
    QuantumGate,
    UnboundParametricQuantumCircuit,
    gates,
)
from quri_parts.qiskit.circuit import convert_circuit, convert_gate


def gate_equal(i1: Gate, i2: Gate) -> bool:
    # Compare braket IR dict expression in pydantic BaseModel.
    return cast(bool, i1.decompositions == i2.decompositions)


def circuit_equal(c1: QiskitQuantumCircuit, c2: QiskitQuantumCircuit) -> bool:
    # Compare braket IR dict expression in pydantic BaseModel.
    return cast(bool, transpile(c1) == transpile(c2))


single_qubit_gate_mapping: Mapping[Callable[[int], QuantumGate], Gate] = {
    gates.Identity: qgate.IGate(),
    gates.X: qgate.XGate(),
    gates.Y: qgate.YGate(),
    gates.Z: qgate.ZGate(),
    gates.H: qgate.HGate(),
    gates.S: qgate.SGate(),
    gates.Sdag: qgate.SdgGate(),
    gates.T: qgate.TGate(),
    gates.Tdag: qgate.TdgGate(),
}
single_qubit_sgate_mapping: Mapping[Callable[[int], QuantumGate], Gate] = {
    gates.SqrtX: UnitaryGate(
        np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]])
    ),
    gates.SqrtXdag: UnitaryGate(
        np.array([[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]])
    ),
    gates.SqrtY: UnitaryGate(
        np.array([[0.5 + 0.5j, -0.5 - 0.5j], [0.5 + 0.5j, 0.5 + 0.5j]])
    ),
    gates.SqrtYdag: UnitaryGate(
        np.array([[0.5 - 0.5j, 0.5 - 0.5j], [-0.5 + 0.5j, 0.5 - 0.5j]])
    ),
}
def test_convert_single_qubit_gate() -> None:
    for qp_factory, qiskit_gate in single_qubit_gate_mapping.items():
        qp_gate = qp_factory(7)
        converted = convert_gate(qp_gate)
        expected = qiskit_gate
        assert gate_equal(converted, expected)


two_qubit_gate_mapping: Mapping[Callable[[int, int], QuantumGate], Gate] = {
    gates.CNOT: qgate.CXGate(),
    gates.CZ: qgate.CZGate(),
    gates.SWAP: qgate.SwapGate(),
}


def test_convert_two_qubit_gate() -> None:
    for qp_factory, qiskit_gate in two_qubit_gate_mapping.items():
        qp_gate = qp_factory(11, 7)
        converted = convert_gate(qp_gate)
        expected = qiskit_gate
        assert gate_equal(converted, expected)


rotation_gate_mapping: Mapping[Callable[[int, float], QuantumGate], Type[Gate]] = {
    gates.RX: qgate.RXGate,
    gates.RY: qgate.RYGate,
    gates.RZ: qgate.RZGate,
}


def test_convert_rotation_gate() -> None:
    for qp_factory, qiskit_type in rotation_gate_mapping.items():
        qp_gate = qp_factory(7, 0.125)
        converted = convert_gate(qp_gate)
        expected = qiskit_type(0.125)
        assert gate_equal(converted, expected)




def test_convert_u1_gate() -> None:
    lmd1 = 0.125
    converted = convert_gate(gates.U1(lmd = lmd1, target_index = 0))
    expected = qgate.U1Gate(lmd1)
    assert gate_equal(converted, expected) 

def test_convert_u2_gate() -> None:
    phi2, lmd2 = 0.125, -0.125
    converted = convert_gate(gates.U2(phi = phi2, lmd = lmd2, target_index = 0))
    expected = qgate.U2Gate(phi2, lmd2)
    assert gate_equal(converted, expected) 
    
def test_convert_u3_gate() -> None:
    theta3, phi3, lmd3 = 0.125, -0.125, 0.625
    converted = convert_gate(gates.U3(theta = theta3, phi = phi3, lmd = lmd3, target_index = 0))
    expected = qgate.U3Gate(theta3, phi3, lmd3)
    assert gate_equal(converted, expected)






multi_qubit_gate_mapping: Mapping[Callable[..., QuantumGate], Type[Gate]] = {
    gates.Pauli: qgate.PauliGate,
    gates.PauliRotation: qgate.PauliEvolutionGate,
}


# def test_convert_multi_qubit_pauli_gate() -> None:
#     converted = convert_gate(gates.Pauli(target_indices= [0, 2, 5], pauli_ids = [1, 2, 3]))
#     expected = qgate.PauliGate(label = "Pauli")
#     assert gate_equal(converted, expected) 

# def test_convert_multi_qubit_pauli_rotation_gate() -> None:
#     converted = convert_gate(gates.PauliRotation(target_indices= [0, 2, 5], pauli_ids = [1, 2, 3], angle = 0.25))
#     expected = qgate.PauliEvolutionGate(operator = [X, Y, Z], time = 0.125, label = None)

#     assert gate_equal(converted, expected) 




def test_convert_circuit() -> None:
    circuit = QuantumCircuit(3)
    original_gates = [
        gates.X(1),
        gates.H(2),
        gates.CNOT(0, 2),
        gates.RX(0, 0.125),
    ]
    for gate in original_gates:
        circuit.add_gate(gate)

    converted = convert_circuit(circuit)
    assert converted.num_qubits == 3

    expected = QiskitQuantumCircuit(3)
    expected.x(1)
    expected.h(2)
    expected.cnot(0, 2)
    expected.rx(0.125, 0)
    assert circuit_equal(converted, expected)


# def test_convert_bound_parametric_circuit() -> None:
#     circuit = UnboundParametricQuantumCircuit(3)
#     circuit.add_X_gate(1)
#     circuit.add_ParametricRX_gate(0)
#     circuit.add_H_gate(2)
#     circuit.add_ParametricRY_gate(1)
#     circuit.add_CNOT_gate(0, 2)
#     circuit.add_ParametricRZ_gate(2)
#     circuit.add_RX_gate(0, 0.125)
#     circuit.add_Pauli_gate(
#         (0, 1, 2), (1, 2, 3)
#     )  # Expected to be decomposed by default transpiler.

#     params = list(2 * np.pi * np.random.random(3))
#     circuit_bound = circuit.bind_parameters(params)
#     converted = convert_circuit(circuit_bound)
#     assert converted.qubit_count == 3

#     expected = (
#         Circuit()
#         .x(1)
#         .rx(0, params[0])
#         .h(2)
#         .ry(1, params[1])
#         .cnot(0, 2)
#         .rz(2, params[2])
#         .rx(0, 0.125)
#         .x(0)
#         .y(1)
#         .z(2)
#     )
#     assert circuit_equal(converted, expected)


# def test_convert_bound_linear_mapped_parametric_circuit() -> None:
#     circuit = LinearMappedUnboundParametricQuantumCircuit(3)
#     theta, phi, rot = circuit.add_parameters("theta", "phi", "rot")
#     circuit.add_X_gate(1)
#     circuit.add_ParametricRX_gate(0, {theta: 0.5})
#     circuit.add_H_gate(2)
#     circuit.add_ParametricRY_gate(1, phi)
#     circuit.add_CNOT_gate(0, 2)
#     circuit.add_ParametricRZ_gate(2, {theta: -0.5})
#     circuit.add_RX_gate(0, 0.125)
#     circuit.add_ParametricPauliRotation_gate(
#         (0, 1, 2), (1, 2, 3), {rot: 0.5}
#     )  # Expected to be decomposed by default transpiler.

#     params = list(2 * np.pi * np.random.random(3))
#     circuit_bound = circuit.bind_parameters(params)
#     converted = convert_circuit(circuit_bound)
#     assert converted.qubit_count == 3

#     expected = (
#         Circuit()
#         .x(1)
#         .rx(0, params[0] * 0.5)
#         .h(2)
#         .ry(1, params[1])
#         .cnot(0, 2)
#         .rz(2, params[0] * -0.5)
#         .rx(0, 0.125)
#         .h(0)
#         .rx(1, np.pi / 2.0)
#         .cnot(1, 0)
#         .cnot(2, 0)
#         .rz(0, params[2] * 0.5)
#         .cnot(1, 0)
#         .cnot(2, 0)
#         .h(0)
#         .rx(1, -np.pi / 2.0)
#     )
#     assert circuit_equal(converted, expected)

