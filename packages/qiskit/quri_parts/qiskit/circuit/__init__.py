# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping, Sequence
from typing import Callable, Optional, Type

import numpy as np
from quri_parts.circuit import NonParametricQuantumCircuit, QuantumGate, gate_names
from quri_parts.circuit.gate_names import (
    MultiQubitGateNameType,
    ParametricGateNameType,
    SingleQubitGateNameType,
    TwoQubitGateNameType,
    is_gate_name,
    is_multi_qubit_gate_name,
    is_parametric_gate_name,
    is_single_qubit_gate_name,
    is_two_qubit_gate_name,
)
from quri_parts.circuit.transpile import (
    CircuitTranspiler,
    PauliDecomposeTranspiler,
    PauliRotationDecomposeTranspiler,
    SequentialTranspiler,
)

import qiskit.circuit.library as qgate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.extensions import UnitaryGate

# from qiskit.providers import Backend
from qiskit.opflow import X, Y, Z

#: CircuitTranspiler to convert a circuit configuration suitable for Qiskit.
QiskitTranspiler: Callable[[], CircuitTranspiler] = lambda: SequentialTranspiler(
    [PauliDecomposeTranspiler(), PauliRotationDecomposeTranspiler()]
)


_single_qubit_gate_qiskit: Mapping[SingleQubitGateNameType, Type[Gate]] = {
    gate_names.Identity: qgate.IGate,
    gate_names.X: qgate.XGate,
    gate_names.Y: qgate.YGate,
    gate_names.Z: qgate.ZGate,
    gate_names.H: qgate.HGate,
    gate_names.S: qgate.SGate,
    gate_names.Sdag: qgate.SdgGate,
    gate_names.T: qgate.TGate,
    gate_names.Tdag: qgate.TdgGate,
    gate_names.SqrtX: qgate.SXGate,
    gate_names.SqrtXdag: qgate.SXdgGate,
}

_single_qubit_rotation_gate_qiskit: Mapping[SingleQubitGateNameType, Type[Gate]] = {
    gate_names.RX: qgate.RXGate,
    gate_names.RY: qgate.RYGate,
    gate_names.RZ: qgate.RZGate,
}

_two_qubit_gate_qiskit: Mapping[TwoQubitGateNameType, Type[Gate]] = {
    gate_names.CNOT: qgate.CXGate,
    gate_names.CZ: qgate.CZGate,
    gate_names.SWAP: qgate.SwapGate,
}

_multi_qubit_gate_qiskit: Mapping[MultiQubitGateNameType, Type[Gate]] = {
    gate_names.Pauli: qgate.PauliGate,
    gate_names.PauliRotation: qgate.PauliEvolutionGate,
}


_parametric_gate_qiskit: Mapping[ParametricGateNameType, Type[Gate]] = {
    gate_names.ParametricRX: qgate.RXGate,
    gate_names.ParametricRY: qgate.RYGate,
    gate_names.ParametricRZ: qgate.RZGate,
}

_U_gate_qiskit: Mapping[SingleQubitGateNameType, Type[Gate]] = {
    gate_names.U1: qgate.U1Gate,
    gate_names.U2: qgate.U2Gate,
    gate_names.U3: qgate.U3Gate,
}

_special_named_gate_matrix: Mapping[
    SingleQubitGateNameType, Sequence[Sequence[complex]]
] = {
    gate_names.SqrtY: [[0.5 + 0.5j, -0.5 - 0.5j], [0.5 + 0.5j, 0.5 + 0.5j]],
    gate_names.SqrtYdag: [[0.5 - 0.5j, 0.5 - 0.5j], [-0.5 + 0.5j, 0.5 - 0.5j]],
}


def convert_gate(gate: QuantumGate) -> Gate:
    if not is_gate_name(gate.name):
        raise ValueError(f"Unknown gate name: {gate.name}")

    if is_single_qubit_gate_name(gate.name):
        if gate.name in _single_qubit_gate_qiskit:
            return _single_qubit_gate_qiskit[gate.name]()
        elif gate.name in _single_qubit_rotation_gate_qiskit:
            return _single_qubit_rotation_gate_qiskit[gate.name](gate.params[0])
        elif gate.name in _U_gate_qiskit:
            return _U_gate_qiskit[gate.name](*gate.params, label=None)
        elif gate.name in _special_named_gate_matrix:
            s_matrix = _special_named_gate_matrix[gate.name]
            return UnitaryGate(np.array(s_matrix))
        else:
            assert False, "Unreachable"

    elif is_two_qubit_gate_name(gate.name) and gate.name in _two_qubit_gate_qiskit:
        return _two_qubit_gate_qiskit[gate.name]()

    elif is_multi_qubit_gate_name(gate.name) and gate.name in _multi_qubit_gate_qiskit:
        if gate.name == gate_names.Pauli:
            q_gate = qgate.PauliGate(label="Pauli")
            pauli_str = ""
            for p in gate.pauli_ids:
                if p == 1:
                    pauli_str += "X"
                elif p == 2:
                    pauli_str += "Y"
                elif p == 3:
                    pauli_str += "Z"
                q_gate.params = [pauli_str]
            return q_gate
        elif gate.name == gate_names.PauliRotation:
            operator = 1
            for p in gate.pauli_ids:
                if p == 1:
                    operator ^= X
                elif p == 2:
                    operator ^= Y
                elif p == 3:
                    operator ^= Z
                else:
                    raise ValueError("Invalid Pauli index.")
            return qgate.PauliEvolutionGate(operator, time=float(gate.params[0] / 2))

    elif is_parametric_gate_name(gate.name):
        assert False, "Unreachable"
    else:
        assert False, "Unreachable"

    raise NotImplementedError(
        f"Conversion of {gate.name} to qiskit has not been implemented."
    )


def convert_circuit(
    circuit: NonParametricQuantumCircuit,
    transpiler: Optional[CircuitTranspiler] = QiskitTranspiler(),
) -> QuantumCircuit:
    if transpiler is not None:
        circuit = transpiler(circuit)

    qiskit_circuit = QuantumCircuit(circuit.qubit_count)
    for gate in circuit.gates:
        indices = (
            list(gate.control_indices) + list(gate.target_indices)
            if gate.control_indices
            else list(gate.target_indices)
        )
        qiskit_circuit.append(convert_gate(gate), qargs=indices)
    return qiskit_circuit


__all__ = ["QiskitTranspiler", "convert_gate", "convert_circuit"]
