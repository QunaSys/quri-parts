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
from typing import Callable, Type, cast

import qulacs
from typing_extensions import assert_never

from quri_parts.circuit import (
    LinearMappedUnboundParametricQuantumCircuitBase,
    NonParametricQuantumCircuit,
    QuantumGate,
    UnboundParametricQuantumCircuitBase,
    UnboundParametricQuantumCircuitProtocol,
    gate_names,
)
from quri_parts.circuit.gate_names import (
    MultiQubitGateNameType,
    SingleQubitGateNameType,
    ThreeQubitGateNameType,
    TwoQubitGateNameType,
    is_gate_name,
    is_multi_qubit_gate_name,
    is_parametric_gate_name,
    is_single_qubit_gate_name,
    is_three_qubit_gate_name,
    is_two_qubit_gate_name,
    is_unitary_matrix_gate_name,
)

_single_qubit_gate_qulacs: Mapping[
    SingleQubitGateNameType, Type[qulacs.QuantumGateBase]
] = {
    gate_names.Identity: qulacs.gate.Identity,  # type: ignore
    gate_names.X: qulacs.gate.X,  # type: ignore
    gate_names.Y: qulacs.gate.Y,  # type: ignore
    gate_names.Z: qulacs.gate.Z,  # type: ignore
    gate_names.H: qulacs.gate.H,  # type: ignore
    gate_names.S: qulacs.gate.S,  # type: ignore
    gate_names.Sdag: qulacs.gate.Sdag,  # type: ignore
    gate_names.SqrtX: qulacs.gate.sqrtX,  # type: ignore
    gate_names.SqrtXdag: qulacs.gate.sqrtXdag,  # type: ignore
    gate_names.SqrtY: qulacs.gate.sqrtY,  # type: ignore
    gate_names.SqrtYdag: qulacs.gate.sqrtYdag,  # type: ignore
    gate_names.T: qulacs.gate.T,  # type: ignore
    gate_names.Tdag: qulacs.gate.Tdag,  # type: ignore
    gate_names.U1: qulacs.gate.U1,  # type: ignore
    gate_names.U2: qulacs.gate.U2,  # type: ignore
    gate_names.U3: qulacs.gate.U3,  # type: ignore
}

_single_qubit_reverse_rotation_gate_qulacs: Mapping[
    SingleQubitGateNameType, Type[qulacs.QuantumGateBase]
] = {
    gate_names.RX: qulacs.gate.RX,  # type: ignore
    gate_names.RY: qulacs.gate.RY,  # type: ignore
    gate_names.RZ: qulacs.gate.RZ,  # type: ignore
}

_two_qubit_gate_qulacs: Mapping[TwoQubitGateNameType, Type[qulacs.QuantumGateBase]] = {
    gate_names.CNOT: qulacs.gate.CNOT,  # type: ignore
    gate_names.CZ: qulacs.gate.CZ,  # type: ignore
    gate_names.SWAP: qulacs.gate.SWAP,  # type: ignore
}

_three_qubit_gate_qulacs: Mapping[
    ThreeQubitGateNameType, Type[qulacs.QuantumGateBase]
] = {
    gate_names.TOFFOLI: qulacs.gate.TOFFOLI,  # type: ignore
}

_multi_pauli_gate_qulacs: Mapping[
    MultiQubitGateNameType, Type[qulacs.QuantumGateBase]
] = {
    gate_names.Pauli: qulacs.gate.Pauli,  # type: ignore
    gate_names.PauliRotation: qulacs.gate.PauliRotation,  # type: ignore
}

_parametric_gate_qulacs = {
    gate_names.ParametricRX: qulacs.gate.ParametricRX,
    gate_names.ParametricRY: qulacs.gate.ParametricRY,
    gate_names.ParametricRZ: qulacs.gate.ParametricRZ,
    gate_names.ParametricPauliRotation: qulacs.gate.ParametricPauliRotation,
}


def convert_gate(
    gate: QuantumGate,
) -> qulacs.QuantumGateBase:
    if not is_gate_name(gate.name):
        raise ValueError(f"Unknown gate name: {gate.name}")

    if is_single_qubit_gate_name(gate.name):
        if gate.name in _single_qubit_gate_qulacs:
            return _single_qubit_gate_qulacs[gate.name](
                *gate.target_indices, *gate.params
            )
        elif gate.name in _single_qubit_reverse_rotation_gate_qulacs:
            neg_params = (-p for p in gate.params)
            return _single_qubit_reverse_rotation_gate_qulacs[gate.name](
                *gate.target_indices, *neg_params
            )
    elif is_two_qubit_gate_name(gate.name):
        return _two_qubit_gate_qulacs[gate.name](
            *gate.control_indices, *gate.target_indices
        )
    elif is_three_qubit_gate_name(gate.name):
        return _three_qubit_gate_qulacs[gate.name](
            *gate.control_indices, *gate.target_indices
        )
    elif is_multi_qubit_gate_name(gate.name):
        neg_params = (-p for p in gate.params)
        return _multi_pauli_gate_qulacs[gate.name](
            gate.target_indices, gate.pauli_ids, *neg_params  # type: ignore
        )
    elif is_unitary_matrix_gate_name(gate.name):
        return qulacs.gate.DenseMatrix(gate.target_indices, gate.unitary_matrix)  # type: ignore  # noqa: E501
    elif is_parametric_gate_name(gate.name):
        raise ValueError("Parametric gates are not supported")
    else:
        # It seems that currently assert_never does not work here
        # assert_never(gate.name)
        assert False, "Unreachable"

    return cast(qulacs.QuantumGateBase, None)


def convert_circuit(circuit: NonParametricQuantumCircuit) -> qulacs.QuantumCircuit:
    qulacs_circuit = qulacs.QuantumCircuit(circuit.qubit_count)
    for gate in circuit.gates:
        qulacs_circuit.add_gate(convert_gate(gate))
    return qulacs_circuit


def convert_parametric_circuit(
    circuit: UnboundParametricQuantumCircuitProtocol,
) -> tuple[
    qulacs.ParametricQuantumCircuit, Callable[[Sequence[float]], Sequence[float]]
]:
    param_circuit: UnboundParametricQuantumCircuitBase
    param_mapper: Callable[[Sequence[float]], Sequence[float]]
    if isinstance(circuit, LinearMappedUnboundParametricQuantumCircuitBase):
        param_mapping = circuit.param_mapping
        param_circuit = circuit.primitive_circuit()
        orig_param_mapper = param_mapping.seq_mapper

        def param_mapper(s: Sequence[float]) -> Sequence[float]:
            return tuple(-p for p in orig_param_mapper(s))

    elif isinstance(circuit, UnboundParametricQuantumCircuitBase):
        param_circuit = circuit

        def param_mapper(s: Sequence[float]) -> Sequence[float]:
            return tuple(-p for p in s)

    else:
        raise ValueError(f"Unsupported parametric circuit type: {type(circuit)}")

    qulacs_circuit = qulacs.ParametricQuantumCircuit(circuit.qubit_count)
    for gate, _ in param_circuit._gates:
        if is_parametric_gate_name(gate.name):
            if gate.name == gate_names.ParametricRX:
                qulacs_circuit.add_parametric_RX_gate(gate.target_indices[0], 0)
            elif gate.name == gate_names.ParametricRY:
                qulacs_circuit.add_parametric_RY_gate(gate.target_indices[0], 0)
            elif gate.name == gate_names.ParametricRZ:
                qulacs_circuit.add_parametric_RZ_gate(gate.target_indices[0], 0)
            elif gate.name == gate_names.ParametricPauliRotation:
                qulacs_circuit.add_parametric_multi_Pauli_rotation_gate(
                    gate.target_indices, gate.pauli_ids, 0  # type: ignore
                )
            else:
                assert_never(gate.name)
        else:
            assert isinstance(
                gate, QuantumGate
            ), f"gate should be a QuantumGate. actual={gate}"
            qulacs_circuit.add_gate(convert_gate(gate))

    return qulacs_circuit, param_mapper


__all__ = ["convert_gate", "convert_circuit", "convert_parametric_circuit"]
