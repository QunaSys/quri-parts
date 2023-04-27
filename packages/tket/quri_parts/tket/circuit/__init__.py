from collections.abc import Mapping
from typing import Callable, Union, cast

from numpy import array, pi
from pytket import Circuit, OpType  # type: ignore
from pytket.circuit import Unitary1qBox  # type: ignore

from quri_parts.circuit import NonParametricQuantumCircuit, QuantumGate, gate_names
from quri_parts.circuit.gate_names import (
    SingleQubitGateNameType,
    ThreeQubitGateNameType,
    TwoQubitGateNameType,
    is_parametric_gate_name,
    is_single_qubit_gate_name,
    is_three_qubit_gate_name,
    is_two_qubit_gate_name,
    is_unitary_matrix_gate_name,
)

_single_qubit_gate_tket: Mapping[SingleQubitGateNameType, OpType] = {
    gate_names.Identity: OpType.noop,
    gate_names.X: OpType.X,
    gate_names.Y: OpType.Y,
    gate_names.Z: OpType.Z,
    gate_names.H: OpType.H,
    gate_names.S: OpType.S,
    gate_names.Sdag: OpType.Sdg,
    gate_names.SqrtX: OpType.SX,
    gate_names.SqrtXdag: OpType.SXdg,
    gate_names.T: OpType.T,
    gate_names.Tdag: OpType.Tdg,
}

_single_qubit_rotation_gate_tket: Mapping[
    SingleQubitGateNameType,
    Union[
        Callable[[float], OpType],
        Callable[[float, float], OpType],
        Callable[[float, float, float], OpType],
    ],
] = {
    gate_names.U1: OpType.U1,
    gate_names.U2: OpType.U2,
    gate_names.U3: OpType.U3,
    gate_names.RX: OpType.Rx,
    gate_names.RY: OpType.Ry,
    gate_names.RZ: OpType.Rz,
}

_two_qubit_gate_tket: Mapping[TwoQubitGateNameType, OpType] = {
    gate_names.CNOT: OpType.CX,
    gate_names.CZ: OpType.CZ,
    gate_names.SWAP: OpType.SWAP,
}

_three_qubit_gate_tket: Mapping[ThreeQubitGateNameType, OpType] = {
    gate_names.TOFFOLI: OpType.CCX,
}


def convert_gate(
    gate: QuantumGate,
) -> OpType:
    if is_single_qubit_gate_name(gate.name):
        if gate.name in _single_qubit_gate_tket:
            return _single_qubit_gate_tket[gate.name]

        else:
            return _single_qubit_rotation_gate_tket[gate.name]

    elif is_two_qubit_gate_name(gate.name):
        return _two_qubit_gate_tket[gate.name]

    elif is_three_qubit_gate_name(gate.name):
        return _three_qubit_gate_tket[gate.name]

    elif is_unitary_matrix_gate_name(gate.name):
        return cast(OpType, Unitary1qBox(gate.unitary_matrix))

    elif is_parametric_gate_name(gate.name):
        raise ValueError("Parametric gates are not supported")
    else:
        raise NotImplementedError(
            f"{gate.name} conversion to Cirq gate not implemented"
        )


def convert_circuit(circuit: NonParametricQuantumCircuit) -> Circuit:
    tket_circuit = Circuit(circuit.qubit_count)
    for gate in circuit.gates:
        if gate.name in _single_qubit_gate_tket:
            target_qubit = gate.target_indices
            tket_circuit.add_gate(convert_gate(gate), target_qubit)
            continue
        if gate.name in _single_qubit_rotation_gate_tket:
            target_qubit = gate.target_indices
            params = array(gate.params) / pi
            tket_circuit.add_gate(convert_gate(gate), params, target_qubit)
            continue
        if gate.name in _two_qubit_gate_tket:
            if gate.name == gate_names.SWAP:
                target_qubit = gate.target_indices
                tket_circuit.add_gate(convert_gate(gate), target_qubit)
                continue
            else:
                target_qubit = tuple(gate.target_indices)
                control_qubit = tuple(gate.control_indices)
                tket_circuit.add_gate(convert_gate(gate), control_qubit + target_qubit)
                continue
        if gate.name in _three_qubit_gate_tket:
            target_qubit = tuple(gate.target_indices)
            control_qubit = tuple(gate.control_indices)
            tket_circuit.add_gate(convert_gate(gate), control_qubit + target_qubit)
            continue
        if gate.name == "UnitaryMatrix":
            target_qubit = gate.target_indices
            tket_circuit.add_unitary1qbox(convert_gate(gate), target_qubit[0])
            continue
        else:
            raise ValueError("This gate is not supported.")

    return tket_circuit


__all__ = ["convert_gate", "convert_circuit"]
