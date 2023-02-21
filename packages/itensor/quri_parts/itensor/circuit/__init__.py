from collections.abc import Mapping, Sequence

import juliacall
from juliacall import Main as jl
from quri_parts.circuit import (
    LinearMappedUnboundParametricQuantumCircuitBase,
    NonParametricQuantumCircuit,
    QuantumGate,
    UnboundParametricQuantumCircuitBase,
    UnboundParametricQuantumCircuitProtocol,
    gate_names,
)
from quri_parts.circuit.gate_names import (
    SingleQubitGateNameType,
    TwoQubitGateNameType,
    is_gate_name,
    is_single_qubit_gate_name,
    is_two_qubit_gate_name,
)

# For now, we only support gates which is defined [here](https://github.com/ITensor/ITensors.jl/blob/d5ed4061f1e6224d0135fd7690c3be2fbecd0d9d/src/physics/site_types/qubit.jl)

_single_qubit_gate_itensor: Mapping[SingleQubitGateNameType, str] = {
    gate_names.X: "X",
    gate_names.Y: "Y",
    gate_names.Z: "Z",
    gate_names.H: "H",
    gate_names.S: "S",
}

_single_qubit_rotation_gate_itensor: Mapping[SingleQubitGateNameType, str] = {
    gate_names.RX: "Rx",
    gate_names.RY: "Ry",
    gate_names.RZ: "Rz",
}

_two_qubit_gate_itensor: Mapping[TwoQubitGateNameType, str] = {
    gate_names.CNOT: "CNOT",
    gate_names.CZ: "CZ",
    gate_names.SWAP: "SWAP",
}


def convert_circuit(
    circuit: NonParametricQuantumCircuit, s: juliacall.VectorValue
) -> juliacall.VectorValue:
    gate_list: juliacall.VectorValue = jl.gate_list()
    for gate in circuit.gates:
        if not is_gate_name(gate.name):
            raise ValueError(f"Unknown gate name: {gate.name}")

        if is_single_qubit_gate_name(gate.name):
            if gate.name in _single_qubit_gate_itensor:
                gate_list = jl.add_gate(
                    gate_list,
                    _single_qubit_gate_itensor[gate.name],
                    gate.target_indices[0] + 1,
                )
            elif gate.name in _single_qubit_rotation_gate_itensor:
                gate_list = jl.add_gate(
                    gate_list,
                    _single_qubit_rotation_gate_itensor[gate.name],
                    gate.target_indices[0] + 1,
                    gate.params[0],
                )
            else:
                raise ValueError(f"Unknown single qubit gate name: {gate.name}")
        elif is_two_qubit_gate_name(gate.name):
            if gate.name == "SWAP":
                gate_list = jl.add_gate(
                    gate_list,
                    _two_qubit_gate_itensor[gate.name],
                    gate.target_indices[0] + 1,
                    gate.target_indices[1] + 1,
                )
            else:
                gate_list = jl.add_gate(
                    gate_list,
                    _two_qubit_gate_itensor[gate.name],
                    gate.control_indices[0] + 1,
                    gate.target_indices[0] + 1,
                )
        else:
            raise ValueError(f"Unknown gate name: {gate.name}")
    circuit = jl.ops(gate_list, s)
    return circuit
