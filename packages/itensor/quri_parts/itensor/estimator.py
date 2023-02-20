import os
from collections.abc import Iterable, Mapping
from typing import NamedTuple, Union

import juliacall
from juliacall import Main as jl
from quri_parts.circuit import NonParametricQuantumCircuit, gate_names
from quri_parts.core.estimator import Estimatable, Estimate, QuantumEstimator
from quri_parts.core.operator import Operator, PauliLabel, pauli_name, zero
from quri_parts.core.state import (
    CircuitQuantumState,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
    QuantumStateVector,
)
from typing_extensions import TypeAlias

path = os.getcwd()
library_path = os.path.join(path, "packages/itensor/quri_parts/itensor/library.jl")

jl.seval("using ITensors")
include_statement = 'include("' + library_path + '")'
print(include_statement)
jl.seval(include_statement)


class _Estimate(NamedTuple):
    value: complex
    error: float = 0.0


#: A type alias for state classes supported by Qulacs estimators.
#: Qulacs estimators support both of circuit states and state vectors.
QulacsStateT: TypeAlias = Union[CircuitQuantumState, QuantumStateVector]

#: A type alias for parametric state classes supported by Qulacs estimators.
#: Qulacs estimators support both of circuit states and state vectors.
QulacsParametricStateT: TypeAlias = Union[
    ParametricCircuitQuantumState, ParametricQuantumStateVector
]


from quri_parts.circuit.gate_names import (
    MultiQubitGateNameType,
    SingleQubitGateNameType,
    TwoQubitGateNameType,
    is_gate_name,
    is_multi_qubit_gate_name,
    is_parametric_gate_name,
    is_single_qubit_gate_name,
    is_two_qubit_gate_name,
    is_unitary_matrix_gate_name,
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
            gate_list = jl.add_gate(
                gate_list,
                _two_qubit_gate_itensor[gate.name],
                gate.target_indices[0] + 1,
                gate.target_indices[1] + 1,
            )
        else:
            raise ValueError(f"Unknown gate name: {gate.name}")
    circuit = jl.ops(gate_list, s)
    return circuit


def convert_operator(
    operator: Union[Operator, PauliLabel], s: juliacall.VectorValue
) -> juliacall.AnyValue:
    paulis: Iterable[tuple[PauliLabel, complex]]
    if isinstance(operator, Operator):
        paulis = operator.items()
    else:
        paulis = [(operator, 1)]
    os: juliacall.AnyValue = jl.OpSum()
    for pauli, coef in paulis:
        pauli_gates: juliacall.VectorValue = jl.gate_list()
        for i, p in pauli:
            pauli_gates = jl.add_pauli(pauli_gates, pauli_name(p), i + 1)
        os = jl.add_coef_pauli(os, coef, pauli_gates)
    op: juliacall.AnyValue = jl.MPO(os, s)
    return op


def _estimate(operator: Estimatable, state: QulacsStateT) -> Estimate[complex]:
    if operator == zero():
        return _Estimate(value=0.0)
    qubits = state.qubit_count
    s: juliacall.VectorValue = jl.siteinds("Qubit", qubits)
    psi: juliacall.AnyValue = jl.initState(s, qubits)

    # create ITensor circuit
    circuit = convert_circuit(state.circuit, s)

    # create ITensor operator
    op = convert_operator(operator, s)

    # calculate expectation value
    psi = jl.apply(circuit, psi)
    exp: float = jl.expectation(psi, op)

    return _Estimate(value=exp)


def create_itensor_mps_estimator() -> QuantumEstimator[QulacsStateT]:
    """Returns a :class:`~QuantumEstimator` that uses ITensor MPS simulator
    to calculate expectation values."""

    return _estimate


if __name__ == "__main__":
    from quri_parts.core.operator import pauli_label
    from quri_parts.core.state import ComputationalBasisState

    pauli = pauli_label("Z0 Z2 Z5")
    state = ComputationalBasisState(6, bits=0b110010)
    estimator = create_itensor_mps_estimator()
    estimate = estimator(pauli, state)
    assert estimate.value == -1
    assert estimate.error == 0

    operator = Operator(
        {
            pauli_label("Z0 Z2 Z5"): 0.25,
            pauli_label("Z1 Z2 Z4"): 0.5j,
        }
    )
    state = ComputationalBasisState(6, bits=0b110010)
    estimator = create_itensor_mps_estimator()
    estimate = estimator(operator, state)
    print(estimate.value)
    assert estimate.value == -0.25 + 0.5j
    assert estimate.error == 0
