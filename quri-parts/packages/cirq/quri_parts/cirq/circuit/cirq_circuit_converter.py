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

from cirq.circuits.circuit import Circuit
from cirq.ops.common_gates import CNOT, CZ, H, Rx, Ry, Rz, S, T
from cirq.ops.identity import I
from cirq.ops.pauli_gates import X, Y, Z
from cirq.ops.raw_types import Gate
from cirq.ops.swap_gates import SWAP
from cirq.ops.three_qubit_gates import CCX
from cirq.protocols.unitary_protocol import unitary

from quri_parts.circuit import (
    ImmutableQuantumCircuit,
    QuantumCircuit,
    QuantumGate,
    UnitaryMatrix,
    gate_names,
)
from quri_parts.circuit.gate_names import (
    SingleQubitGateNameType,
    ThreeQubitGateNameType,
    TwoQubitGateNameType,
)

_single_qubit_gate_quri_parts: Mapping[Gate, SingleQubitGateNameType] = {
    I: gate_names.Identity,
    X: gate_names.X,
    Y: gate_names.Y,
    Z: gate_names.Z,
    H: gate_names.H,
    S: gate_names.S,
    S**-1: gate_names.Sdag,
    X**0.5: gate_names.SqrtX,
    X**-0.5: gate_names.SqrtXdag,
    Y**0.5: gate_names.SqrtY,
    Y**-0.5: gate_names.SqrtYdag,
    T: gate_names.T,
    T**-1: gate_names.Tdag,
}

_two_qubit_gate_quri_parts: Mapping[Gate, TwoQubitGateNameType] = {
    CNOT: gate_names.CNOT,
    CZ: gate_names.CZ,
    SWAP: gate_names.SWAP,
}

_three_qubit_gate_quri_parts: Mapping[Gate, ThreeQubitGateNameType] = {
    CCX: gate_names.TOFFOLI,
}


def circuit_from_cirq(cirq_circuit: Circuit) -> ImmutableQuantumCircuit:
    """Converts a :class:`cirq.Circuit` to
    :class:`~ImmutableQuantumCircuit`."""
    qubit_count = (
        max([getattr(qubit, 'x', 0) for qubit in cirq_circuit.all_qubits()]) + 1
    )
    circuit = QuantumCircuit(qubit_count)

    for operation in cirq_circuit.all_operations():
        gate = operation.gate
        if gate is not None and gate in _single_qubit_gate_quri_parts:
            circuit.add_gate(
                QuantumGate(
                    name=(
                        _single_qubit_gate_quri_parts[gate]
                        if gate is not None
                        else ''
                    ),
                    target_indices=(getattr(operation.qubits[0], 'x', 0),),
                )
            )
        elif gate in [CNOT, CZ]:
            circuit.add_gate(
                QuantumGate(
                    name=_two_qubit_gate_quri_parts[gate],
                    target_indices=(getattr(operation.qubits[1], 'x', 0),),
                    control_indices=(getattr(operation.qubits[0], 'x', 0),),
                )
            )
        elif gate == SWAP:
            circuit.add_gate(
                QuantumGate(
                    name=_two_qubit_gate_quri_parts[gate],
                    target_indices=(
                        getattr(operation.qubits[0], 'x', 0),
                        getattr(operation.qubits[1], 'x', 0),
                    ),
                )
            )
        elif isinstance(gate, (Rx, Ry, Rz)):
            circuit.add_gate(
                QuantumGate(
                    name=str(operation.gate)[:2].upper(),
                    target_indices=(getattr(operation.qubits[0], 'x', 0),),
                    params=(gate._rads,),
                )
            )
        elif gate is not None and gate in _three_qubit_gate_quri_parts:
            circuit.add_gate(
                QuantumGate(
                    name=_three_qubit_gate_quri_parts[gate] if gate is not None else '',
                    target_indices=(getattr(operation.qubits[2], 'x', 0),),
                    control_indices=(
                        getattr(operation.qubits[0], 'x', 0),
                        getattr(operation.qubits[1], 'x', 0),
                    ),
                )
            )
        else:
            circuit.add_gate(
                UnitaryMatrix(
                    target_indices=[getattr(i, 'x', 0) for i in operation.qubits],
                    unitary_matrix=unitary(gate).tolist(),
                )
            )
    return circuit
