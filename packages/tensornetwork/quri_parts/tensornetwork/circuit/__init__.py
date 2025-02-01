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
from typing import Callable, Optional, Sequence

from quri_parts.circuit import QuantumCircuit, gate_names
from quri_parts.circuit.gate_names import (
    SingleQubitGateNameType,
    ThreeQubitGateNameType,
    TwoQubitGateNameType,
    is_gate_name,
    is_single_qubit_gate_name,
    is_three_qubit_gate_name,
    is_two_qubit_gate_name,
)
from quri_parts.circuit.transpile import (
    CircuitTranspiler,
    PauliDecomposeTranspiler,
    PauliRotationDecomposeTranspiler,
    SequentialTranspiler,
)

import tensornetwork as tn
from quri_parts.tensornetwork.circuit import gates
from quri_parts.tensornetwork.circuit.gates import QuantumGate
from tensornetwork import AbstractNode, Edge, NodeCollection

_single_qubit_gate_tensornetwork: Mapping[SingleQubitGateNameType, str] = {
    gate_names.Identity: gates.I,
    gate_names.X: gates.X,
    gate_names.Y: gates.Y,
    gate_names.Z: gates.Z,
    gate_names.H: gates.H,
    gate_names.S: gates.S,
    gate_names.Sdag: gates.Sdag,
    gate_names.SqrtX: gates.SqrtX,
    gate_names.SqrtXdag: gates.SqrtXdag,
    gate_names.SqrtY: gates.SqrtY,
    gate_names.SqrtYdag: gates.SqrtYdag,
    gate_names.T: gates.T,
    gate_names.Tdag: gates.Tdag,
}

_single_qubit_rotation_gate_tensornetwork: Mapping[SingleQubitGateNameType, str] = {
    gate_names.RX: gates.Rx,
    gate_names.RY: gates.Ry,
    gate_names.RZ: gates.Rz,
    gate_names.U1: gates.U1,
    gate_names.U2: gates.U2,
    gate_names.U3: gates.U3,
}

_two_qubit_gate_tensornetwork: Mapping[TwoQubitGateNameType, str] = {
    gate_names.CNOT: gates.CNOT,
    gate_names.CZ: gates.CZ,
    gate_names.SWAP: gates.SWAP,
}

_three_qubit_gate_tensornetwork: Mapping[ThreeQubitGateNameType, str] = {
    gate_names.TOFFOLI: gates.Toffoli,
}

#: CircuitTranspiler to convert a circit configuration suitable for tensornetwork.
TensorNetworkTranspiler: Callable[[], CircuitTranspiler] = lambda: SequentialTranspiler(
    [PauliDecomposeTranspiler(), PauliRotationDecomposeTranspiler()]
)


class TensorNetworkLayer(NodeCollection):
    """Tensor network representation of a quantum circuit and operators.

    This class subclasses :class:`~NodeCollection` and provides input
    and output edges for the circuit/operator, each of which represents
    a qubit.
    """

    def __init__(
        self,
        input_edges: Sequence[Edge],
        output_edges: Sequence[Edge],
        container: set[AbstractNode] | list[AbstractNode],
    ):
        self.input_edges = input_edges
        self.output_edges = output_edges
        super().__init__(container)

    def copy(self) -> "TensorNetworkLayer":
        """Returns a copy of itself."""
        circuit_node_mapping, circuit_edge_mapping = tn.copy(
            self._container, conjugate=False
        )
        circuit_nodes = {circuit_node_mapping[n] for n in self._container}
        circuit_input_edges = [circuit_edge_mapping[e] for e in self.input_edges]
        circuit_output_edges = [circuit_edge_mapping[e] for e in self.output_edges]

        return TensorNetworkLayer(
            circuit_input_edges, circuit_output_edges, circuit_nodes
        )


def convert_circuit(
    circuit: QuantumCircuit,
    transpiler: Optional[CircuitTranspiler] = TensorNetworkTranspiler(),
) -> NodeCollection:
    """Convert an :class:`~ImmutableQuantumCircuit` to a tensornetwork
    NodeCollection.

    Args:
        circuit: the quantum circuit to convert to a node collection
        transpiler: optional transpiler to use
    """
    qubit_count = circuit.qubit_count
    in_out_map: Sequence[dict[str, Optional[Edge]]] = [
        {"in": None, "out": None} for _ in range(qubit_count)
    ]

    def connect_gate(node: QuantumGate, qubits: Sequence[int], qubit_count: int):
        for i, q in enumerate(qubits):
            if in_out_map[q]["in"]:
                in_out_map[q]["out"] ^ node[i]
                in_out_map[q]["out"] = node[qubit_count + i]
            else:
                assert in_out_map[q]["out"] is None
                in_out_map[q]["in"] = node[i]
                in_out_map[q]["out"] = node[qubit_count + i]

    if transpiler is not None:
        circuit = transpiler(circuit)

    node_collection = set()
    for gate in circuit.gates:
        if not is_gate_name(gate.name):
            raise ValueError(f"Unknown gate name: {gate.name}")

        if is_single_qubit_gate_name(gate.name):
            if gate.name in _single_qubit_gate_tensornetwork:
                node = _single_qubit_gate_tensornetwork[gate.name]()
                node_collection.add(node)
            elif gate.name in _single_qubit_rotation_gate_tensornetwork:
                if len(gate.params) == 1:
                    node = _single_qubit_rotation_gate_tensornetwork[gate.name](
                        gate.params[0]
                    )
                else:
                    raise ValueError("Invalid number of parameters.")
                node_collection.add(node)
            else:
                raise ValueError(f"{gate.name} gate is not supported.")
            connect_gate(node, gate.target_indices, 1)
        if is_two_qubit_gate_name(gate.name):
            if gate.name in _two_qubit_gate_tensornetwork:
                node = _two_qubit_gate_tensornetwork[gate.name]()
                node_collection.add(node)
            else:
                raise ValueError(f"{gate.name} gate is not supported.")
            if gate.name == "SWAP":
                connect_gate(node, gate.target_indices, 2)
            else:
                indices = gate.control_indices + gate.target_indices
                connect_gate(node, indices, 2)
        if is_three_qubit_gate_name(gate.name):
            if gate.name in _three_qubit_gate_tensornetwork:
                node = _three_qubit_gate_tensornetwork[gate.name]()
                node_collection.add(node)
            else:
                raise ValueError(f"{gate.name} gate is not supported.")
            indices = gate.control_indices + gate.target_indices
            connect_gate(node, indices, 3)

    for m in in_out_map:
        assert m["in"] is not None and m["out"] is not None

    input_edges = [m["in"] for m in in_out_map]
    output_edges = [m["out"] for m in in_out_map]
    return TensorNetworkLayer(input_edges, output_edges, node_collection)
