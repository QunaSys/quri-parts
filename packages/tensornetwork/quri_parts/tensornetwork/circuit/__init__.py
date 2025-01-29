# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensornetwork as tn

from collections.abc import Mapping
from typing import Callable, Optional, Sequence

from tensornetwork import NodeCollection, Edge, AbstractNode

from quri_parts.circuit import ImmutableQuantumCircuit, gate_names
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
from quri_parts.tensornetwork.circuit import gates

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

#: CircuitTranspiler to convert a circit configuration suitable for ITensor.
TensorNetworkTranspiler: Callable[[], CircuitTranspiler] = lambda: SequentialTranspiler(
    [PauliDecomposeTranspiler(), PauliRotationDecomposeTranspiler()]
)


class TensorNetworkLayer(NodeCollection):
    """Tensor network representation of a quantum circuit and operators.

    This class subclasses :class:`~NodeCollection` and provides input and output edges for the circuit/operator, each of which represents a qubit.
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
        circuit_node_mapping, circuit_edge_mapping = tn.copy(self, conjugate=False)
        circuit_nodes = {circuit_node_mapping[n] for n in self._container}
        circuit_input_edges = [circuit_edge_mapping[e] for e in self.input_edges]
        circuit_output_edges = [circuit_edge_mapping[e] for e in self.output_edges]

        return TensorNetworkLayer(
            circuit_input_edges, circuit_output_edges, circuit_nodes
        )


class TensorNetworkState(NodeCollection):
    """Tensor network representation of a quantum state.

    This class subclasses :class:`~NodeCollection` and provides output edges for the state, each of which represents a qubit
    """

    def __init__(
        self, edges: Sequence[Edge], container: set[AbstractNode] | list[AbstractNode]
    ):
        self.edges = edges
        super().__init__(container)

    def with_gates_applied(self, circuit: TensorNetworkLayer) -> "TensorNetworkState":
        """Returns a new :class:`~TensorNetworkState` with the given
        :class:`~TensorNetworkCircuit` applied."""
        circuit = circuit.copy()
        state = self.copy()

        for e, f in zip(state.edges, circuit.input_edges):
            e ^ f

        node_set = state._container.union(circuit._container)
        return TensorNetworkState(circuit.output_edges, node_set)

    def copy(self) -> "TensorNetworkState":
        """Returns a copy of itself."""
        state_node_mapping, state_edge_mapping = tn.copy(self, conjugate=False)
        state_nodes = {state_node_mapping[n] for n in self._container}
        state_edges = [state_edge_mapping[e] for e in self.edges]

        return TensorNetworkState(state_edges, state_nodes)

    def conjugate(self) -> "TensorNetworkState":
        """Returns a conjugated copy of itself."""
        state_node_mapping, state_edge_mapping = tn.copy(self, conjugate=True)
        state_nodes = {state_node_mapping[n] for n in self._container}
        state_edges = [state_edge_mapping[e] for e in self.edges]

        return TensorNetworkState(state_edges, state_nodes)


def convert_state() -> TensorNetworkState:
    ...


def convert_circuit(
    circuit: ImmutableQuantumCircuit,
    transpiler: Optional[CircuitTranspiler] = TensorNetworkTranspiler(),
) -> NodeCollection:
    """Convert an :class:`~ImmutableQuantumCircuit` to a tensornetwork
    NodeCollection.

    Args:
        circuit: the quantum circuit to convert to a node collection
        transpiler: optional transpiler to use
    """
    if transpiler is not None:
        circuit = transpiler(circuit)

    node_collection = NodeCollection()
    for gate in circuit.gates:
        if not is_gate_name(gate.name):
            raise ValueError(f"Unknown gate name: {gate.name}")

        if is_single_qubit_gate_name(gate.name):
            if gate.name in _single_qubit_gate_tensornetwork:
                gate_list = jl.add_single_qubit_gate(
                    gate_list,
                    _single_qubit_gate_tensornetwork[gate.name],
                    gate.target_indices[0] + 1,
                )
            elif gate.name in _single_qubit_rotation_gate_tensornetwork:
                if len(gate.params) == 1:
                    gate_list = jl.add_single_qubit_rotation_gate(
                        gate_list,
                        _single_qubit_rotation_gate_tensornetwork[gate.name],
                        gate.target_indices[0] + 1,
                        gate.params[0],
                    )
                elif len(gate.params) == 2:
                    gate_list = jl.add_single_qubit_rotation_gate(
                        gate_list,
                        _single_qubit_rotation_gate_tensornetwork[gate.name],
                        gate.target_indices[0] + 1,
                        gate.params[0],
                        gate.params[1],
                    )
                elif len(gate.params) == 3:
                    gate_list = jl.add_single_qubit_rotation_gate(
                        gate_list,
                        _single_qubit_rotation_gate_tensornetwork[gate.name],
                        gate.target_indices[0] + 1,
                        gate.params[0],
                        gate.params[1],
                        gate.params[2],
                    )
                else:
                    raise ValueError("Invalid number of parameters.")
            else:
                raise ValueError(f"{gate.name} gate is not supported.")
        elif is_two_qubit_gate_name(gate.name):
            if gate.name == "SWAP":
                gate_list = jl.add_two_qubit_gate(
                    gate_list,
                    _two_qubit_gate_tensornetwork[gate.name],
                    gate.target_indices[0] + 1,
                    gate.target_indices[1] + 1,
                )
            else:
                gate_list = jl.add_two_qubit_gate(
                    gate_list,
                    _two_qubit_gate_tensornetwork[gate.name],
                    gate.control_indices[0] + 1,
                    gate.target_indices[0] + 1,
                )
        elif is_three_qubit_gate_name(gate.name):
            gate_list = jl.add_three_qubit_gate(
                gate_list,
                _three_qubit_gate_tensornetwork[gate.name],
                gate.control_indices[0] + 1,
                gate.control_indices[1] + 1,
                gate.target_indices[0] + 1,
            )
        else:
            raise ValueError(f"{gate.name} gate is not supported.")
    circuit = jl.ops(gate_list, qubit_sites)
    return circuit
