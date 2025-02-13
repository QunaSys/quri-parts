# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence

import numpy as np
from quri_parts.core.state import GeneralCircuitQuantumState

import tensornetwork as tn
from quri_parts.tensornetwork.circuit import TensorNetworkLayer, convert_circuit
from tensornetwork import AbstractNode, Edge, NodeCollection


class TensorNetworkState(NodeCollection):
    """Tensor network representation of a quantum state.

    This class subclasses :class:`~NodeCollection` and provides output
    edges for the state, each of which represents a qubit
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
        state_node_mapping, state_edge_mapping = tn.copy(
            self._container, conjugate=False
        )
        state_nodes = {state_node_mapping[n] for n in self._container}
        state_edges = [state_edge_mapping[e] for e in self.edges]

        return TensorNetworkState(state_edges, state_nodes)

    def conjugate(self) -> "TensorNetworkState":
        """Returns a conjugated copy of itself."""
        state_node_mapping, state_edge_mapping = tn.copy(
            self._container, conjugate=True
        )
        state_nodes = {state_node_mapping[n] for n in self._container}
        state_edges = [state_edge_mapping[e] for e in self.edges]

        return TensorNetworkState(state_edges, state_nodes)

    def contract(self, method: str = "greedy") -> "TensorNetworkState":
        """Returns a copy of self after contracting internal tensor network."""
        copy = self.copy()
        node = tn.contractors.greedy(copy._container, output_edge_order=copy.edges)

        return TensorNetworkState(copy.edges, {node})


def get_zero_state(qubit_count: int) -> TensorNetworkState:
    """Returns the zero state for the given number of qubits."""
    qubits = [tn.Node(np.array([1.0, 0.0])) for _ in range(qubit_count)]
    zero_state_edges = [q[0] for q in qubits]
    return TensorNetworkState(zero_state_edges, qubits)


def convert_state(state: GeneralCircuitQuantumState) -> TensorNetworkState:
    qubit_count = state.qubit_count
    zero_state = get_zero_state(qubit_count)
    state_circuit = convert_circuit(state.circuit)
    return zero_state.with_gates_applied(state_circuit)
