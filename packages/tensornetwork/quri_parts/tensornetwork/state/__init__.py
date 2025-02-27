# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Mapping, Optional, Sequence, Text, Union, Dict, Tuple

import numpy as np
import numpy.typing as npt
import tensornetwork as tn
from tensornetwork import AbstractNode, Edge, Node, NodeCollection, Tensor
from h5py import Group

from quri_parts.core.state import CircuitQuantumState
from quri_parts.tensornetwork.circuit import TensorNetworkLayer, convert_circuit
from quri_parts.tensornetwork.circuit.gates import QuantumGate


class MappedNode(AbstractNode):  # type: ignore
    """This is a convenience class for single-qubits."""

    def __init__(
        self,
        node: Node,
        qubit_index: int,
        qubit_edge_index: int,
        name: Optional[Text] = None,
    ) -> None:
        self.node = node
        self.backend = node.backend
        self.qubit_index = qubit_index
        self.qubit_edge_index = qubit_edge_index  # This may not be 0 in an MPS
        for e in self:
            if e.node1 == self.node:
                e.node1 = self
            if e.node2 == self.node:
                e.node2 = self
        if name is not None:
            self.name = name

    @property
    def dtype(self) -> Tensor:
        return self.node.dtype

    @property
    def qubit_edge(self) -> Edge:
        return self[self.qubit_edge_index]

    def copy(self, conjugate: bool = False) -> "MappedNode":
        """Returns a copy of itself."""
        node_copy = self.node.copy(conjugate)
        mapped_node = MappedNode(
            node_copy,
            self.qubit_index,
            self.qubit_edge_index,
            self.name,
        )
        return mapped_node

    @property
    def edges(self) -> List["Edge"]:
        return self.node.edges

    @edges.setter
    def edges(self, edges: List["Edge"]) -> None:
        self.node.edges = edges

    @property
    def name(self) -> Text:
        return self.node.name

    @name.setter
    def name(self, name) -> None:
        if not isinstance(name, str):
            raise TypeError("Node name should be str type")
        self.node.name = name

    @property
    def axis_names(self) -> List[Text]:
        return self.node.axis_names

    @axis_names.setter
    def axis_names(self, axis_names: List[Text]) -> None:
        self.node.axis_names = axis_names

    def disable(self) -> None:
        if self.node.is_disabled:
            raise ValueError("Node {} is already disabled".format(self.name))
        self.node.is_disabled = True

    def get_tensor(self) -> Tensor:
        return self.node.get_tensor()

    def set_tensor(self, tensor) -> None:
        return self.node.set_tensor(tensor)

    @property
    def shape(self) -> Tuple[Optional[int], ...]:
        return self.node.shape

    @property
    def tensor(self) -> Tensor:
        return self.node.tensor

    @tensor.setter
    def tensor(self, tensor: Tensor) -> None:
        self.node.tensor = tensor

    @classmethod
    def _load_node(cls, _: Group) -> "AbstractNode":
        """load a node based on hdf5 data.

        Args:
          node_data: h5py group that contains the serialized node data

        Returns:
          The loaded node.
        """
        raise NotImplementedError("Loading nodes is not supported for MappedNode")

    def _save_node(self, _: Group) -> None:
        """Abstract method to enable saving nodes to hdf5. Only serializing common
        properties is implemented. Should be overwritten by subclasses.

        Args:
          node_group: h5py group where data is saved
        """
        raise NotImplementedError("Saving nodes is not supported for MappedNode")

    def to_serial_dict(self) -> Dict:
        """Return a serializable dict representing the node.

        Returns: A dict object.
        """
        raise NotImplementedError("Serializing nodes is not supported for MappedNode")

    @classmethod
    def from_serial_dict(cls, _: Dict) -> "AbstractNode":
        """Return a node given a serialized dict representing it.

        Args:
          serial_dict: A python dict representing a serialized node.

        Returns:
          A node.
        """
        raise NotImplementedError("Serializing nodes is not supported for MappedNode")


class TensorNetworkState(NodeCollection):  # type: ignore
    """Tensor network representation of a quantum state.

    This class subclasses :class:`~NodeCollection` and provides output
    edges for the state, each of which represents a qubit
    """

    def __init__(
        self,
        edges: Sequence[Edge],
        container: Union[set[AbstractNode], list[AbstractNode]],
        tensor_map: Sequence[Mapping[int, Union[AbstractNode]]],
    ):
        self.edges = edges
        self.tensor_map = tensor_map
        super().__init__(container)

    def with_gates_applied(self, circuit: TensorNetworkLayer) -> "TensorNetworkState":
        """Returns a new :class:`~TensorNetworkState` with the given
        :class:`~TensorNetworkCircuit` applied."""
        circuit = circuit.copy()
        state = self.copy()

        for e, f in zip(state.edges, circuit.input_edges):
            e ^ f

        node_set = state._container.union(circuit._container)
        tensor_map = list(state.tensor_map) + list(circuit.tensor_map)
        return TensorNetworkState(circuit.output_edges, node_set, tensor_map)

    def copy(self, conjugate: bool = False) -> "TensorNetworkState":
        """Returns a copy of itself."""
        state_node_mapping, state_edge_mapping = tn.copy(
            self._container, conjugate=conjugate
        )
        state_nodes = {state_node_mapping[n] for n in self._container}
        state_edges = [state_edge_mapping[e] for e in self.edges]
        tensor_map = [
            {q: state_node_mapping[n] for q, n in mapping.items()}
            for mapping in self.tensor_map
        ]

        return TensorNetworkState(state_edges, state_nodes, tensor_map)

    def conjugate(self) -> "TensorNetworkState":
        """Returns a conjugated copy of itself."""

        return self.copy(conjugate=True)

    def contract(self, method: str = "greedy") -> "TensorNetworkState":
        """Returns a copy of self after contracting internal tensor network."""
        copy = self.copy()
        if method == "greedy":
            node = tn.contractors.greedy(copy._container, output_edge_order=copy.edges)
        else:
            raise NotImplementedError(
                "The requested contraction algorithms is not available"
            )
        tensor_map = {q: node for q in range(len(copy.edges))}

        return TensorNetworkState(copy.edges, {node}, [tensor_map])


def svd(
    node: AbstractNode,
    output_edge_mapping: Mapping[int, Edge],
    left_mps_edges: Optional[List[Edge]] = None,
    right_mps_edges: Optional[List[Edge]] = None,
    max_bond_dimension: Optional[int] = None,
    max_truncation_err: Optional[float] = None,
) -> tuple[Mapping[int, AbstractNode], Mapping[int, Edge]]:
    """Singular value decomposition used to update the mps."""

    qubit_node_mapping = {}
    right_node = node
    left_edges = [list(output_edge_mapping.values())[0]]
    if left_mps_edges is not None:
        left_edges.extend(left_mps_edges)
    right_edges = list(output_edge_mapping.values())[1:]
    if right_mps_edges is not None:
        right_edges.extend(right_mps_edges)

    qubit_list = list(output_edge_mapping.keys())
    for q in qubit_list[:-1]:
        left_node, right_node, _ = tn.split_node(
            right_node,
            left_edges,
            right_edges,
            max_singular_values=max_bond_dimension,
            max_truncation_err=max_truncation_err,
        )
        qubit_node_mapping[q] = left_node
        left_edges = [right_edges.pop(0)]
        all_edges = left_edges + right_edges
        for e in right_node:
            if e not in all_edges:
                left_edges.append(e)

    qubit_node_mapping[qubit_list[-1]] = right_node

    return qubit_node_mapping, output_edge_mapping


class TensorNetworkStateMPS(TensorNetworkState):
    """Tensor network representation of a quantum state using mps.

    This class subclasses :class:`~TensorNetworkState` and provides a
    method for updating the state based on an applied circuit.
    """

    def __init__(
        self,
        edges: Sequence[Edge],
        container: Union[set[AbstractNode], list[AbstractNode]],
        tensor_map: Sequence[Mapping[int, Union[AbstractNode]]],
        max_bond_dimension: Optional[int] = None,
        max_truncation_err: Optional[float] = None,
    ):
        super().__init__(edges, container, tensor_map)
        self.max_bond_dimension = max_bond_dimension
        self.max_truncation_err = max_truncation_err

    def with_gates_applied(
        self, circuit: TensorNetworkLayer
    ) -> "TensorNetworkStateMPS":
        """Returns a new :class:`~TensorNetworkState` with the given
        :class:`~TensorNetworkCircuit` applied."""
        circuit = circuit.copy()
        state = self.copy()

        for e, f in zip(state.edges, circuit.input_edges):
            e ^ f

        node_set = state._container.union(circuit._container)
        tensor_map = list(state.tensor_map) + list(circuit.tensor_map)
        return TensorNetworkStateMPS(
            circuit.output_edges,
            node_set,
            tensor_map,
            max_bond_dimension=self.max_bond_dimension,
            max_truncation_err=self.max_truncation_err,
        )

    def copy(self, conjugate: bool = False) -> "TensorNetworkStateMPS":
        """Returns a copy of itself."""
        state_node_mapping, state_edge_mapping = tn.copy(
            self._container, conjugate=conjugate
        )
        state_nodes = {state_node_mapping[n] for n in self._container}
        state_edges = [state_edge_mapping[e] for e in self.edges]
        tensor_map = [
            {q: state_node_mapping[n] for q, n in mapping.items()}
            for mapping in self.tensor_map
        ]

        return TensorNetworkStateMPS(
            state_edges,
            state_nodes,
            tensor_map,
            max_bond_dimension=self.max_bond_dimension,
            max_truncation_err=self.max_truncation_err,
        )

    def conjugate(self) -> "TensorNetworkStateMPS":
        """Returns a conjugated copy of itself."""

        return self.copy(conjugate=True)

    def contract(self, method: str = "tebd") -> "TensorNetworkStateMPS":
        """Returns a copy of self after contracting internal tensor network."""
        if method != "tebd":
            return super().contract(method=method)

        copy = self.copy()
        if len(copy.tensor_map) == 1:
            return copy

        tensor_map: dict[int, MappedNode] = dict(copy.tensor_map[0])

        for mapping in copy.tensor_map[1:]:
            mapped_qubits = []
            for q, n in mapping.items():
                if q in mapped_qubits:
                    continue
                if q not in tensor_map:
                    raise ValueError("State qubits and circuit qubits do not match")
                assert isinstance(n, QuantumGate)
                all_qubits = list(n.input_qubit_edge_mapping.keys())
                mapped_qubits.extend(all_qubits)

                # Contract nodes
                output_edges = {}
                node_set = set()
                for qb in all_qubits:
                    node_set.add(tensor_map[qb])
                    node_set.add(n)
                    output_edges[qb] = n.output_qubit_edge_mapping[qb]
                left_mps_edges = [
                    e
                    for e in tensor_map[all_qubits[0]].edges
                    if (e.node1 not in node_set) or (e.node2 not in node_set) and not e.is_dangling()
                ]
                right_mps_edges = [
                    e
                    for e in tensor_map[all_qubits[-1]].edges
                    if (e.node1 not in node_set) or (e.node2 not in node_set) and not e.is_dangling()
                ]
                all_non_contracted_edges = (
                    list(output_edges.values()) + left_mps_edges + right_mps_edges
                )
                contracted_node = tn.contractors.greedy(
                    node_set, all_non_contracted_edges
                )

                # Perform SVD and update tensor map
                mps_node_mapping, mps_edge_mapping = svd(
                    contracted_node,
                    output_edges,
                    left_mps_edges=left_mps_edges,
                    right_mps_edges=right_mps_edges,
                    max_bond_dimension=self.max_bond_dimension,
                    max_truncation_err=self.max_truncation_err,
                )
                for qb in all_qubits:
                    node = mps_node_mapping[qb]
                    index = [
                        i
                        for i in range(len(node.edges))
                        if node[i] == mps_edge_mapping[qb]
                    ][0]
                    tensor_map[qb] = MappedNode(node, qb, index, name=f"MPS q={qb}")
                    # connected_nodes = edge.get_nodes()
                    # for m in connected_nodes:
                    #     if m and m != node:
                    #         edge.disconnect()
                    #         assert isinstance(m, QuantumGate)
                    #         tensor_map[qb].qubit_edge ^ m.input_qubit_edge_mapping[qb]

        return TensorNetworkStateMPS(
            [tensor_map[i].qubit_edge for i in sorted(tensor_map)],
            list(tensor_map.values()),
            [tensor_map],
            max_bond_dimension=self.max_bond_dimension,
            max_truncation_err=self.max_truncation_err,
        )


def get_zero_state(qubit_count: int, backend: str = "numpy") -> TensorNetworkState:
    """Returns the zero state for the given number of qubits."""
    qubits: list[MappedNode] = []
    zero_state_edges: list[Edge] = []
    tensor_map: dict[int, MappedNode] = {}
    for q in range(qubit_count):
        node = Node(np.array([1.0, 0.0], dtype=np.complex128), backend=backend)
        mapped_node = MappedNode(node, q, 0, name=f"|0> q={q}")
        qubits.append(mapped_node)
        zero_state_edges.append(mapped_node[0])
        tensor_map[q] = mapped_node
    return TensorNetworkState(zero_state_edges, qubits, [tensor_map])


def convert_state(
    state: CircuitQuantumState, backend: str = "numpy"
) -> TensorNetworkState:
    qubit_count = state.qubit_count
    zero_state = get_zero_state(qubit_count, backend=backend)
    state_circuit = convert_circuit(state.circuit, backend=backend)
    tn_state = zero_state.with_gates_applied(state_circuit)
    return tn_state


def convert_state_mps(
    state: CircuitQuantumState,
    backend: str = "numpy",
    max_bond_dimension: Optional[int] = None,
    max_truncation_err: Optional[float] = None,
) -> TensorNetworkStateMPS:
    tensornetwork_state = convert_state(state, backend=backend)
    return TensorNetworkStateMPS(
        tensornetwork_state.edges,
        tensornetwork_state._container,
        tensornetwork_state.tensor_map,
        max_bond_dimension=max_bond_dimension,
        max_truncation_err=max_truncation_err,
    )
