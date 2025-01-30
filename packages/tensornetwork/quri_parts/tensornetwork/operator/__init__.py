# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable, Sequence
from typing import Optional, Union

import numpy as np
from tensornetwork import Node, NodeCollection, split_node, split_node_full_svd

from quri_parts.core.operator import Operator, PauliLabel, pauli_label
from quri_parts.tensornetwork.circuit import TensorNetworkLayer, TensorNetworkOperator

_PAULI_OPERATOR_DATA_MAP = {
    0: [[1, 0], [0, 1]],
    1: [[0, 1], [1, 0]],
    2: [[0, 1j], [-1j, 0]],
    3: [[1, 0], [0, -1]],
}


def _kron(pauli_list: Sequence[Sequence[Sequence[int]]]):
    """Calculate the tensor product of a list of arrays."""

    prod = np.array(pauli_list[0])
    for u in pauli_list[1:]:
        prod = np.kron(np.array(u), prod)

    return prod


def get_observable_data(pauli_list: Sequence[Sequence[Sequence[int]]], dim=2, n=None):
    """Take a list of matrices that represent pauli operators and calculate
    their tensor-product."""

    _obs = _kron(pauli_list)
    if n is None:
        n = len(pauli_list)
    obs_data = np.reshape(_obs, [dim for _ in range(2 * n)])

    return obs_data


def pauli_label_to_array(
    pauli: PauliLabel, index_list: Optional[Sequence[int]]
) -> Sequence[Sequence[Sequence[int]]]:
    """Convert a :class:`~PauliLabel` to a numpy array.

    If this function is used to convert a :class:`~PauliLabel` belonging
    to an :class:`~Operator`, then an index_list, describing all of the
    indices acted on by the :class:`~Operator`, should be passed as an
    argument. The returned array will then be padded with identity
    operators where needed.
    """
    if index_list is None:
        index_list, pauli_id_list = pauli.index_and_pauli_id_list
        this_index_list = index_list
    else:
        this_index_list, pauli_id_list = pauli.index_and_pauli_id_list
    data_list = list(map(_PAULI_OPERATOR_DATA_MAP.get, pauli_id_list))
    nq = len(index_list)
    if this_index_list != index_list:
        for i, q in enumerate(index_list):
            if q not in this_index_list:
                data_list.insert(i, _PAULI_OPERATOR_DATA_MAP[0])

    array = get_observable_data(data_list, dim=2, n=nq)

    return array


def operator_to_tensor(operator: Union[Operator, PauliLabel]) -> TensorNetworkOperator:
    """Convert an :class:`~Operator` or a :class:`~PauliLabel` to a
    :class:`~TensorNetworkLayer`.

    Args:
        operator: Input operator
        qubit_count: Number of qubits of that operator
    Outputs:
        Operator as a :class:`~TensorNetworkLayer`
    """
    if isinstance(operator, PauliLabel):
        qubit_count = len(operator.qubit_indices())
        data = pauli_label_to_array(operator)
    else:
        assert isinstance(operator, Operator)
        indices_list = [{i for i in p.qubit_indices()} for p in operator.keys()]
        all_indices = set()
        for i in indices_list:
            all_indices.update(i)
        qubit_count = len(all_indices)
        data = sum(
            map(
                lambda p, c: pauli_label_to_array(p, all_indices) * c,
                operator.keys(),
                operator.values(),
            )
        )

    op = Node(data)
    tensor = TensorNetworkOperator(
        all_indices, op[:qubit_count], op[qubit_count:], {op}
    )

    return tensor


def tensor_to_mpo(
    operator: TensorNetworkOperator,
    max_bond_dimension: Optional[int] = None,
    max_truncation_err: Optional[float] = None,
    qubits_per_node: int = 1,
) -> TensorNetworkOperator:
    """Perform the singular value decomposition on an operator represented
    using :class:`~TensorNetworkLayer` and return the matrix product operator
    (MPO) as a :class:`~TensorNetworkLayer`

    Args:
        operator: Input operator
        max_bond_dimension: Optional specification of MPO bond-dimension
        max_truncation_error: Optional specification of truncation error tolerance
        qubits_per_node: Number of physical qubits attached to each node, defaults to 1
    Outputs:
        MPO as a :class:`~TensorNetworkLayer`
    """
    assert len(operator._container) == 1

    operator_copy = operator.copy()
    if qubits_per_node == 1:
        edge_groups = list(zip(operator_copy.input_edges, operator_copy.output_edges))
    else:
        raise NotImplementedError()
    right_edges = [e for eg in edge_groups for e in eg]

    nodes = set()
    right_node = operator_copy._container.pop()  # Assumes only one node in the operator
    for eg in edge_groups[:-1]:
        left_edges = [e for e in eg]
        right_edges = [e for e in right_edges if e not in left_edges]
        left_node, right_node, _ = split_node(
            right_node,
            left_edges,
            right_edges,
            max_singular_values=max_bond_dimension,
            max_truncation_err=max_truncation_err,
        )
        nodes.add(left_node)
    nodes.add(right_node)

    return TensorNetworkOperator(
        operator_copy.index_list,
        operator_copy.input_edges,
        operator_copy.output_edges,
        nodes,
    )
