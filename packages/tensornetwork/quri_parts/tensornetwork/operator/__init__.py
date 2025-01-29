# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from collections.abc import Iterable, Sequence
from typing import Union, Optional

from tensornetwork import Node, NodeCollection, split_node, split_node_full_svd

from quri_parts.core.operator import Operator, PauliLabel
from quri_parts.tensornetwork.circuit import TensorNetworkLayer

_PAULI_OPERATOR_DATA_MAP = {
    0: [[1, 0], [0, 1]],
    1: [[0, 1], [1, 0]],
    2: [[0, -1j], [1j, 0]],
    3: [[1, 0], [0, -1]],
}


def _kron(pauli_list: Sequence[Sequence[Sequence[int]]]):
    """Calculate the tensor product of a list of arrays"""

    prod = np.array(pauli_list[0])
    for u in pauli_list[1:]:
        prod = np.kron(np.array(u), prod)

    return prod


def get_observable_data(pauli_list: Sequence[Sequence[Sequence[int]]], dim=2, n=None):
    """Take a list of matrices that represent pauli operators and
    calculate their tensor-product.
    """

    _obs = _kron(pauli_list)
    if n is None:
        n = len(pauli_list)
    obs_data = np.reshape(_obs, [dim for _ in range(2 * n)])

    return obs_data


def pauli_label_to_array(pauli: PauliLabel) -> Sequence[Sequence[Sequence[int]]]:
    index_list, pauli_id_list = pauli.index_and_pauli_id_list
    data_list = list(map(_PAULI_OPERATOR_DATA_MAP.get, pauli_id_list))
    lq = index_list[0]
    rq = index_list[-1]
    nq = rq - lq + 1
    if not len(index_list) == nq:
        for i in range(lq, rq + 1):
            if i not in index_list:
                data_list.insert(i - lq, _PAULI_OPERATOR_DATA_MAP[0])
    
    array = get_observable_data(data_list, dim=2, n=nq)

    return array

def operator_to_tensor(operator: Union[Operator, PauliLabel], qubit_count: int) -> TensorNetworkLayer:
    """Convert an :class:`~Operator` or a :class:`~PauliLabel` to a
    :class:`~TensorNetworkLayer`.

    Args:
        operator: Input operator
        qubit_count: Number of qubits of that operator
    """
    if isinstance(operator,PauliLabel):
        data = pauli_label_to_array(operator)
    assert isinstance(operator, Operator)

    data = sum(map(lambda p, c : pauli_label_to_array(p)*c, operator.keys(), operator.values()))

    op = Node(data)
    tensor = TensorNetworkLayer(op[:qubit_count], op[qubit_count:], {op})

    return tensor


def mpo_from_tensor(
    operator: TensorNetworkLayer,
    max_bond_dimension: Optional[int] = None,
    max_truncation_err: Optional[float] = None,
    qubits_per_node: int = 1,
) -> TensorNetworkLayer:
    """Convert an :class:`~Operator` or a :class:`~PauliLabel` to a
    tensornetwork :class:`~Node`.

    Args:
        operator: Input operator
        max_bond_dimension: Optional specification of MPO bond-dimension
        max_truncation_error: Optional specification of truncation error tolerance
        qubits_per_node: Number of physical qubits attached to each node, defaults to 1
    """
    qubit_count = len(operator.input_edges)

    if qubits_per_node == 1:
        edge_groups = list(zip(operator.input_edges,operator.output_edges))
    else:
        raise NotImplementedError()
    right_edges = []

    left_edges = []
    
    for eg in edge_groups:
        left_edges.extend(eg)
        right_edges
