# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import reduce
from typing import Optional, Union

import numpy as np
import scipy.sparse as sparse

from .operator import PAULI_IDENTITY, Operator, PauliLabel

_sparse_pauli_x = sparse.csc_matrix([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_sparse_pauli_y = sparse.csc_matrix([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_sparse_pauli_z = sparse.csc_matrix([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

_pauli_map = {
    1: _sparse_pauli_x,
    2: _sparse_pauli_y,
    3: _sparse_pauli_z,
}


def _convert_pauli_label_to_sparse(
    single_pauli_label: PauliLabel, n_qubits: Optional[int] = None
) -> sparse.csc_matrix:
    """Convert :class:`~PauliLabel` into scipy sparse matrix."""
    if n_qubits is None:
        assert single_pauli_label != PAULI_IDENTITY, (
            "n_qubits needs to be specified for PAULI_IDENTITY"
            " to be converted to matrix."
        )
        n_qubits = max(single_pauli_label.qubit_indices()) + 1

    single_pauli_list = [
        sparse.identity(2, np.complex128, format="csc") for _ in range(n_qubits)
    ]

    if single_pauli_label != PAULI_IDENTITY:
        assert n_qubits >= max(single_pauli_label.qubit_indices()) + 1
        for bit, pauli in zip(*single_pauli_label.index_and_pauli_id_list):
            single_pauli_list[n_qubits - bit - 1] = _pauli_map[pauli]

    return reduce(lambda o1, o2: sparse.kron(o1, o2, "csc"), single_pauli_list)


def _convert_operator_to_sparse(
    operator: Operator, n_qubits: Optional[int] = None
) -> sparse.csc_matrix:
    """Convert :class:`~Operator` into scipy sparse matrix."""
    if n_qubits is None:
        n_qubits = max(
            [max(op.qubit_indices()) + 1 for op in operator if op != PAULI_IDENTITY]
        )

    return sum(
        [
            coeff * _convert_pauli_label_to_sparse(op, n_qubits)
            for op, coeff in operator.items()
        ]
    )


def get_sparse_matrix(
    operator: Union[PauliLabel, Operator], n_qubits: Optional[int] = None
) -> sparse.csc_matrix:
    """Convert :class:`~PauliLabel` and :class:`~Operator` into scipy sparse
    matrix."""
    if isinstance(operator, PauliLabel):
        return _convert_pauli_label_to_sparse(operator, n_qubits)
    elif isinstance(operator, Operator):
        return _convert_operator_to_sparse(operator, n_qubits)
    else:
        assert False, "operator should be either a PauliLabel or an Operator object."
