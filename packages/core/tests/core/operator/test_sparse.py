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

import numpy as np
import pytest
import scipy.sparse as sparse

from quri_parts.core.operator import (
    PAULI_IDENTITY,
    Operator,
    get_sparse_matrix,
    pauli_label,
    zero,
)
from quri_parts.core.operator.sparse import (
    _convert_operator_to_sparse,
    _convert_pauli_label_to_sparse,
    _convert_pauli_map_to_other_format,
    _pauli_map,
)

I = np.eye(2, dtype=np.complex128)  # noqa: E741
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def test_convert_pauli_map_to_other_format() -> None:
    _convert_pauli_map_to_other_format("csc")
    for p in _pauli_map.values():
        assert isinstance(p, sparse.csc_matrix)

    _convert_pauli_map_to_other_format("csr")
    for p in _pauli_map.values():
        assert isinstance(p, sparse.csr_matrix)

    _convert_pauli_map_to_other_format("bsr")
    for p in _pauli_map.values():
        assert isinstance(p, sparse.bsr_matrix)

    _convert_pauli_map_to_other_format("dia")
    for p in _pauli_map.values():
        assert isinstance(p, sparse.dia_matrix)

    _convert_pauli_map_to_other_format("dok")
    for p in _pauli_map.values():
        assert isinstance(p, sparse.dok_matrix)

    _convert_pauli_map_to_other_format("lil")
    for p in _pauli_map.values():
        assert isinstance(p, sparse.lil_matrix)


def test_convert_operator_to_sparse() -> None:
    operator = Operator({pauli_label("X0 Y3"): -3, PAULI_IDENTITY: 8})
    expected_matrix = -3 * reduce(np.kron, [Y, I, I, X]) + 8 * np.eye(16)
    assert np.allclose(_convert_operator_to_sparse(operator).toarray(), expected_matrix)

    operator = zero()
    assert np.allclose(
        _convert_operator_to_sparse(operator, 3).toarray(), np.zeros((1, 1))
    )

    operator = Operator({PAULI_IDENTITY: 8})
    assert np.allclose(
        _convert_operator_to_sparse(operator, 3).toarray(), 8 * np.eye(8)
    )

    with pytest.raises(ValueError):
        _convert_operator_to_sparse(operator)


def test_convert_pauli_label_to_sparse() -> None:
    test_pauli_label = pauli_label("X0 X1 Y3 Z4")
    expected_matrix = reduce(np.kron, [Z, Y, I, X, X])
    assert np.allclose(
        _convert_pauli_label_to_sparse(test_pauli_label).toarray(), expected_matrix
    )

    with pytest.raises(
        AssertionError,
        match=(
            "The specified number of qubits should not be less then the length"
            " of the pauli operator."
        ),
    ):
        _convert_pauli_label_to_sparse(test_pauli_label, 2).toarray()

    test_pauli_label = PAULI_IDENTITY
    assert np.allclose(
        _convert_pauli_label_to_sparse(test_pauli_label, 3).toarray(), np.eye(8)
    )
    with pytest.raises(
        AssertionError,
        match=(
            "n_qubits needs to be specified for PAULI_IDENTITY"
            " to be converted to matrix."
        ),
    ):
        _convert_pauli_label_to_sparse(test_pauli_label)


def test_get_sparse_matrix() -> None:
    test_pauli_label = pauli_label("X0 X1 Y3 Z4")
    expected_matrix = reduce(np.kron, [Z, Y, I, X, X])
    assert np.allclose(get_sparse_matrix(test_pauli_label).toarray(), expected_matrix)
    test_pauli_label = PAULI_IDENTITY
    assert np.allclose(get_sparse_matrix(test_pauli_label, 3).toarray(), np.eye(8))

    operator = Operator({pauli_label("X0 Y3"): -3, PAULI_IDENTITY: 8})
    expected_matrix = -3 * reduce(np.kron, [Y, I, I, X]) + 8 * np.eye(16)
    assert np.allclose(get_sparse_matrix(operator).toarray(), expected_matrix)

    operator = zero()
    assert np.allclose(
        _convert_operator_to_sparse(operator, 3).toarray(), np.zeros((1, 1))
    )

    operator = Operator({PAULI_IDENTITY: 8})
    assert np.allclose(get_sparse_matrix(operator, 3).toarray(), 8 * np.eye(8))
