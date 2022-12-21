# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import (
    CNOT,
    CZ,
    SWAP,
    H,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    X,
    Y,
    Z,
)
from quri_parts.core.operator import pauli_label
from quri_parts.core.utils.conjugation import conjugation


def test_conjugation() -> None:
    orig_pauli = pauli_label("X0")
    assert conjugation(X(0), orig_pauli) == (pauli_label("X0"), 1.0)
    assert conjugation(Y(0), orig_pauli) == (pauli_label("X0"), -1.0)
    assert conjugation(Z(0), orig_pauli) == (pauli_label("X0"), -1.0)
    assert conjugation(H(0), orig_pauli) == (pauli_label("Z0"), 1.0)
    assert conjugation(S(0), orig_pauli) == (pauli_label("Y0"), 1.0)
    assert conjugation(Sdag(0), orig_pauli) == (pauli_label("Y0"), -1.0)
    assert conjugation(SqrtX(0), orig_pauli) == (pauli_label("X0"), 1.0)
    assert conjugation(SqrtXdag(0), orig_pauli) == (pauli_label("X0"), 1.0)
    assert conjugation(SqrtY(0), orig_pauli) == (pauli_label("Z0"), -1.0)
    assert conjugation(SqrtYdag(0), orig_pauli) == (pauli_label("Z0"), 1.0)
    assert conjugation(CNOT(0, 1), orig_pauli) == (pauli_label("X0 X1"), 1.0)
    assert conjugation(CNOT(1, 0), orig_pauli) == (pauli_label("X0"), 1.0)
    assert conjugation(CZ(0, 1), orig_pauli) == (pauli_label("X0 Z1"), 1.0)
    assert conjugation(CZ(1, 0), orig_pauli) == (pauli_label("X0 Z1"), 1.0)
    assert conjugation(SWAP(0, 1), orig_pauli) == (pauli_label("X1"), 1.0)
    assert conjugation(SWAP(1, 0), orig_pauli) == (pauli_label("X1"), 1.0)

    orig_pauli = pauli_label("Y0")
    assert conjugation(X(0), orig_pauli) == (pauli_label("Y0"), -1.0)
    assert conjugation(Y(0), orig_pauli) == (pauli_label("Y0"), 1.0)
    assert conjugation(Z(0), orig_pauli) == (pauli_label("Y0"), -1.0)
    assert conjugation(H(0), orig_pauli) == (pauli_label("Y0"), -1.0)
    assert conjugation(S(0), orig_pauli) == (pauli_label("X0"), -1.0)
    assert conjugation(Sdag(0), orig_pauli) == (pauli_label("X0"), 1.0)
    assert conjugation(SqrtX(0), orig_pauli) == (pauli_label("Z0"), 1.0)
    assert conjugation(SqrtXdag(0), orig_pauli) == (pauli_label("Z0"), -1.0)
    assert conjugation(SqrtY(0), orig_pauli) == (pauli_label("Y0"), 1.0)
    assert conjugation(SqrtYdag(0), orig_pauli) == (pauli_label("Y0"), 1.0)
    assert conjugation(CNOT(0, 1), orig_pauli) == (pauli_label("Y0 X1"), 1.0)
    assert conjugation(CNOT(1, 0), orig_pauli) == (pauli_label("Y0 Z1"), 1.0)
    assert conjugation(CZ(0, 1), orig_pauli) == (pauli_label("Y0 Z1"), 1.0)
    assert conjugation(CZ(1, 0), orig_pauli) == (pauli_label("Y0 Z1"), 1.0)
    assert conjugation(SWAP(0, 1), orig_pauli) == (pauli_label("Y1"), 1.0)
    assert conjugation(SWAP(1, 0), orig_pauli) == (pauli_label("Y1"), 1.0)

    orig_pauli = pauli_label("Z0")
    assert conjugation(X(0), orig_pauli) == (pauli_label("Z0"), -1.0)
    assert conjugation(Y(0), orig_pauli) == (pauli_label("Z0"), -1.0)
    assert conjugation(Z(0), orig_pauli) == (pauli_label("Z0"), 1.0)
    assert conjugation(H(0), orig_pauli) == (pauli_label("X0"), 1.0)
    assert conjugation(S(0), orig_pauli) == (pauli_label("Z0"), 1.0)
    assert conjugation(Sdag(0), orig_pauli) == (pauli_label("Z0"), 1.0)
    assert conjugation(SqrtX(0), orig_pauli) == (pauli_label("Y0"), -1.0)
    assert conjugation(SqrtXdag(0), orig_pauli) == (pauli_label("Y0"), 1.0)
    assert conjugation(SqrtY(0), orig_pauli) == (pauli_label("X0"), 1.0)
    assert conjugation(SqrtYdag(0), orig_pauli) == (pauli_label("X0"), -1.0)
    assert conjugation(CNOT(0, 1), orig_pauli) == (pauli_label("Z0"), 1.0)
    assert conjugation(CNOT(1, 0), orig_pauli) == (pauli_label("Z0 Z1"), 1.0)
    assert conjugation(CZ(0, 1), orig_pauli) == (pauli_label("Z0"), 1.0)
    assert conjugation(CZ(1, 0), orig_pauli) == (pauli_label("Z0"), 1.0)
    assert conjugation(SWAP(0, 1), orig_pauli) == (pauli_label("Z1"), 1.0)
    assert conjugation(SWAP(1, 0), orig_pauli) == (pauli_label("Z1"), 1.0)
