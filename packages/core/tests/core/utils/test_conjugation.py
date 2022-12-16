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
