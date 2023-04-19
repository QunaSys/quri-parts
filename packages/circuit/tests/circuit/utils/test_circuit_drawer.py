import pytest

from quri_parts.circuit import CNOT, RZ, SWAP, PauliRotation, UnitaryMatrix, X
from quri_parts.circuit.utils.circuit_drawer import _generate_gate_aa


def test_generate_gate_aa() -> None:
    expected = ["  ___  ", " | X | ", "-|0  |-", " |___| "]
    assert _generate_gate_aa(X(0), gate_idx=0) == expected

    expected = ["  ___  ", " |RZ | ", "-|2  |-", " |___| "]
    assert _generate_gate_aa(RZ(0, 0.1), gate_idx=2) == expected

    expected = [
        "  ___  ",
        " |CX | ",
        "-|5  |-",
        " |___| ",
        "   |   ",
        "   |   ",
        "   ●   ",
    ]
    assert _generate_gate_aa(CNOT(1, 0), gate_idx=5) == expected

    expected = [
        "       ",
        "  4    ",
        "---x---",
        "   |   ",
        "   |   ",
        "   |   ",
        "---|---",
        "   |   ",
        "   |   ",
        "   |   ",
        "---x---",
        "       ",
    ]
    assert _generate_gate_aa(SWAP(0, 2), gate_idx=4) == expected

    expected = [
        "  ___  ",
        " |PR | ",
        "-|5  |-",
        " |_ _| ",
        "  | |  ",
        "  | |  ",
        "--| |--",
        "  | |  ",
        " _| |_ ",
        " |   | ",
        "-|   |-",
        " |___| ",
    ]
    assert _generate_gate_aa(PauliRotation((0, 2), (0, 1), 0.1), gate_idx=5) == expected

    expected = [
        "  ___  ",
        " |Mat| ",
        "-|6  |-",
        " |   | ",
        " |   | ",
        " |   | ",
        "-|   |-",
        " |___| ",
    ]
    assert (
        _generate_gate_aa(
            UnitaryMatrix(
                target_indices=(0, 1),
                unitary_matrix=[
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            ),
            gate_idx=6,
        )
        == expected
    )

    with pytest.warns(Warning):
        _generate_gate_aa(X(0), gate_idx=1000)
