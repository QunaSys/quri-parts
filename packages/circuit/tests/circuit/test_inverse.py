from typing import Callable, Sequence, Union

import numpy as np
from typing_extensions import TypeAlias

from quri_parts.circuit import (
    CNOT,
    RX,
    RY,
    RZ,
    U1,
    U2,
    U3,
    H,
    PauliRotation,
    QuantumCircuit,
    QuantumGate,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    T,
    Tdag,
    inverse_circuit,
    inverse_gate,
)
from quri_parts.circuit.gates import RXFactory, RYFactory, RZFactory, U1Factory

GateFactory: TypeAlias = Union[RXFactory, RYFactory, RZFactory, U1Factory]

_inverse_pairs: list[
    tuple[Callable[[int], QuantumGate], Callable[[int], QuantumGate]]
] = [(S, Sdag), (SqrtX, SqrtXdag), (SqrtY, SqrtYdag), (T, Tdag)]
_single_param_rotations: list[Callable[[int, float], QuantumGate]] = [RX, RY, RZ, U1]


_inverse_circuit_pairs: list[tuple[QuantumCircuit, QuantumCircuit]] = [
    (QuantumCircuit(8, gates=gatelist), QuantumCircuit(8, gates=inv_gatelist))
    for gatelist, inv_gatelist in zip(
        [
            [S(0), T(0)],
            [SqrtX(0), SqrtY(0)],
            [H(0), CNOT(0, 1)],
            [
                PauliRotation((0, 1, 2), (3, 2, 1), np.pi / 7),
                PauliRotation((2, 3, 1), (3, 2, 2), -np.pi / 11),
            ],
        ],
        [
            [Tdag(0), Sdag(0)],
            [SqrtYdag(0), SqrtXdag(0)],
            [CNOT(0, 1), H(0)],
            [
                PauliRotation((2, 3, 1), (3, 2, 2), np.pi / 11),
                PauliRotation((0, 1, 2), (3, 2, 1), -np.pi / 7),
            ],
        ],
    )
]
_single_param_rotation_circuit: Callable[
    [Sequence[GateFactory], Sequence[float]], QuantumCircuit
] = lambda gates, theta: QuantumCircuit(
    8, gates=[gate(0, param) for gate, param in zip(gates, theta)]
)

_multi_idx_param_rotation_circuit: Callable[
    [Sequence[Sequence[int]], Sequence[Sequence[int]], Sequence[float]], QuantumCircuit
] = lambda target_indices, pauli_ids, thetas: QuantumCircuit(
    8,
    gates=[
        PauliRotation(target_idx, pauli_id, param)
        for target_idx, pauli_id, param in zip(target_indices, pauli_ids, thetas)
    ],
)


def _assert_inverse_gates(a: QuantumGate, b: QuantumGate) -> None:
    assert a == inverse_gate(b)
    assert inverse_gate(a) == b


def _assert_inverse_circuits(a: QuantumCircuit, b: QuantumCircuit) -> None:
    assert a == inverse_circuit(b)
    assert inverse_circuit(a) == b


def test_inverse_gate() -> None:
    for lf, rf in _inverse_pairs:
        _assert_inverse_gates(lf(7), rf(7))

    for f in _single_param_rotations:
        theta = np.random.rand()
        _assert_inverse_gates(f(7, theta), f(7, -theta))

    theta, phi, lmd = np.random.rand(3)
    _assert_inverse_gates(U2(7, phi, lmd), U2(7, -phi, -lmd))
    _assert_inverse_gates(U3(7, theta, phi, lmd), U3(7, -theta, -phi, -lmd))

    target_idx = np.random.choice(range(8), 4, replace=False).tolist()
    pauli_ids = [np.random.randint(1, 4) for _ in range(4)]
    angle = np.random.random()
    _assert_inverse_gates(
        PauliRotation(target_idx, pauli_ids, angle),
        PauliRotation(target_idx, pauli_ids, -angle),
    )


def test_inverse_circuit() -> None:
    for lc, rc in _inverse_circuit_pairs:
        _assert_inverse_circuits(lc, rc)

    trials = 10
    depth = 5
    for _ in range(trials):
        theta = np.random.rand(depth)

        # circuit with single qubit rotation gates
        gateset = np.array(
            [np.random.choice(np.array([RX, RY, RZ, U1])) for _ in range(depth)]
        )
        lc = _single_param_rotation_circuit(gateset.tolist(), theta.tolist())
        theta_inv = (-theta).tolist()
        theta_inv.reverse()
        gateset_inv = gateset.tolist()
        gateset_inv.reverse()
        rc = _single_param_rotation_circuit(gateset_inv, theta_inv)
        _assert_inverse_circuits(lc, rc)

        # circuit with multi pauli rotation gates
        target_indices = [
            np.random.choice(range(8), 4, replace=False).tolist() for _ in range(depth)
        ]
        pauli_ids = [np.random.randint(1, 4, size=4).tolist() for _ in range(depth)]
        lc = _multi_idx_param_rotation_circuit(
            target_indices, pauli_ids, theta.tolist()
        )
        inv_target_indices = target_indices[::-1]
        inv_pauli_ids = pauli_ids[::-1]
        rc = _multi_idx_param_rotation_circuit(
            inv_target_indices, inv_pauli_ids, theta_inv
        )
        _assert_inverse_circuits(lc, rc)
