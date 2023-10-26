from collections.abc import Sequence

from quri_parts.circuit import NonParametricQuantumCircuit, QuantumCircuit
from quri_parts.circuit.gate import QuantumGate
from quri_parts.circuit.gate_names import (
    CLIFFORD_GATE_NAMES,
    SINGLE_QUBIT_GATE_NAMES,
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    TOFFOLI,
    U1,
    U2,
    U3,
    GateNameType,
    CliffordGateNameType,
    H,
    Identity,
    Pauli,
    PauliRotation,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    T,
    Tdag,
    UnitaryMatrix,
    X,
    Y,
    Z,
)

from .transpiler import (
    CircuitTranspilerProtocol,
    SequentialTranspiler,
    CircuitTranspiler,
)
from .multi_pauli_decomposer import (
    PauliDecomposerTranspiler,
    PauliRotationDecomposerTranspiler,
)
import gate_kind_decomposer as dc
from .unitary_matrix_decomposer import (
    SingleQubitUnitaryMatrix2RYRZTranspiler,
    TwoQubitUnitarymatrixKAKTranspiler,
)
from .fuse import Rotation2NamedTranspiler
from .identity_manipulation import IdentityEliminationTranspiler


# TODO Create systematically
_equiv_clifford_table = {
    H: [[S, SqrtX, S]],
    X: [[Y, Z], [SqrtX, SqrtX], [SqrtXdag, SqrtXdag], [H, Z, H], [H, S, S, H]],
    Y: [
        [Z, X],
        # [SqrtY, SqrtY],
        # [SqrtYdag, SqrtYdag],
        [H, X, H],
        [S, S, X],
        [Z, H, Z, H],
        [S, S, H, S, S, H],
    ],
    Z: [[X, Y], [S, S], [Sdag, Sdag], [X, H, X, H], [X, S, S, X]],
    SqrtX: [[Sdag, H, Sdag], [S, Z, H, Z, S], [S, S, S, H, S, S, S]],
    SqrtXdag: [[S, H, S], [Z, SqrtX, Z], [S, S, SqrtX, S, S]],
    SqrtY: [[Z, H], [S, S, H], [Sdag, SqrtX, S], [Z, S, SqrtX, S], [S, S, S, SqrtX, S]],
    SqrtYdag: [
        [H, Z],
        [H, S, S],
        [S, SqrtX, Sdag],
        [S, SqrtX, Z, S],
        [S, SqrtX, S, S, S],
    ],
    S: [[Z, Sdag], [Sdag, Sdag, Sdag]],
    Sdag: [[Z, S], [S, S, S]],
}


class CliffordConversionTranspiler(CircuitTranspilerProtocol):
    def __init__(self, target_gateset: Sequence[CliffordGateNameType]):
        self._gateset = set(target_gateset)
        # TODO check gate type

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        ret = []

        for gate in circuit.gates:
            if gate.name not in CLIFFORD_GATE_NAMES & SINGLE_QUBIT_GATE_NAMES:
                ret.append(gate)
                continue

            if gate.name not in _equiv_clifford_table:
                ret.append(gate)
                continue

            for candidate in _equiv_clifford_table[gate.name]:
                if set(candidate) <= self._gateset:
                    ret.extend(
                        [
                            QuantumGate(name=name, target_indices=gate.target_indices)
                            for name in candidate
                        ]
                    )
                    break
            else:
                ret.append(gate)

        return QuantumCircuit(circuit.qubit_count, gates=ret)


class RotationConversionTranspiler(CircuitTranspilerProtocol):
    def __init__(
        self,
        target_rotation: Sequence[GateNameType],
        target_clifford: Sequence[CliffordGateNameType],
    ):
        self._target_rotation = set(target_rotation)
        self._target_clifford = set(target_clifford)
        # TODO check gate type

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        if {RX, RY, RZ} & self._gateset == {RX, RY, RZ}:
            ...
        elif {RX, RY} & self._gateset == {RX, RY}:
            # Rz -> Rx(pi/2) Ry(-theta) Rx(-pi/2)
            ...
        elif {RY, RZ} & self._gateset == {RY, RZ}:
            # Rx -> Rz(pi/2) Ry(theta) Rz(-pi/2)
            ...
        elif {RX, RZ} & self._gateset == {RX, RZ}:
            # Ry -> Rx(pi/2) Ry(-theta) Rx(-pi/2)
            ...
        elif RX in self._gateset:
            # H(YZ) + Rx
            ...
        elif RY in self._gateset:
            # H(XZ) + Ry
            ...
        elif RZ in self._gateset:
            # H(XY) + Rz
            ...
        else:
            # Rotation2Named
            # RXRY -> RZ -> HST
            ...

        ...


class GateSetConversionTranspiler(CircuitTranspilerProtocol):
    def __init__(self, target_gateset: Sequence[GateNameType]):
        self._gateset = set(target_gateset)

    def _add_decomposer(
        self, name_decomp_dict: dict[GateNameType, CircuitTranspiler]
    ) -> list[CircuitTranspiler]:
        ts = []
        for name, trans in name_decomp_dict.items():
            if name not in self._gateset:
                ts.append(trans)
        return ts

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        ts = []

        if Identity not in self._gateset:
            ts.append(IdentityEliminationTranspiler())
            # ts.append(dc.Identity2RZTranspiler())

        # TODO consider to check if the circuit contains unitary matrix gate actually.
        complex_gate_table = {
            Pauli: PauliDecomposerTranspiler(),
            PauliRotation: PauliRotationDecomposerTranspiler(),
            UnitaryMatrix: SequentialTranspiler(
                [
                    SingleQubitUnitaryMatrix2RYRZTranspiler(),
                    TwoQubitUnitarymatrixKAKTranspiler(),
                ]
            ),
            TOFFOLI: dc.TOFFOLI2HTTdagCNOTTranspiler(),
            U1: dc.U1ToRZTranspiler(),
            U2: dc.U2ToRXRZTranspiler(),
            U3: dc.U3ToRXRZTranspiler(),
        }
        ts.extend(self._add_decomposer(complex_gate_table))

        if SWAP not in self._gateset:
            ts.append(dc.SWAP2CNOTTranspiler())
        if CZ not in self._gateset:
            ts.append(dc.CZ2CNOTHTranspiler())
        if CNOT not in self._gateset:
            ts.append(dc.CNOT2CZHTranspiler())
        if CNOT not in self._gateset and CZ not in self._gateset:
            raise ValueError(
                "2 qubit gates cannot be converted into the given gateset."
            )

        if self._gateset & CLIFFORD_GATE_NAMES & SINGLE_QUBIT_GATE_NAMES:
            ts.append(Rotation2NamedTranspiler())

        # TODO Clifford table conversion
        ts.append(
            CliffordConversionTranspiler(
                self._gateset & CLIFFORD_GATE_NAMES & SINGLE_QUBIT_GATE_NAMES
            )
        )

        single_qubit_clifford_table = {
            H: dc.H2RXRYTranspiler(),
            X: dc.X2RXTranspiler(),
            Y: dc.Y2RYTranspiler(),
            Z: dc.Z2RZTranspiler(),
            SqrtX: dc.SqrtX2RXTranspiler(),
            SqrtXdag: dc.SqrtXdag2RXTranspiler(),
            SqrtY: dc.SqrtY2RYTranspiler(),
            SqrtYdag: dc.SqrtYdag2RYTranspiler(),
            S: dc.S2RZTranspiler(),
            Sdag: dc.Sdag2RZTranspiler(),
            T: dc.T2RZTranspiler(),
            Tdag: dc.Tdag2RZTranspiler(),
        }
        ts.extend(self._add_decomposer(single_qubit_clifford_table))

        ts.extend(
            RotationConversionTranspiler(
                target_rotation=self._gateset & {RX, RY, RZ},
                target_clifford=self._gateset
                & CLIFFORD_GATE_NAMES
                & SINGLE_QUBIT_GATE_NAMES,
            )
        )

        # if {RX, RY, RZ} & self._gateset == {RX, RY, RZ}:
        #     ...
        # elif {RX, RY} & self._gateset == {RX, RY}:
        #     # Rz -> Rx(pi/2) Ry(-theta) Rx(-pi/2)
        #     ...
        # elif {RY, RZ} & self._gateset == {RY, RZ}:
        #     # Rx -> Rz(pi/2) Ry(theta) Rz(-pi/2)
        #     ...
        # elif {RX, RZ} & self._gateset == {RX, RZ}:
        #     # Ry -> Rx(pi/2) Ry(-theta) Rx(-pi/2)
        #     ...
        # elif RX in self._gateset:
        #     # H(YZ) + Rx
        #     ...
        # elif RY in self._gateset:
        #     # H(XZ) + Ry
        #     ...
        # elif RZ in self._gateset:
        #     # H(XY) + Rz
        #     ...
        # else:
        #     # Rotation2Named
        #     # RXRY -> RZ -> HST
        #     ...

        return SequentialTranspiler(ts)(circuit)
