from collections.abc import Sequence
from math import pi

import gate_kind_decomposer as dc

from quri_parts.circuit import NonParametricQuantumCircuit, QuantumCircuit
from quri_parts.circuit import gate as gf
from quri_parts.circuit.gate import QuantumGate
from quri_parts.circuit.gate_names import (
    CLIFFORD_GATE_NAMES,
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    SINGLE_QUBIT_GATE_NAMES,
    SWAP,
    TOFFOLI,
    U1,
    U2,
    U3,
    CliffordGateNameType,
    GateNameType,
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

from .fuse import FuseRotationTranspiler, Rotation2NamedTranspiler
from .identity_manipulation import IdentityEliminationTranspiler
from .multi_pauli_decomposer import (
    PauliDecomposerTranspiler,
    PauliRotationDecomposerTranspiler,
)
from .transpiler import (
    CircuitTranspiler,
    CircuitTranspilerProtocol,
    GateKindDecomposer,
    SequentialTranspiler,
)
from .unitary_matrix_decomposer import (
    SingleQubitUnitaryMatrix2RYRZTranspiler,
    TwoQubitUnitarymatrixKAKTranspiler,
)

# TODO Generate systematically
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


class RZ2RXRYTranspiler(GateKindDecomposer):
    @property
    def target_gate_names(self) -> Sequence[str]:
        return [RZ]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        qubit = gate.target_indices[0]
        theta = gate.params[0]
        return [
            gf.RX(qubit, pi / 2.0),
            gf.RY(qubit, -theta),
            gf.RX(qubit, -pi / 2.0),
        ]


class RY2RXRZTranspiler(GateKindDecomposer):
    @property
    def target_gate_names(self) -> Sequence[str]:
        return [RY]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        qubit = gate.target_indices[0]
        theta = gate.params[0]
        return [
            gf.RX(qubit, pi / 2.0),
            gf.RZ(qubit, theta),
            gf.RX(qubit, -pi / 2.0),
        ]


class RX2RYRZTranspiler(GateKindDecomposer):
    @property
    def target_gate_names(self) -> Sequence[str]:
        return [RZ]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        qubit = gate.target_indices[0]
        theta = gate.params[0]
        return [
            gf.RZ(qubit, pi / 2.0),
            gf.RY(qubit, theta),
            gf.RZ(qubit, -pi / 2.0),
        ]


class RX2RZHTranspiler(GateKindDecomposer):
    @property
    def target_gate_names(self) -> Sequence[str]:
        return [RX]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        qubit = gate.target_indices[0]
        theta = gate.params[0]
        return [
            gf.H(qubit),
            gf.RZ(qubit, theta),
            gf.H(qubit),
        ]


class RY2RZHTranspiler(GateKindDecomposer):
    @property
    def target_gate_names(self) -> Sequence[str]:
        return [RY]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        qubit = gate.target_indices[0]
        theta = gate.params[0]
        return [
            gf.RZ(qubit, -pi / 2.0),
            gf.H(qubit),
            gf.RZ(qubit, theta),
            gf.H(qubit),
            gf.RZ(qubit, pi / 2.0),
        ]


class IdentityTranspiler(CircuitTranspilerProtocol):
    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        return circuit


class RotationConversionTranspiler(CircuitTranspilerProtocol):
    def __init__(
        self,
        target_rotation: Sequence[GateNameType],
        target_clifford: Sequence[CliffordGateNameType],
    ):
        self._target_rotation = set(target_rotation)
        self._target_clifford = set(target_clifford)
        self._gateset = self._target_rotation | self._target_clifford
        self._decomposer = self._construct_decomposer()
        # TODO check gate type

    def _construct_decomposer(self) -> CircuitTranspiler:
        rot_to_trans_map = {
            frozenset({RX, RY, RZ}): IdentityTranspiler(),
            frozenset({RX, RY}): RZ2RXRYTranspiler(),
            frozenset({RY, RZ}): RX2RYRZTranspiler(),
            frozenset({RX, RZ}): RY2RXRZTranspiler(),
        }

        if H in self._target_clifford:
            rot_to_trans_map[frozenset({RZ})] = SequentialTranspiler(
                [RX2RZHTranspiler(), RY2RZHTranspiler()]
            )
        elif SqrtX in self._target_clifford:
            rot_to_trans_map[frozenset({RZ})] = SequentialTranspiler(
                [dc.RX2RZSqrtXTranspiler(), dc.RY2RZSqrtXTranspiler()]
            )

        # TODO support {RX}, {RY}, and {}

        return rot_to_trans_map.get(
            frozenset(self._target_rotation), IdentityTranspiler()
        )

    def _validate(self, circuit: NonParametricQuantumCircuit) -> None:
        for gate in circuit.gates:
            if gate.name not in self._gateset:
                raise ValueError(f"{gate} cannot be converted into the target gateset.")

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        tr_circuit = self._decomposer(circuit)
        self._validate(tr_circuit)
        return tr_circuit


class GateSetConversionTranspiler(CircuitTranspilerProtocol):
    def __init__(self, target_gateset: Sequence[GateNameType]):
        self._gateset = set(target_gateset)
        self._target_clifford = (
            self._gateset & CLIFFORD_GATE_NAMES & SINGLE_QUBIT_GATE_NAMES
        )
        self._target_rotation = self._gateset & {RX, RY, RZ}
        self._decomposer = self._construct_decomposer()

    def _construct_decomposer(self) -> CircuitTranspiler:
        ts = []

        ts.extend(self._construct_complex_gateset_decomposer())
        ts.extend(self._construct_two_qubit_gate_decomposer())

        if self._target_clifford:
            ts.extend(
                [
                    Rotation2NamedTranspiler(),
                    CliffordConversionTranspiler(self._target_clifford),
                ]
            )

        if Identity not in self._gateset:
            ts.append(IdentityEliminationTranspiler())

        ts.extend(self._construct_clifford_to_rotation_decomposer())
        ts.append(FuseRotationTranspiler())
        ts.extend(
            RotationConversionTranspiler(
                target_rotation=self._target_rotation,
                target_clifford=self._target_clifford,
            )
        )
        ts.append(FuseRotationTranspiler())

        return SequentialTranspiler(ts)

    def _construct_complex_gateset_decomposer(self) -> list[CircuitTranspiler]:
        return self._collect_decomposers_for_target_gateset(
            {
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
        )

    def _construct_two_qubit_gate_decomposer(self) -> list[CircuitTranspiler]:
        return self._collect_decomposers_for_target_gateset(
            {
                SWAP: dc.SWAP2CNOTTranspiler(),
                CZ: dc.CZ2CNOTHTranspiler(),
                CNOT: dc.CNOT2CZHTranspiler(),
            }
        )

    def _construct_clifford_to_rotation_decomposer(self) -> list[CircuitTranspiler]:
        return self._collect_decomposers_for_target_gateset(
            {
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
        )

    def _collect_decomposers_for_target_gateset(
        self,
        name_decomp_dict: dict[GateNameType, CircuitTranspiler],
    ) -> list[CircuitTranspiler]:
        ts = []
        for name, trans in name_decomp_dict.items():
            if name not in self._gateset:
                ts.append(trans)
        return ts

    def _validate(self, circuit: NonParametricQuantumCircuit) -> None:
        for gate in circuit.gates:
            if gate.name not in self._gateset:
                raise ValueError(f"{gate} cannot be converted into the target gateset.")

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        tr_circuit = self._decomposer(circuit)
        self._validate(tr_circuit)
        return tr_circuit
