# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping, Sequence
from math import pi
from typing import cast

from quri_parts.circuit import NonParametricQuantumCircuit, QuantumCircuit
from quri_parts.circuit import gates as gf
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
from .gate_kind_decomposer import (
    CNOT2CZHTranspiler,
    CZ2CNOTHTranspiler,
    H2RXRYTranspiler,
    RX2RZSqrtXTranspiler,
    RY2RZSqrtXTranspiler,
    S2RZTranspiler,
    Sdag2RZTranspiler,
    SqrtX2RXTranspiler,
    SqrtXdag2RXTranspiler,
    SqrtY2RYTranspiler,
    SqrtYdag2RYTranspiler,
    SWAP2CNOTTranspiler,
    T2RZTranspiler,
    Tdag2RZTranspiler,
    TOFFOLI2HTTdagCNOTTranspiler,
    U1ToRZTranspiler,
    U2ToRXRZTranspiler,
    U3ToRXRZTranspiler,
    X2RXTranspiler,
    Y2RYTranspiler,
    Z2RZTranspiler,
)
from .identity_manipulation import IdentityEliminationTranspiler
from .multi_pauli_decomposer import (
    PauliDecomposeTranspiler,
    PauliRotationDecomposeTranspiler,
)
from .transpiler import (
    CircuitTranspiler,
    CircuitTranspilerProtocol,
    GateKindDecomposer,
    SequentialTranspiler,
)
from .unitary_matrix_decomposer import (
    SingleQubitUnitaryMatrix2RYRZTranspiler,
    TwoQubitUnitaryMatrixKAKTranspiler,
)

# TODO Generate systematically
_equiv_clifford_table: Mapping[str, list[list[str]]] = {
    H: [[S, SqrtX, S]],
    X: [[Y, Z], [SqrtX, SqrtX], [SqrtXdag, SqrtXdag], [H, Z, H], [H, S, S, H]],
    Y: [[Z, X], [H, X, H], [S, S, X], [Z, H, Z, H], [S, S, H, S, S, H]],
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
    """A CircuitTranspiler that converts Clifford gates in a circuit into the desired
    Clifford gate sequences.

    Convert the Clifford gates in the circuit into gate sequences containing only the
    user-specified Clifford gates. Such conversions are done on a best-effort basis and
    may leave non target Clifford gates in the output.

    Clifford gates that could not be converted or non-Clifford gates will remain in
    place.

    Args:
        target_gateset: A Sequence of Clifford gate names to output.
    """

    def __init__(self, target_gateset: Sequence[CliffordGateNameType]):
        self._gateset = set(target_gateset)
        if self._gateset - (CLIFFORD_GATE_NAMES & SINGLE_QUBIT_GATE_NAMES):
            raise ValueError(
                "Target gateset must contain only single qubit clifford gates."
            )

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        ret = []

        for gate in circuit.gates:
            if gate.name not in CLIFFORD_GATE_NAMES & SINGLE_QUBIT_GATE_NAMES:
                ret.append(gate)
                continue

            if gate.name in self._gateset:
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
    """A CircuitTranspiler that converts RZ gates in a circuit into the gate sequences
    containing RX and RY gates.
    """

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
    """A CircuitTranspiler that converts RY gates in a circuit into the gate sequences
    containing RX and RZ gates.
    """

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
    """A CircuitTranspiler that converts RX gates in a circuit into the gate sequences
    containing RY and RZ gates.
    """

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [RX]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        qubit = gate.target_indices[0]
        theta = gate.params[0]
        return [
            gf.RZ(qubit, pi / 2.0),
            gf.RY(qubit, theta),
            gf.RZ(qubit, -pi / 2.0),
        ]


class RX2RZHTranspiler(GateKindDecomposer):
    """A CircuitTranspiler that converts RX gates in a circuit into the gate sequences
    containing RZ and H gates.
    """

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
    """A CircuitTranspiler that converts RY gates in a circuit into the gate sequences
    containing RZ and H gates.
    """

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
    """A CircuitTranspiler returns the same circuit as the input."""

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        return circuit


class RotationConversionTranspiler(CircuitTranspilerProtocol):
    """A CircuitTranspiler that converts rotation gates (RX, RY, and RZ) in a circuit to
    each other.

    ...

    Args:
        target_rotation: A Sequence of rotation gate names to output.
        favorable_clifford: A Sequence of Clifford gate names to be prioritesed as an
            output.
    """

    def __init__(
        self,
        target_rotation: Sequence[GateNameType],
        favorable_clifford: Sequence[CliffordGateNameType] = (),
    ):
        self._target_rotation = set(target_rotation)
        self._favorable_clifford = set(favorable_clifford)
        self._decomposer = self._construct_decomposer()
        # TODO Check gate type

    def _construct_decomposer(self) -> CircuitTranspiler:
        rot_to_trans_map = {
            frozenset({RX, RY, RZ}): IdentityTranspiler(),
            frozenset({RX, RY}): RZ2RXRYTranspiler(),
            frozenset({RY, RZ}): RX2RYRZTranspiler(),
            frozenset({RX, RZ}): RY2RXRZTranspiler(),
            frozenset({RZ}): SequentialTranspiler(
                [RX2RZHTranspiler(), RY2RZHTranspiler()]
            ),
            # TODO Support {RX}, {RY}, and {}
        }

        if H not in self._favorable_clifford and SqrtX in self._favorable_clifford:
            rot_to_trans_map[frozenset({RZ})] = SequentialTranspiler(
                [RX2RZSqrtXTranspiler(), RY2RZSqrtXTranspiler()]
            )
            # TODO Support {RX}, {RY}, and {}

        return rot_to_trans_map.get(
            frozenset(self._target_rotation), IdentityTranspiler()
        )

    def _validate(self, circuit: NonParametricQuantumCircuit) -> None:
        for gate in circuit.gates:
            if gate.name in {RX, RY, RZ} and gate.name not in self._target_rotation:
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
        self._target_clifford = cast(
            set[CliffordGateNameType],
            self._gateset & CLIFFORD_GATE_NAMES & SINGLE_QUBIT_GATE_NAMES,
        )
        self._target_rotation = self._gateset & {RX, RY, RZ}
        self._decomposer = self._construct_decomposer()

    def _construct_decomposer(self) -> CircuitTranspiler:
        ts = []

        ts.extend(self._construct_complex_gate_decomposer())
        ts.extend(self._construct_two_qubit_gate_decomposer())

        if self._target_clifford:
            ts.extend(
                [
                    Rotation2NamedTranspiler(),  # Optimizer
                    CliffordConversionTranspiler(tuple(self._target_clifford)),
                ]
            )

        if Identity not in self._gateset:
            ts.append(IdentityEliminationTranspiler())

        ts.extend(self._construct_clifford_to_rotation_decomposer())
        ts.extend(
            [
                FuseRotationTranspiler(),  # Optimizer
                RotationConversionTranspiler(
                    target_rotation=tuple(self._target_rotation),
                    favorable_clifford=tuple(self._target_clifford),
                ),
                FuseRotationTranspiler(),  # Optimizer
            ]
        )

        return SequentialTranspiler(ts)

    def _construct_complex_gate_decomposer(self) -> list[CircuitTranspiler]:
        return self._collect_decomposers_for_target_gateset(
            {
                Pauli: PauliDecomposeTranspiler(),
                PauliRotation: PauliRotationDecomposeTranspiler(),
                UnitaryMatrix: SequentialTranspiler(
                    [
                        SingleQubitUnitaryMatrix2RYRZTranspiler(),
                        TwoQubitUnitaryMatrixKAKTranspiler(),
                    ]
                ),
                TOFFOLI: TOFFOLI2HTTdagCNOTTranspiler(),
                U1: U1ToRZTranspiler(),
                U2: U2ToRXRZTranspiler(),
                U3: U3ToRXRZTranspiler(),
            }
        )

    def _construct_two_qubit_gate_decomposer(self) -> list[CircuitTranspiler]:
        return self._collect_decomposers_for_target_gateset(
            {
                SWAP: SWAP2CNOTTranspiler(),
                CZ: CZ2CNOTHTranspiler(),
                CNOT: CNOT2CZHTranspiler(),
            }
        )

    def _construct_clifford_to_rotation_decomposer(self) -> list[CircuitTranspiler]:
        return self._collect_decomposers_for_target_gateset(
            {
                H: H2RXRYTranspiler(),
                X: X2RXTranspiler(),
                Y: Y2RYTranspiler(),
                Z: Z2RZTranspiler(),
                SqrtX: SqrtX2RXTranspiler(),
                SqrtXdag: SqrtXdag2RXTranspiler(),
                SqrtY: SqrtY2RYTranspiler(),
                SqrtYdag: SqrtYdag2RYTranspiler(),
                S: S2RZTranspiler(),
                Sdag: Sdag2RZTranspiler(),
                T: T2RZTranspiler(),
                Tdag: Tdag2RZTranspiler(),
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
