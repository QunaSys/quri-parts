# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Callable, Protocol

import numpy as np
from typing_extensions import TypeAlias

from quri_parts.circuit import gate_names, gates

from .circuit import NonParametricQuantumCircuit, QuantumCircuit
from .gate import QuantumGate

#: CircuitTranspiler Interface. A function or callable object that can map
#: NonParametricQuantumCircuit to NonParametricQuantumCircuit.
CircuitTranspiler: TypeAlias = Callable[
    [NonParametricQuantumCircuit], NonParametricQuantumCircuit
]


class CircuitTranspilerProtocol(Protocol):
    """Protocol of callable class that transpiles NonParametricQuantumCircuit
    to NonParametricQuantumCircuit."""

    @abstractmethod
    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        ...


class SequentialTranspiler(CircuitTranspilerProtocol):
    """CircuitTranspiler, which applies CircuitTranspilers in sequence.

    Args:
        transpilers: Sequence of CircuitTranspilers.

    Examples:
        .. highlight:: python
        .. code-block:: python

            transpiler = SequentialTranspiler(
                [
                    ATranspiler(),
                    BTranspiler(arg..),
                    ..
                ]
            )
            circuit = transpiler(circuit)
    """

    def __init__(self, transpilers: Sequence[CircuitTranspiler]):
        self._transpilers = transpilers

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        for transpiler in self._transpilers:
            circuit = transpiler(circuit)
        return circuit


class GateDecomposer(CircuitTranspilerProtocol, ABC):
    """Abstract class that represents CircuitTranspiler, such that target gates
    are selected by decision function and each target gate is replaced by a
    sequence of multiple gates."""

    @abstractmethod
    def is_target_gate(self, gate: QuantumGate) -> bool:
        """Determine if a given gate is subject to decomposition.

        Args:
            gate: Gates in the circuit that are scanned from the front.
        """
        ...

    @abstractmethod
    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        """Describe the specific decomposition process. Only the target gates
        satisfying is_target_gate() method are passed.

        Args:
            gate: The gates to be decomposed.
        """
        ...

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        cg: list[QuantumGate] = []
        for gate in circuit.gates:
            if self.is_target_gate(gate):
                cg.extend(self.decompose(gate))
            else:
                cg.append(gate)

        cc = QuantumCircuit(circuit.qubit_count)
        cc.extend(cg)
        return cc


class GateKindDecomposer(GateDecomposer, ABC):
    """Abstract class that represents CircuitTranspiler, such that each gate is
    identified by its gate name and the target gate is replaced by a sequence
    of multiple gates.

    Classes inheriting from this class can be used for
    ParallelDecomposer.
    """

    @property
    @abstractmethod
    def target_gate_names(self) -> Sequence[str]:
        """Returns the set of gate names to be decomposed."."""
        ...

    def is_target_gate(self, gate: QuantumGate) -> bool:
        return gate.name in self.target_gate_names


class ParallelDecomposer(CircuitTranspilerProtocol):
    """CircuitTranspiler, which executes given GateKindDecomposer within a
    single loop traversing the gate from the front.

    Args:
        decomposers: Sequence of GateKindDecomposer with no duplicate gate types
            to act on.
    """

    def __init__(self, decomposers: Sequence[GateKindDecomposer]):
        self._decomposer_map: Mapping[str, GateKindDecomposer] = {}
        for dc in decomposers:
            for name in dc.target_gate_names:
                if name in self._decomposer_map:
                    raise ValueError(
                        "Multiple transpilers acting on the same type of gate"
                        " were given."
                    )
                self._decomposer_map[name] = dc

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        cg: list[QuantumGate] = []
        for gate in circuit.gates:
            if gate.name in self._decomposer_map:
                cg.extend(self._decomposer_map[gate.name].decompose(gate))
            else:
                cg.append(gate)

        cc = QuantumCircuit(circuit.qubit_count)
        cc.extend(cg)
        return cc


class PauliDecomposeTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose multi-qubit Pauli gates into X, Y,
    and Z gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.Pauli]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        indices = gate.target_indices
        pauli_ids = gate.pauli_ids
        ret: list[QuantumGate] = []

        for index, pauli in zip(indices, pauli_ids):
            if pauli == 1:
                ret.append(gates.X(index))
            elif pauli == 2:
                ret.append(gates.Y(index))
            elif pauli == 3:
                ret.append(gates.Z(index))
            else:
                raise ValueError("Pauli id must be either 1, 2, or 3.")

        return ret


class PauliRotationDecomposeTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose multi-qubit PauliRotation gates into
    H, RX, RZ, and CNOT gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.PauliRotation]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        indices = gate.target_indices
        pauli_ids = gate.pauli_ids
        angle = gate.params[0]
        ret: list[QuantumGate] = []

        def rot_gates(rot_sign: int = 1) -> Sequence[QuantumGate]:
            rc = []
            for index, pauli in zip(indices, pauli_ids):
                if pauli == 1:
                    rc.append(gates.H(index))
                elif pauli == 2:
                    rc.append(gates.RX(index, rot_sign * np.pi / 2.0))
                elif pauli == 3:
                    pass
                else:
                    raise ValueError("Pauli id must be either 1, 2, or 3.")
            return rc

        ret.extend(rot_gates(1))
        for i in range(1, len(indices)):
            ret.append(gates.CNOT(indices[i], indices[0]))
        ret.append(gates.RZ(indices[0], angle))
        for i in range(1, len(indices)):
            ret.append(gates.CNOT(indices[i], indices[0]))
        ret.extend(rot_gates(-1))

        return ret


class CNOT2CZHTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose CNOT gates into sequence of H and CZ
    gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.CNOT]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        control, target = gate.control_indices[0], gate.target_indices[0]
        return [
            gates.H(target),
            gates.CZ(control, target),
            gates.H(target),
        ]


class CZ2CNOTHTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose CZ gates into sequence of H and CNOT
    gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.CZ]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        control, target = gate.control_indices[0], gate.target_indices[0]
        return [
            gates.H(target),
            gates.CNOT(control, target),
            gates.H(target),
        ]


class SWAP2CNOTTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose SWAP gates into sequence of CNOT
    gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.SWAP]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target1, target2 = gate.target_indices
        return [
            gates.CNOT(target1, target2),
            gates.CNOT(target2, target1),
            gates.CNOT(target1, target2),
        ]


class Z2HXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose Z gates into sequence of H and X
    gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.Z]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [
            gates.H(target),
            gates.X(target),
            gates.H(target),
        ]


class X2HZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose X gates into sequence of H and Z
    gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.X]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [
            gates.H(target),
            gates.Z(target),
            gates.H(target),
        ]


class X2SqrtXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose X gates into sequence of SqrtX
    gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.X]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gates.SqrtX(target), gates.SqrtX(target)]


class SqrtX2RZHTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose SqrtX gates into sequence of RZ and H
    gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.SqrtX]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [
            gates.RZ(target, -np.pi / 2.0),
            gates.H(target),
            gates.RZ(target, -np.pi / 2.0),
        ]


class H2RZSqrtXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose H gates into sequence of RZ and SqrtX
    gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.H]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [
            gates.RZ(target, np.pi / 2.0),
            gates.SqrtX(target),
            gates.RZ(target, np.pi / 2.0),
        ]


class Y2RZXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose Y gates into sequence of RZ and T
    gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.Y]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [
            gates.RZ(target, -np.pi),
            gates.X(target),
        ]


class Z2RZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose Z gates into RZ gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.Z]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gates.RZ(target, np.pi)]


class SqrtXdag2RZSqrtXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose SqrtXdag gates into sequence of RZ
    and SqrtX gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.SqrtXdag]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [
            gates.RZ(target, -np.pi),
            gates.SqrtX(target),
            gates.RZ(target, -np.pi),
        ]


class SqrtY2RZSqrtXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose SqrtY gates into sequence of RZ and
    SqrtX gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.SqrtY]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [
            gates.RZ(target, -np.pi / 2.0),
            gates.SqrtX(target),
            gates.RZ(target, np.pi / 2.0),
        ]


class SqrtYdag2RZSqrtXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose SqrtYdag gates into sequence of RZ
    and SqrtX gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.SqrtYdag]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [
            gates.RZ(target, np.pi / 2.0),
            gates.SqrtX(target),
            gates.RZ(target, -np.pi / 2.0),
        ]


class S2RZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose S gates into of gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.S]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gates.RZ(target, np.pi / 2.0)]


class Sdag2RZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose Sdag gates into RZ gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.Sdag]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gates.RZ(target, -np.pi / 2.0)]


class T2RZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose T gates into RZ gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.T]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gates.RZ(target, np.pi / 4.0)]


class Tdag2RZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose Tdag gates into RZ gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.Tdag]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gates.RZ(target, -np.pi / 4.0)]


class RX2RZSqrtXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose RX gates into sequence of RZ and
    SqrtX gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.RX]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        theta = gate.params[0]
        return [
            gates.RZ(target, np.pi / 2.0),
            gates.SqrtX(target),
            gates.RZ(target, theta + np.pi),
            gates.SqrtX(target),
            gates.RZ(target, 5.0 * np.pi / 2.0),
        ]


class RY2RZSqrtXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose RY gates into sequence of RZ and
    SqrtX gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.RY]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        theta = gate.params[0]
        return [
            gates.SqrtX(target),
            gates.RZ(target, theta + np.pi),
            gates.SqrtX(target),
            gates.RZ(target, 3.0 * np.pi),
        ]


class U1ToRZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose U1 gates into RZ gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.U1]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        lam = gate.params[0]
        return [gates.RZ(target, lam)]


class U2ToRZSqrtXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose U2 gates into sequence of RZ and
    SqrtX gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.U2]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        phi, lam = gate.params
        return [
            gates.RZ(target, lam - np.pi / 2.0),
            gates.SqrtX(target),
            gates.RZ(target, phi + np.pi / 2.0),
        ]


class U3ToRZSqrtXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose U3 gates into sequence of RZ and
    SqrtX gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.U3]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        theta, phi, lam = gate.params
        return [
            gates.RZ(target, lam),
            gates.SqrtX(target),
            gates.RZ(target, theta + np.pi),
            gates.SqrtX(target),
            gates.RZ(target, phi + 3.0 * np.pi),
        ]


class CliffordApproximationTranspiler(GateDecomposer):
    r"""CircuitTranspiler, which replaces the non_clifford gate into sequence of
    non-parametric Clifford gates.

    If the input gate has angles, this transpiler replaces them with the
    closest value in the set :math:`\{\pi n /2| n\in\mathbb{Z}\}`. Then
    by using the rotation gate transpilers, it is decomposed into a
    sequence of non-parametric Clifford gates.
    """

    def is_target_gate(self, gate: QuantumGate) -> bool:
        return gate.name not in gate_names.CLIFFORD_GATE_NAMES

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        clif_set_x = {0: "Identity", 1: "SqrtX", 2: "X", 3: "SqrtXdag"}
        clif_set_y = {0: "Identity", 1: "SqrtY", 2: "Y", 3: "SqrtYdag"}
        clif_set_z = {0: "Identity", 1: "S", 2: "Z", 3: "Sdag"}
        clif_set = {"RX": clif_set_x, "RY": clif_set_y, "RZ": clif_set_z}

        target = gate.target_indices[0]
        if gate.name == "T":
            return [gates.S(target)]

        elif gate.name == "Tdag":
            return [gates.Sdag(target)]

        elif gate.name in {"RX", "RY", "RZ", "U1", "U2", "U3", "PauliRotation"}:
            if gate.name in {"RX", "RY", "RZ"}:
                transpiled_gates: Sequence[QuantumGate] = [gate]
            if gate.name == "U1":
                transpiled_gates = U1ToRZTranspiler().decompose(gate)
            if gate.name == "U2":
                transpiled_gates = U2ToRZSqrtXTranspiler().decompose(gate)
            if gate.name == "U3":
                transpiled_gates = U3ToRZSqrtXTranspiler().decompose(gate)
            if gate.name == "PauliRotation":
                transpiled_gates = PauliRotationDecomposeTranspiler().decompose(gate)

            appro_gates = []
            for g in transpiled_gates:
                if g.name in {"RX", "RY", "RZ"}:
                    param = g.params[0]
                    angle_int = np.round(2 * param / np.pi) % 4
                    appro_gates.append(
                        QuantumGate(clif_set[g.name][angle_int], (g.target_indices[0],))
                    )
                    continue
                appro_gates.append(g)
            return appro_gates

        else:
            raise NotImplementedError(
                f"The input gate {gate.name} is not supported on the\
                CliffordApproximationTranspiler."
            )


class IdentityInsertionTranspiler(CircuitTranspilerProtocol):
    """If there are qubits to which any gate has not been applied, Identity
    gates are added for those qubits.

    The application of this transpiler ensures that every qubit has at
    least one gate acting on it.
    """

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        non_applied = set(range(circuit.qubit_count))
        for gate in circuit.gates:
            for q in tuple(gate.control_indices) + tuple(gate.target_indices):
                non_applied.discard(q)
            if not non_applied:
                return circuit

        cc = QuantumCircuit(circuit.qubit_count)
        cc.extend(circuit.gates)
        for q in non_applied:
            cc.add_gate(gates.Identity(q))
        return cc


#: CircuitTranspiler to transpile a QuntumCircuit into another
#: QuantumCircuit containing only X, SqrtX, CNOT, and RZ.
RZSetTranspiler: Callable[[], CircuitTranspiler] = lambda: SequentialTranspiler(
    [
        ParallelDecomposer(
            [
                CZ2CNOTHTranspiler(),
                PauliDecomposeTranspiler(),
                PauliRotationDecomposeTranspiler(),
            ]
        ),
        ParallelDecomposer(
            [
                Y2RZXTranspiler(),
                Z2RZTranspiler(),
                H2RZSqrtXTranspiler(),
                SqrtXdag2RZSqrtXTranspiler(),
                SqrtY2RZSqrtXTranspiler(),
                SqrtYdag2RZSqrtXTranspiler(),
                S2RZTranspiler(),
                Sdag2RZTranspiler(),
                T2RZTranspiler(),
                Tdag2RZTranspiler(),
                RX2RZSqrtXTranspiler(),
                RY2RZSqrtXTranspiler(),
                U1ToRZTranspiler(),
                U2ToRZSqrtXTranspiler(),
                U3ToRZSqrtXTranspiler(),
            ]
        ),
    ]
)
