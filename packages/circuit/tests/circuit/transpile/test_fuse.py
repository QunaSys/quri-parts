# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from quri_parts.circuit import QuantumCircuit, QuantumGate, gates
from quri_parts.circuit.transpile import (
    FuseRotationTranspiler,
    NormalizeRotationTranspiler,
    RX2NamedTranspiler,
    RY2NamedTranspiler,
    RZ2NamedTranspiler,
)


def _gates_close(x: QuantumGate, y: QuantumGate) -> bool:
    return (
        x.name == y.name
        and x.target_indices == y.target_indices
        and x.control_indices == y.control_indices
        and np.allclose(x.params, y.params)
        and x.pauli_ids == y.pauli_ids
        and np.allclose(x.unitary_matrix, y.unitary_matrix)
    )


class TestFuseRotation:
    def test_fuse(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.extend(
            [
                gates.H(0),
                gates.RX(1, np.pi / 2.0),
                gates.RX(1, np.pi / 2.0),
                gates.RX(2, np.pi / 2.0),
                gates.RY(0, np.pi / 2.0),
                gates.RY(0, np.pi / 2.0),
                gates.RY(0, np.pi / 2.0),
                gates.CNOT(0, 2),
                gates.X(2),
                gates.RZ(2, np.pi / 2.0),
                gates.RZ(2, np.pi / 2.0),
            ]
        )
        transpiled = FuseRotationTranspiler()(circuit)

        expect = QuantumCircuit(3)
        expect.extend(
            [
                gates.H(0),
                gates.RX(1, np.pi),
                gates.RX(2, np.pi / 2.0),
                gates.RY(0, 3.0 * np.pi / 2.0),
                gates.CNOT(0, 2),
                gates.X(2),
                gates.RZ(2, np.pi),
            ]
        )

        for t, e in zip(transpiled.gates, expect.gates):
            assert _gates_close(t, e)


class TestNormalizeRotation:
    def test_normalize_2pi(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.extend(
            [
                gates.RX(0, 5.0 * np.pi),
                gates.RY(0, -3.0 * np.pi),
                gates.RZ(0, 7.0 / 2.0 * np.pi),
                gates.RX(0, 3.0 / np.pi),
                gates.RY(0, 2.0 * np.pi),
                gates.RZ(0, 0.0),
            ]
        )
        transpiled = NormalizeRotationTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                gates.RX(0, np.pi),
                gates.RY(0, np.pi),
                gates.RZ(0, 3.0 / 2.0 * np.pi),
                gates.RX(0, 3.0 / np.pi),
                gates.RY(0, 0.0),
                gates.RZ(0, 0.0),
            ]
        )

        for t, e in zip(transpiled.gates, expect.gates):
            assert _gates_close(t, e)

    def test_normalize_pi(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.extend(
            [
                gates.RX(0, 3.0 / 2.0 * np.pi),
                gates.RY(0, -3.0 / 4.0 * np.pi),
                gates.RZ(0, 7.0 / 2.0 * np.pi),
                gates.RX(0, np.pi),
                gates.RY(0, -np.pi),
                gates.RZ(0, 0.0),
            ]
        )
        transpiled = NormalizeRotationTranspiler((-np.pi, np.pi))(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                gates.RX(0, -np.pi / 2.0),
                gates.RY(0, -3.0 / 4.0 * np.pi),
                gates.RZ(0, -np.pi / 2.0),
                gates.RX(0, -np.pi),
                gates.RY(0, -np.pi),
                gates.RZ(0, 0.0),
            ]
        )

        for t, e in zip(transpiled.gates, expect.gates):
            assert _gates_close(t, e)

    def test_normalize_5pi(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.extend(
            [
                gates.RZ(0, -2.0 * np.pi),
                gates.RY(0, -np.pi),
                gates.RX(0, 0.0),
                gates.RY(0, np.pi),
                gates.RZ(0, 2.0 * np.pi),
            ]
        )
        transpiled = NormalizeRotationTranspiler((3.0 * np.pi, 5.0 * np.pi))(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                gates.RZ(0, 4.0 * np.pi),
                gates.RY(0, 3.0 * np.pi),
                gates.RX(0, 4.0 * np.pi),
                gates.RY(0, 3.0 * np.pi),
                gates.RZ(0, 4.0 * np.pi),
            ]
        )

        for t, e in zip(transpiled.gates, expect.gates):
            assert _gates_close(t, e)

    def test_normalize_minus3pi(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.extend(
            [
                gates.RZ(0, -2.0 * np.pi),
                gates.RY(0, -np.pi),
                gates.RX(0, 0.0),
                gates.RY(0, np.pi),
                gates.RZ(0, 2.0 * np.pi),
            ]
        )
        transpiled = NormalizeRotationTranspiler((-5.0 * np.pi, -3.0 * np.pi))(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                gates.RZ(0, -4.0 * np.pi),
                gates.RY(0, -5.0 * np.pi),
                gates.RX(0, -4.0 * np.pi),
                gates.RY(0, -5.0 * np.pi),
                gates.RZ(0, -4.0 * np.pi),
            ]
        )

        for t, e in zip(transpiled.gates, expect.gates):
            assert _gates_close(t, e)


class TestRotation2Named:
    def test_rx2named(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.extend(
            [
                gates.RX(0, 0.0),
                gates.RX(0, np.pi / 2.0),
                gates.RX(0, np.pi),
                gates.RX(0, np.pi * 3.0 / 2.0),
            ]
        )
        transpiled = RX2NamedTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                gates.Identity(0),
                gates.SqrtX(0),
                gates.X(0),
                gates.SqrtXdag(0),
            ]
        )

        for t, e in zip(transpiled.gates, expect.gates):
            assert _gates_close(t, e)

    def test_ry2named(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.extend(
            [
                gates.RY(0, 0.0),
                gates.RY(0, np.pi / 2.0),
                gates.RY(0, np.pi),
                gates.RY(0, np.pi * 3.0 / 2.0),
            ]
        )
        transpiled = RY2NamedTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                gates.Identity(0),
                gates.SqrtY(0),
                gates.Y(0),
                gates.SqrtYdag(0),
            ]
        )

        for t, e in zip(transpiled.gates, expect.gates):
            assert _gates_close(t, e)

    def test_rz2named(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.extend(
            [
                gates.RZ(0, 0.0),
                gates.RZ(0, np.pi / 4.0),
                gates.RZ(0, np.pi / 2.0),
                gates.RZ(0, np.pi * 3.0 / 4.0),
                gates.RZ(0, np.pi),
                gates.RZ(0, np.pi * 5.0 / 4.0),
                gates.RZ(0, np.pi * 3.0 / 2.0),
                gates.RZ(0, np.pi * 7.0 / 4.0),
                gates.RZ(0, np.pi * 2.0),
                gates.RZ(0, -np.pi),
                gates.RZ(0, -np.pi / 2.0),
                gates.RZ(0, -np.pi / 4.0),
            ]
        )
        transpiled = RZ2NamedTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                gates.Identity(0),
                gates.T(0),
                gates.S(0),
                gates.S(0),
                gates.T(0),
                gates.Z(0),
                gates.Z(0),
                gates.T(0),
                gates.Sdag(0),
                gates.Tdag(0),
                gates.Identity(0),
                gates.Z(0),
                gates.Sdag(0),
                gates.Tdag(0),
            ]
        )

        for t, e in zip(transpiled.gates, expect.gates):
            assert _gates_close(t, e)
