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
    CliffordConversionTranspiler,
    GateSetConversionTranspiler,
    RotationConversionTranspiler,
    RX2RYRZTranspiler,
    RX2RZHTranspiler,
    RY2RXRZTranspiler,
    RY2RZHTranspiler,
    RZ2RXRYTranspiler,
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


def _circuit_close(x: QuantumCircuit, y: QuantumCircuit) -> bool:
    return len(x.gates) == len(y.gates) and all(
        _gates_close(a, b) for a, b in zip(x.gates, y.gates)
    )


class TestCliffordConversion:
    ...


class TestRotationConversion:
    def test_rx2ryrz_transpile(self) -> None:
        theta = 2.0 * np.pi * np.random.rand()

        circuit = QuantumCircuit(1)
        circuit.add_RX_gate(0, theta)
        transpiled = RX2RYRZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                gates.RZ(0, np.pi / 2.0),
                gates.RY(0, theta),
                gates.RZ(0, -np.pi / 2.0),
            ]
        )
        assert _circuit_close(transpiled, expect)

    def test_rx2rzh_transpile(self) -> None:
        theta = 2.0 * np.pi * np.random.rand()

        circuit = QuantumCircuit(1)
        circuit.add_RX_gate(0, theta)
        transpiled = RX2RZHTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                gates.H(0),
                gates.RZ(0, theta),
                gates.H(0),
            ]
        )
        assert _circuit_close(transpiled, expect)

    def test_ry2rxrz_transpile(self) -> None:
        theta = 2.0 * np.pi * np.random.rand()

        circuit = QuantumCircuit(1)
        circuit.add_RY_gate(0, theta)
        transpiled = RY2RXRZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                gates.RX(0, np.pi / 2.0),
                gates.RZ(0, theta),
                gates.RX(0, -np.pi / 2.0),
            ]
        )
        assert _circuit_close(transpiled, expect)

    def test_ry2rzh_transpile(self) -> None:
        theta = 2.0 * np.pi * np.random.rand()

        circuit = QuantumCircuit(1)
        circuit.add_RY_gate(0, theta)
        transpiled = RY2RZHTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                gates.RZ(0, -np.pi / 2.0),
                gates.H(0),
                gates.RZ(0, theta),
                gates.H(0),
                gates.RZ(0, np.pi / 2.0),
            ]
        )
        assert _circuit_close(transpiled, expect)

    def test_rz2rxry_transpile(self) -> None:
        theta = 2.0 * np.pi * np.random.rand()

        circuit = QuantumCircuit(1)
        circuit.add_RZ_gate(0, theta)
        transpiled = RZ2RXRYTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                gates.RX(0, np.pi / 2.0),
                gates.RY(0, -theta),
                gates.RX(0, -np.pi / 2.0),
            ]
        )
        assert _circuit_close(transpiled, expect)


class TestGateSetConversion:
    ...
