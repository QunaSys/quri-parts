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

from quri_parts.circuit import CNOT, RX, RY, RZ, SWAP, H, QuantumCircuit, QuantumGate, X
from quri_parts.circuit.transpile import FuseRotationTranspiler


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
                H(0),
                RX(1, np.pi / 2),
                RX(1, np.pi / 2),
                RX(2, np.pi / 2),
                RY(0, np.pi / 2),
                RY(0, np.pi / 2),
                RY(0, np.pi / 2),
                CNOT(0, 2),
                X(2),
                RZ(2, np.pi / 2),
                RZ(2, np.pi / 2),
            ]
        )
        transpiled = FuseRotationTranspiler()(circuit)

        expect = QuantumCircuit(3)
        expect.extend(
            [
                H(0),
                RX(1, np.pi),
                RX(2, np.pi / 2),
                RY(0, np.pi * 3 / 2),
                CNOT(0, 2),
                X(2),
                RZ(2, np.pi),
            ]
        )

        for t, e in zip(transpiled.gates, expect.gates):
            assert _gates_close(t, e)

    def test_no_change(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.extend(
            [RZ(0, np.pi), RZ(1, np.pi), H(0), RX(0, np.pi), RY(0, np.pi), SWAP(1, 0)]
        )
        transpiled = FuseRotationTranspiler()(circuit)

        expect = QuantumCircuit(2)
        expect.extend(
            [RZ(0, np.pi), RZ(1, np.pi), H(0), RX(0, np.pi), RY(0, np.pi), SWAP(1, 0)]
        )

        for t, e in zip(transpiled.gates, expect.gates):
            assert _gates_close(t, e)
