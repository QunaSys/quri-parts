# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import CNOT, SWAP, H, Identity, QuantumCircuit, X
from quri_parts.circuit.transpile import IdentityInsertionTranspiler


class TestIdentityInsertion:
    def test_fill_void(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.extend([H(0), CNOT(0, 2), X(2)])
        transpiled = IdentityInsertionTranspiler()(circuit)

        expect = QuantumCircuit(3)
        expect.extend([H(0), CNOT(0, 2), X(2), Identity(1)])

        assert transpiled.gates == expect.gates

    def test_fill_empty(self) -> None:
        circuit = QuantumCircuit(2)
        transpiled = IdentityInsertionTranspiler()(circuit)

        expect = QuantumCircuit(2)
        expect.extend([Identity(0), Identity(1)])

        assert transpiled.gates == expect.gates

    def test_no_change(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.extend([H(0), SWAP(1, 0)])
        transpiled = IdentityInsertionTranspiler()(circuit)

        expect = QuantumCircuit(2)
        expect.extend([H(0), SWAP(1, 0)])

        assert transpiled.gates == expect.gates
