# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from quri_parts.circuit import (
    CNOT,
    RX,
    ImmutableQuantumCircuit,
    ParametricRX,
    QuantumCircuit,
    QuantumGate,
    X,
)

_GATES: list[QuantumGate] = [X(0), RX(0, angle=1.0), CNOT(0, 1)]


def mutable_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    for gate in _GATES:
        circuit.add_gate(gate)
    return circuit


def immutable_circuit() -> ImmutableQuantumCircuit:
    q_circuit = mutable_circuit()
    return ImmutableQuantumCircuit(q_circuit)


class TestQuantumCircuit:
    def test_add_gate(self) -> None:
        circuit = mutable_circuit()
        assert circuit.qubit_count == 2
        assert len(circuit.gates) == len(_GATES)
        assert circuit.gates == tuple(_GATES)
        with pytest.raises(ValueError):
            circuit.add_gate(ParametricRX(0))  # type: ignore
        with pytest.raises(ValueError):
            circuit.add_gate(X(3))

    def test_get_mutable_copy(self) -> None:
        circuit = mutable_circuit()
        circuit_copied = circuit.get_mutable_copy()
        assert id(circuit) != id(circuit_copied)
        assert circuit == circuit_copied

    def test_freeze(self) -> None:
        circuit = mutable_circuit()
        immutable_circuit = circuit.freeze()
        assert isinstance(immutable_circuit, ImmutableQuantumCircuit)
        assert circuit == immutable_circuit
        assert circuit.gates == immutable_circuit.gates

        mut_circuit = immutable_circuit.get_mutable_copy()
        assert circuit == mut_circuit
        assert id(circuit) != id(mut_circuit)
        assert circuit.gates == mut_circuit.gates

    def test_depth(self) -> None:
        circuit = mutable_circuit()
        assert circuit.depth == 3
        immutable_circuit = circuit.freeze()
        assert immutable_circuit.depth == 3

        circuit_2 = QuantumCircuit(3)
        circuit_2.add_X_gate(0)
        circuit_2.add_CNOT_gate(0, 2)
        circuit_2.add_X_gate(1)
        assert circuit_2.depth == 2
        immutable_circuit_2 = circuit_2.freeze()
        assert immutable_circuit_2.depth == 2


class TestImmutableQuantumCircuit:
    def test_get_mutable_copy(self) -> None:
        circuit = immutable_circuit()
        mut_circuit = circuit.get_mutable_copy()
        assert isinstance(mut_circuit, QuantumCircuit)
        assert circuit == mut_circuit

    def test_freeze(self) -> None:
        circuit = immutable_circuit()
        circuit_ = circuit.freeze()
        assert id(circuit) == id(circuit_)
