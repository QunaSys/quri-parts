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

from quri_parts.circuit import QuantumCircuit, gate_names, gates
from quri_parts.circuit.transpile import (
    coupled_qubit_indices,
    extract_qubit_coupling_path,
    gate_count,
    gate_weighted_depth,
    qubit_couplings,
)


def _circuit_3() -> QuantumCircuit:
    circuit = QuantumCircuit(3)
    circuit.extend(
        [
            gates.H(0),
            gates.CNOT(0, 1),
            gates.X(1),
            gates.CNOT(1, 2),
            gates.H(2),
        ]
    )
    return circuit


def _circuit_5() -> QuantumCircuit:
    circuit = QuantumCircuit(5)
    circuit.extend(
        [
            gates.H(0),
            gates.TOFFOLI(0, 1, 2),
            gates.CZ(2, 4),
            gates.X(3),
            gates.CNOT(3, 4),
            gates.H(4),
        ]
    )
    return circuit


def test_gate_weighted_depth() -> None:
    circuit_3 = _circuit_3()
    assert 5 == gate_weighted_depth(circuit_3, {}, default_weight=1)
    assert 2 == gate_weighted_depth(circuit_3, {gate_names.CNOT: 1})
    assert 20 == gate_weighted_depth(circuit_3, {gate_names.H: 1, gate_names.CNOT: 9})

    circuit_5 = _circuit_5()
    assert 5 == gate_weighted_depth(circuit_5, {}, default_weight=1)
    assert 2 == gate_weighted_depth(circuit_5, {gate_names.H: 1})
    assert 1 == gate_weighted_depth(circuit_5, {gate_names.CZ: 1, gate_names.X: 1})
    assert 22 == gate_weighted_depth(
        circuit_5,
        {gate_names.CZ: 5, gate_names.CNOT: 5, gate_names.TOFFOLI: 10},
        default_weight=1,
    )

    assert 0 == gate_weighted_depth(QuantumCircuit(1), {})


def test_gate_count() -> None:
    circuit = _circuit_3()

    assert 5 == gate_count(circuit)
    assert 3 == gate_count(circuit, qubit_indices=[1])
    assert 2 == gate_count(circuit, gate_names=[gate_names.CNOT])
    assert 3 == gate_count(
        circuit, qubit_indices=[1, 2], gate_names=[gate_names.CNOT, gate_names.H]
    )

    assert 0 == gate_count(QuantumCircuit(0))


def test_qubit_couplings() -> None:
    assert qubit_couplings(_circuit_3()) == [(0, 1), (1, 2)]
    assert qubit_couplings(_circuit_3(), include_singles=True) == [
        (0,),
        (0, 1),
        (1,),
        (1, 2),
        (2,),
    ]
    assert qubit_couplings(_circuit_5()) == [(0, 1, 2), (2, 4), (3, 4)]
    assert qubit_couplings(_circuit_5(), include_singles=True) == [
        (0,),
        (0, 1, 2),
        (2, 4),
        (3,),
        (3, 4),
        (4,),
    ]
    assert qubit_couplings(QuantumCircuit(1)) == []
    assert qubit_couplings(QuantumCircuit(1), include_singles=True) == []


def test_coupled_qubit_indices() -> None:
    qs3 = coupled_qubit_indices(_circuit_3())
    assert len(qs3) == 1 and sorted(qs3[0]) == list(range(3))
    qs5 = coupled_qubit_indices(_circuit_5())
    assert len(qs5) == 1 and sorted(qs5[0]) == list(range(5))
    assert coupled_qubit_indices(QuantumCircuit(1)) == []


def test_extract_qubit_path() -> None:
    paths = extract_qubit_coupling_path(_circuit_3())
    assert list(sorted(paths)[0]) == [0, 1, 2]

    with pytest.raises(ValueError):
        paths = extract_qubit_coupling_path(_circuit_5())
