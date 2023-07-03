# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import CNOT, CZ, SWAP, H, QuantumCircuit, X
from quri_parts.circuit.topology import (
    SquareLattice,
    SquareLatticeSWAPInsertionTranspiler,
    approx_reliable_coupling_subgraph,
    effectively_coupled_qubit_counts,
    reliable_coupling_single_stroke_path,
    reliable_coupling_single_stroke_path_qubit_mapping,
)


def test_square_lattice_swap_insertion() -> None:
    circuit = QuantumCircuit(9)
    gates = [
        H(0),
        CNOT(0, 8),
        CZ(3, 0),
        SWAP(5, 1),
        X(4),
        CZ(2, 7),
    ]
    circuit.extend(gates)
    square_lattice = SquareLattice(3, 3)
    transpiled = SquareLatticeSWAPInsertionTranspiler(square_lattice)(circuit)

    expect = QuantumCircuit(9)
    expect_gates = [
        H(0),
        SWAP(0, 1),
        SWAP(1, 2),
        SWAP(2, 5),
        CNOT(5, 8),
        SWAP(2, 5),
        SWAP(1, 2),
        SWAP(0, 1),
        CZ(3, 0),
        SWAP(5, 4),
        SWAP(4, 1),
        SWAP(5, 4),
        X(4),
        SWAP(2, 1),
        SWAP(1, 4),
        CZ(4, 7),
        SWAP(1, 4),
        SWAP(2, 1),
    ]
    expect.extend(expect_gates)

    assert transpiled.gates == expect.gates


def _cnot_errors_3_loops() -> dict[tuple[int, int], float]:
    cs = {
        (0, 1): 0.3,  # loop 3
        (1, 2): 0.3,
        (2, 0): 0.3,
        (2, 3): 0.4,  # bridge
        (3, 4): 0.2,  # loop 4
        (4, 5): 0.2,
        (5, 6): 0.2,
        (6, 3): 0.2,
        (6, 7): 0.5,  # bridge
        (7, 8): 0.1,  # loop 5
        (8, 9): 0.1,
        (9, 10): 0.1,
        (10, 11): 0.1,
        (11, 7): 0.1,
    }
    rs = {(b, a): e for (a, b), e in cs.items()}
    return cs | rs


def _cnot_errors_steps() -> dict[tuple[int, int], float]:
    cs = {
        (0, 1): 0.5,  # loop 8
        (1, 2): 0.4,
        (2, 3): 0.3,
        (3, 4): 0.2,
        (4, 5): 0.1,
        (5, 6): 0.8,
        (6, 7): 0.7,
        (7, 0): 0.6,
    }
    rs = {(b, a): e for (a, b), e in cs.items()}
    return cs | rs


def test_effectively_coupled_qubits_counts() -> None:
    map_loops = _cnot_errors_3_loops()
    assert [12] == list(effectively_coupled_qubit_counts(map_loops, 1.0))
    assert [5, 7] == sorted(effectively_coupled_qubit_counts(map_loops, 0.45))
    assert [3, 4, 5] == sorted(effectively_coupled_qubit_counts(map_loops, 0.35))
    assert [4, 5] == sorted(effectively_coupled_qubit_counts(map_loops, 0.25))
    assert [5] == sorted(effectively_coupled_qubit_counts(map_loops, 0.15))
    assert [] == sorted(effectively_coupled_qubit_counts(map_loops, 0.05))


def test_approx_reliable_coupling_subgraph() -> None:
    map_loops = _cnot_errors_3_loops()
    graphs_3_5 = approx_reliable_coupling_subgraph(map_loops, 5)
    assert len(graphs_3_5) == 1
    assert set(range(7, 12)) == set(map(int, graphs_3_5[0].nodes))
    graphs_3_7 = approx_reliable_coupling_subgraph(map_loops, 7)
    assert len(graphs_3_7) == 1
    assert set(range(7)) == set(map(int, graphs_3_7[0].nodes))
    graphs_3_12 = approx_reliable_coupling_subgraph(map_loops, 12)
    assert len(graphs_3_12) == 1
    assert set(range(12)) == set(map(int, graphs_3_12[0].nodes))
    assert [] == approx_reliable_coupling_subgraph(map_loops, 13)

    map_steps = _cnot_errors_steps()
    graphs_s_3 = approx_reliable_coupling_subgraph(map_steps, 3)
    assert len(graphs_s_3) == 1
    assert set([3, 4, 5]) == set(map(int, graphs_s_3[0].nodes))
    graphs_s_8 = approx_reliable_coupling_subgraph(map_steps, 8)
    assert len(graphs_s_8) == 1
    assert set(range(8)) == set(map(int, graphs_s_8[0].nodes))
    assert [] == approx_reliable_coupling_subgraph(map_steps, 13)


def test_reliable_coupling_single_stroke_path() -> None:
    map_loops = _cnot_errors_3_loops()
    assert set(range(12)) == set(
        reliable_coupling_single_stroke_path(map_loops, 12, True)
    )
    assert set(range(3, 12)) == set(
        reliable_coupling_single_stroke_path(map_loops, 9, True)
    )
    assert set(range(7, 12)) == set(
        reliable_coupling_single_stroke_path(map_loops, 5, True)
    )
    assert [] == list(reliable_coupling_single_stroke_path(map_loops, 13, True))

    map_steps = _cnot_errors_steps()
    assert [5, 4, 3, 2, 1, 0, 7, 6] == list(
        reliable_coupling_single_stroke_path(map_steps, 8, True)
    )
    assert [2, 3, 4, 5] == list(
        reliable_coupling_single_stroke_path(map_steps, 4, True)
    )
    assert [3, 4, 5] == list(reliable_coupling_single_stroke_path(map_steps, 3, True))
    assert [] == list(reliable_coupling_single_stroke_path(map_steps, 9, True))


def test_reliable_coupling_single_stroke_path_qubit_mapping() -> None:
    circuit = QuantumCircuit(4)
    circuit.extend(
        [
            H(0),
            CNOT(0, 1),
            H(1),
            CNOT(1, 2),
            H(2),
            CNOT(2, 3),
            H(3),
        ]
    )
    cnot_errors = _cnot_errors_steps()
    transpiler = reliable_coupling_single_stroke_path_qubit_mapping(
        circuit, cnot_errors, True
    ).circuit_transpiler
    transpiled = transpiler(circuit)

    expect = QuantumCircuit(7)
    expect.extend(
        [
            H(5),
            CNOT(5, 4),
            H(4),
            CNOT(4, 3),
            H(3),
            CNOT(3, 2),
            H(2),
        ]
    )

    assert transpiled.gates == expect.gates
