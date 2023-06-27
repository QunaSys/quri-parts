import itertools as it
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Optional, cast

import networkx as nx

from quri_parts.circuit import NonParametricQuantumCircuit


def gate_weighted_depth(
    circuit: NonParametricQuantumCircuit,
    weights: Mapping[str, int],
    default_weight: int = 0,
) -> int:
    qubit_depths: dict[int, int] = defaultdict()
    _weights = dict(weights)
    for gate in circuit.gates:
        qubits = tuple(gate.control_indices) + tuple(gate.target_indices)
        depth = _weights.get(gate.name, default_weight) + max(
            qubit_depths.get(q, 0) for q in qubits
        )
        for q in qubits:
            qubit_depths[q] = depth
    return max(qubit_depths.values()) if qubit_depths else 0


def gate_count(
    circuit: NonParametricQuantumCircuit,
    qubit_indices: Sequence[int] = (),
    gate_names: Sequence[str] = (),
) -> int:
    any_qubits, any_gates = not qubit_indices, not gate_names
    target_qubits, target_gates = set(qubit_indices), set(gate_names)
    count = 0
    for gate in circuit.gates:
        qubits = set(tuple(gate.control_indices) + tuple(gate.target_indices))
        if (any_gates or gate.name in target_gates) and (
            any_qubits or qubits & target_qubits
        ):
            count += 1
    return count


def qubit_couplings(
    circuit: NonParametricQuantumCircuit,
) -> Sequence[tuple[int, ...]]:
    coupling = set()
    for gate in circuit.gates:
        qubits = tuple(gate.control_indices) + tuple(gate.target_indices)
        if len(qubits) >= 2:
            coupling.add(qubits)
    return list(coupling)


def extract_qubit_coupling_path(
    circuit: NonParametricQuantumCircuit,
) -> Sequence[int]:
    couplings = set()
    for qs in qubit_couplings(circuit):
        if len(qs) > 2:
            raise ValueError("The given circuit contains a more than 3 qubits gate.")
        couplings |= {qs, (qs[1], qs[0])}
    graph = nx.parse_adjlist([f"{a} {b}" for a, b in couplings])
    paths = []
    nodes = list(graph.nodes)
    for s in nodes:
        for e in nodes:
            ps = nx.all_simple_paths(graph, s, e)
            paths.extend([list(map(int, p)) for p in ps if len(p) == len(nodes)])
    if len(paths) == 2 and set(paths[0]) == set(paths[1]):
        return list(map(int, paths[0]))
    return []
