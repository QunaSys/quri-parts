import itertools as it
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Optional, cast

import networkx as nx

from quri_parts.circuit import NonParametricQuantumCircuit


def gate_weighted_depth(
    circuit: NonParametricQuantumCircuit,
    weights: Mapping[str, int] = {},
    default_weight: int = 1,
) -> int:
    qubit_depths: dict[int, int] = defaultdict()
    _weights = dict(weights)
    for gate in circuit.gates:
        qubits = tuple(gate.control_indices) + tuple(gate.target_indices)
        depth = _weights.get(gate.name, default_weight) + max(
            qubit_depths[q] for q in qubits
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
    count = 0
    target_qubits = set(qubit_indices)
    target_gates = set(gate_names)
    for gate in circuit.gates:
        qubits = set(tuple(gate.control_indices) + tuple(gate.target_indices))
        if (any_gates or gate.name in target_gates) and (
            any_qubits or qubits <= target_qubits
        ):
            count += 1
    return count


def qubit_couplings(
    circuit: NonParametricQuantumCircuit,
) -> Sequence[tuple[int, int]]:
    coupling = set()
    for gate in circuit.gates:
        qubits = tuple(gate.control_indices) + tuple(gate.target_indices)
        if len(qubits) < 2:
            continue
        coupling |= set(it.product(qubits, repeat=2))
    return cast(Sequence[tuple[int, int]], list(coupling))


def extract_qubit_path(circuit: NonParametricQuantumCircuit) -> Optional[Sequence[int]]:
    coupling = set(qubit_couplings(circuit))
    for a, b in coupling:
        coupling.add((b, a))
    graph = nx.parse_adjlist([f"{a} {b}" for a, b in coupling])
    if nx.is_eulerian(graph):
        return list(map(int, nx.eulerian_path(graph).nodes))
    return None
