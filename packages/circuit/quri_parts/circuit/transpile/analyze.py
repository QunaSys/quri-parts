import itertools as it
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Optional

import networkx as nx

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.gate_names import NonParametricGateNameType


def gate_weighted_depth(
    circuit: NonParametricQuantumCircuit,
    weights: Mapping[NonParametricGateNameType, int],
    default_weight: int = 1,
) -> int:
    qubit_depths: dict[int, int] = defaultdict()
    for gate in circuit.gates:
        qubits = tuple(gate.control_indices) + tuple(gate.target_indices)
        depth = weights.get(gate.name, default_weight) + max(
            qubit_depths[q] for q in qubits
        )
    for q in qubits:
        qubit_depths[q] = depth
    return max(qubit_depths.values()) if qubit_depths else 0


def gate_count(
    circuit: NonParametricQuantumCircuit,
    qubit_indices: Sequence[int] = (),
    target_gates: Sequence[NonParametricGateNameType] = (),
) -> int:
    any_qubits, any_gates = not qubit_indices, not target_gates
    count = 0
    target_qubits = set(qubit_indices)
    target_gates = set(target_gates)
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
    for gate in circuit:
        qubits = tuple(gate.control_indices) + tuple(gate.target_indices)
        if len(qubits) < 2:
            continue
        coupling |= set(it.product(qubits, repeat=2))
    return list(coupling)


def extract_qubit_path(circuit: NonParametricQuantumCircuit) -> Optional[Sequence[int]]:
    coupling = set(qubit_couplings(circuit))
    for a, b in coupling:
        coupling.add((b, a))
    graph = nx.parse_adjlist([f"{a} {b}" for a, b in coupling])
    if nx.is_eulerian(graph):
        return map(int, nx.eulerian_path(graph).nodes)
    return None
