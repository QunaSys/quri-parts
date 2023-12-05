from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from itertools import combinations

import networkx as nx

from quri_parts.circuit import NonParametricQuantumCircuit


def gate_weighted_depth(
    circuit: NonParametricQuantumCircuit,
    weights: Mapping[str, int],
    default_weight: int = 0,
) -> int:
    """Calculate circuit depth by setting different weights for each gate type.

    Args:
        circuit: Target NonParametricQuantumCircuit.
        weights: Mapping from gate names to their weights.
        default_weight: Gate weights not included in the weights argument.
            If the argument is omitted, 0 is used by default.
    """
    qubit_depths: dict[int, int] = defaultdict()
    for gate in circuit.gates:
        qubits = tuple(gate.control_indices) + tuple(gate.target_indices)
        depth = weights.get(gate.name, default_weight) + max(
            qubit_depths.get(q, 0) for q in qubits
        )
        for q in qubits:
            qubit_depths[q] = depth
    return max(qubit_depths.values()) if qubit_depths else 0


def gate_count(
    circuit: NonParametricQuantumCircuit,
    gate_names: Collection[str] = (),
    qubit_indices: Collection[int] = (),
    qubit_counts: Collection[int] = (),
) -> int:
    """Count the number of gates that satisfy the given qubit indices and gate
    names.

    Args:
        circuit: Target NonParametricQuantumCircuit.
        gate_names: Specify the target gate names. If omitted, all gate types
            are covered.
        qubit_indices: Specify qubit indices to which the target gates act on.
            If any one of the qubits matches, the gate is counted. If omitted,
            all qubits are covered.
        qubit_counts: Specify number of qubits of the target gates. If omitted,
            all number of qubits are covered.
    """
    count = 0
    for gate in circuit.gates:
        qubit_count = len(gate.control_indices) + len(gate.target_indices)
        satisfy_qubit_indices = any(
            q in qubit_indices for q in (*gate.control_indices, *gate.target_indices)
        )
        if (
            (not gate_names or gate.name in gate_names)
            and (not qubit_indices or satisfy_qubit_indices)
            and (not qubit_counts or qubit_count in qubit_counts)
        ):
            count += 1
    return count


def qubit_couplings(
    circuit: NonParametricQuantumCircuit,
    include_singles: bool = False,
) -> Sequence[tuple[int, ...]]:
    """Returns the set of tuples of the qubit indices of the multiple qubit
    gates in the circuit."""
    coupling = set()
    for gate in circuit.gates:
        qubits = tuple(gate.control_indices) + tuple(gate.target_indices)
        if include_singles or len(qubits) >= 2:
            coupling.add(qubits)
    return sorted(list(coupling))


def coupled_qubit_graphs(circuit: NonParametricQuantumCircuit) -> Sequence[nx.Graph]:
    """Returns a Sequence of each connected graphs with the coupled qubits as
    edges."""
    adjlist = []
    for qs in qubit_couplings(circuit):
        for pair in combinations(qs, 2):
            adjlist.append(" ".join(map(str, pair)))
    graph = nx.parse_adjlist(adjlist)
    return [graph.subgraph(c).copy() for c in nx.connected_components(graph)]


def coupled_qubit_indices(
    circuit: NonParametricQuantumCircuit,
) -> Sequence[Sequence[int]]:
    """Returns a Sequence of qubit indices of each connected graphs with the
    coupled qubits as edges."""
    return [list(map(int, g.nodes)) for g in coupled_qubit_graphs(circuit)]


def extract_qubit_coupling_path(
    circuit: NonParametricQuantumCircuit,
) -> list[list[int]]:
    """Returns paths containing all the qubits from the graph with the coupled
    qubits in the circuit as edges."""
    couplings = set(qubit_couplings(circuit))
    if any(len(qs) > 2 for qs in couplings):
        raise ValueError("The given circuit contains a more than 2 qubits gate.")
    graph = nx.parse_adjlist([f"{a} {b}" for a, b in couplings])
    paths = []
    nodes = set(graph.nodes)
    for s in nodes:
        for e in nodes - {s}:
            ps = nx.all_simple_paths(graph, s, e)
            paths.extend([list(map(int, p)) for p in ps if len(p) == len(nodes)])
    return [list(map(int, path)) for path in paths]
