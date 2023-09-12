import itertools as it
from collections.abc import Mapping, Sequence
from typing import cast

import networkx as nx
import numpy as np
from typing_extensions import TypeAlias

from quri_parts.backend.qubit_mapping import BackendQubitMapping
from quri_parts.circuit import NonParametricQuantumCircuit, gate_names
from quri_parts.circuit.transpile import (
    extract_qubit_coupling_path,
    gate_count,
)

#: Represents qubit couplings and their interested 2 qubit gate error rates.
CouplingMapWithErrors: TypeAlias = Mapping[tuple[int, int], float]


def effectively_coupled_subgraph(
    two_qubit_errors: CouplingMapWithErrors, two_qubit_error_threshold: float
) -> Sequence[nx.Graph]:
    adjlist = [
        f"{a} {b}"
        for (a, b), e in two_qubit_errors.items()
        if e < two_qubit_error_threshold
    ]
    graph = nx.parse_adjlist(adjlist)
    return [graph.subgraph(c).copy() for c in nx.connected_components(graph)]


def effectively_coupled_qubit_counts(
    two_qubit_errors: CouplingMapWithErrors, two_qubit_error_threshold: float
) -> Sequence[int]:
    """Returns a list of the number of qubits in subgraphs consisting only of
    coupled qubits with two qubit gate errors smaller than the specified
    threshold."""
    return [
        len(c)
        for c in effectively_coupled_subgraph(
            two_qubit_errors, two_qubit_error_threshold
        )
    ]


def _sorted_undirected(
    two_qubit_errors: CouplingMapWithErrors,
) -> Sequence[tuple[int, int]]:
    ud = []
    for (a, b), e in sorted(two_qubit_errors.items()):
        if (b, a) not in ud:
            ud.append(((a, b), e))
    return next(zip(*sorted(ud, key=lambda x: x[1])))


def _list_to_graph(coupling_list: Sequence[tuple[int, int]]) -> nx.Graph:
    return nx.parse_adjlist([f"{a} {b}" for a, b in coupling_list])


def approx_reliable_coupling_subgraph(
    two_qubit_errors: CouplingMapWithErrors, qubit_count: int
) -> Sequence[nx.Graph]:
    """Returns a list of subgraphs with relatively small two qubit gate errors
    that contain the specified number or more qubits."""
    sorted_edges = _sorted_undirected(two_qubit_errors)
    for i in range(qubit_count - 1, len(sorted_edges)):
        best_nodes = _list_to_graph(sorted_edges[:i])
        enough_nodes = [
            best_nodes.subgraph(c)
            for c in nx.connected_components(best_nodes)
            if len(c) >= qubit_count
        ]
        if enough_nodes:
            return enough_nodes
    return []


def graph_to_coupling_list(graph: nx.Graph) -> Sequence[tuple[int, ...]]:
    """Convert networkx Graph (contains integer name nodes) to Sequence of
    integer tuples."""
    return list(map(lambda qs: tuple(map(int, qs)), graph.edges))


def _length_satisfactory_paths(
    graph: nx.Graph, qubit_count: int
) -> Sequence[Sequence[int]]:
    nodes = list(graph.nodes)
    ret = []
    for s in nodes:
        for e in nodes:
            ps = nx.all_simple_paths(graph, s, e)
            ret.extend([list(map(int, p)) for p in ps if len(p) == qubit_count])
    return ret


def _approx_reliable_coupling_single_stroke_paths(
    two_qubit_errors: CouplingMapWithErrors, qubit_count: int
) -> Sequence[Sequence[int]]:
    sorted_edges = _sorted_undirected(two_qubit_errors)
    for i in range(qubit_count - 1, len(sorted_edges)):
        best_nodes = _list_to_graph(sorted_edges[:i])
        enough_nodes = [
            best_nodes.subgraph(c)
            for c in nx.connected_components(best_nodes)
            if len(c) >= qubit_count
        ]
        ret = list(
            it.chain.from_iterable(
                [_length_satisfactory_paths(g, qubit_count) for g in enough_nodes]
            )
        )
        if ret:
            return ret
    return []


def _two_qubit_path_fidelity(
    two_qubit_errors: CouplingMapWithErrors, path: Sequence[int]
) -> float:
    return cast(
        float, np.prod([1.0 - two_qubit_errors[q] for q in zip(path, path[1:])])
    )


def _circuit_fidelity_for_repetitive_ansatz(
    two_qubit_errors: CouplingMapWithErrors,
    readout_errors: Mapping[int, float],
    reps: int,
    path: Sequence[int],
) -> float:
    two_qubit_1_line = _two_qubit_path_fidelity(two_qubit_errors, path)
    readout = cast(float, np.prod([1.0 - readout_errors[q] for q in path]))
    return two_qubit_1_line**reps * readout


def circuit_fidelity_by_two_qubit_gate_and_readout_errors(
    two_qubit_errors: CouplingMapWithErrors,
    readout_errors: Mapping[int, float],
    circuit: NonParametricQuantumCircuit,
    default_two_qubit_error: float = 0.0,
    default_readout_error: float = 0.0,
) -> float:
    fidelity = 1.0
    qubits = set()

    for gate in circuit.gates:
        qs = tuple(gate.control_indices) + tuple(gate.target_indices)
        qubits |= set(qs)
        if len(qs) != 2:
            continue
        error = two_qubit_errors.get(cast(tuple[int, int], qs), default_two_qubit_error)
        fidelity *= 1.0 - error

    readout = cast(
        float,
        np.prod([1.0 - readout_errors.get(q, default_readout_error) for q in qubits]),
    )

    return fidelity * readout


def reliable_coupling_single_stroke_path(
    two_qubit_errors: CouplingMapWithErrors,
    readout_errors: Mapping[int, float],
    qubit_count: int,
    reps: int = 1,
    exact: bool = True,
) -> Sequence[int]:
    """Find the path with the specified number of qubits that has maximum
    fidelity across the entire path considering 2 qubit gate errors. If no such
    path is found, an empty list is returned.

    Args:
        two_qubit_errors: Mapping representing the coupling of qubits and their
            error rates.
        readout_errors: Mapping from qubits to their readout errors.
        qubit_count: Required number of qubits in the path.
        reps: Number of repeated 2 qubit gate layers of the circuit.
        exact: Specify whether to search for the optimal solution. Approximate solutions
            can also be selected when the target graph is large and the computational
            cost is high.
    """
    if exact:
        ps = _length_satisfactory_paths(
            _list_to_graph(list(two_qubit_errors.keys())), qubit_count
        )
    else:
        ps = _approx_reliable_coupling_single_stroke_paths(
            two_qubit_errors, qubit_count
        )
    return (
        max(
            ps,
            key=lambda p: _circuit_fidelity_for_repetitive_ansatz(
                two_qubit_errors, readout_errors, reps, p
            ),
        )
        if ps
        else []
    )


def reliable_coupling_single_stroke_path_qubit_mapping(
    circuit: NonParametricQuantumCircuit,
    two_qubit_errors: CouplingMapWithErrors,
    readout_errors: Mapping[int, float],
    exact: bool = True,
) -> BackendQubitMapping:
    """BackendQubitMapping generator, that maps qubit indices so that the
    fidelity considering 2 qubit gate errors is maximized across the entire
    coupling path.

    When a single path containing all the qubits in the circuit can be found
    from the graph with the coupled qubits in the circuit as edges, the qubit
    indices of the circuit are reallocated if a path with the required number
    of qubits is found from the qubit coupling map of the device.

    Args:
        circuit: Target NonParametricQuantumCircuit.
        two_qubit_errors: Mapping representing the couplings of qubits and their
            error rates.
        readout_errors: Mapping from qubits to their readout_errors.
        exact: Specify whether to search for the optimal solution. Approximate solutions
            can also be selected when the target graph is large and the computational
            cost is high.
    """
    paths = extract_qubit_coupling_path(circuit)
    if (
        len(paths) == 2
        and paths[0] == list(reversed(paths[1]))
        and len(paths[0]) == circuit.qubit_count
    ):
        circuit_path = sorted(paths)[0]
    else:
        raise ValueError("Qubits in the given circuit is not sequentially coupled.")
    two_qubit_gate_count_for_each_qubit = max(
        gate_count(
            circuit,
            gate_names={gate_names.CNOT, gate_names.CZ, gate_names.SWAP},
            qubit_indices={q},
        )
        for q in range(circuit.qubit_count)
    )
    qubit_path = reliable_coupling_single_stroke_path(
        two_qubit_errors,
        readout_errors,
        qubit_count=circuit.qubit_count,
        reps=two_qubit_gate_count_for_each_qubit // 2,
        exact=exact,
    )
    if not qubit_path:
        raise ValueError(
            "Cannot find single stroke path in coupling map for the given circuit."
        )
    qubit_mapping = {c: q for c, q in zip(circuit_path, qubit_path)}
    return BackendQubitMapping(qubit_mapping)
