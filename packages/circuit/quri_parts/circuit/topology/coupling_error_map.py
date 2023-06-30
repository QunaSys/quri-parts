import itertools as it
from collections.abc import Mapping, Sequence
from typing import cast

import networkx as nx
import numpy as np
from typing_extensions import TypeAlias

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.transpile import (
    CircuitTranspilerProtocol,
    QubitRemappingTranspiler,
    extract_qubit_coupling_path,
)

#: Represents qubit couplings and their CNOT error rates.
CouplingMapWithCxErrors: TypeAlias = Mapping[tuple[int, int], float]


def effectively_coupled_qubits_counts(
    two_qubit_errors: CouplingMapWithCxErrors, two_qubit_error_threshold: float
) -> Sequence[int]:
    """Returns a list of the number of qubits in subgraphs consisting only of
    coupled qubits with two qubit gate errors smaller than the specified
    threshold."""
    adjlist = [
        f"{a} {b}"
        for (a, b), e in two_qubit_errors.items()
        if e < two_qubit_error_threshold
    ]
    graph = nx.parse_adjlist(adjlist)
    return [len(c) for c in nx.connected_components(graph)]


def _sorted_undirected(
    two_qubit_errors: CouplingMapWithCxErrors,
) -> Sequence[tuple[int, int]]:
    ud = []
    for (a, b), e in sorted(two_qubit_errors.items()):
        if (a, b) not in ud and (b, a) not in ud:
            ud.append(((a, b), e))
    return next(zip(*sorted(ud, key=lambda x: x[1])))


def _directed(g: Sequence[tuple[int, int]]) -> Sequence[tuple[int, int]]:
    rs = set((b, a) for a, b in g)
    return list(rs | set(g))


def _list_to_graph(coupling_list: Sequence[tuple[int, int]]) -> nx.Graph:
    return nx.parse_adjlist([f"{a} {b}" for a, b in coupling_list])


def approx_reliable_coupling_subgraph(
    two_qubit_errors: CouplingMapWithCxErrors, qubit_count: int
) -> Sequence[nx.Graph]:
    """Returns a list of subgraphs with relatively small two qubit gate errors that
    contain the specified number or more qubits."""
    sorted_edges = _sorted_undirected(two_qubit_errors)
    for i in range(qubit_count - 1, len(sorted_edges)):
        best_nodes = _list_to_graph(_directed(sorted_edges[:i]))
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
    two_qubit_errors: CouplingMapWithCxErrors, qubit_count: int
) -> Sequence[Sequence[int]]:
    sorted_edges = _sorted_undirected(two_qubit_errors)
    for i in range(qubit_count - 1, len(sorted_edges)):
        best_nodes = _list_to_graph(_directed(sorted_edges[:i]))
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


def _path_fidelity(
    two_qubit_errors: CouplingMapWithCxErrors, path: Sequence[int]
) -> float:
    return cast(float, np.prod([1 - two_qubit_errors[q] for q in zip(path, path[1:])]))


def reliable_coupling_single_stroke_path(
    two_qubit_errors: CouplingMapWithCxErrors,
    qubit_count: int,
    exact: bool = True,
) -> Sequence[int]:
    """Find the path with the specified number of qubits that has maximum
    fidelity across the entire path considering 2 qubit gate errors. If no
    such path is found, an empty list is returned.

    Args:
        two_qubit_errors: A dictionary representing the coupling of qubits and their
            error rates.
        qubit_count: Number of qubits in the path.
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
    return max(ps, key=lambda p: _path_fidelity(two_qubit_errors, p)) if ps else []


class ReliableSingleStrokeCouplingPathQubitMappingTranspiler(CircuitTranspilerProtocol):
    """A CircuitTranspiler, that maps qubit indices so that the fidelity
    considering 2 qubit gate errors is maximized across the entire coupling path.

    If all the qubits of a given circuit are coupled by 2 qubit gates and can be
    arranged in a row, the qubit indices of the circuit are reallocated if a path with
    the required number of qubits is found from the qubit coupling map.

    Args:
        two_qubit_errors: A dictionary representing the coupling of qubits and their
            error rates.
        exact: Specify whether to search for the optimal solution. Approximate solutions
            can also be selected when the target graph is large and the computational
            cost is high.
    """

    def __init__(self, two_qubit_errors: CouplingMapWithCxErrors, exact: bool = True):
        self._two_qubit_errors = two_qubit_errors
        self._exact = exact

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        circuit_path = extract_qubit_coupling_path(circuit)
        if circuit_path is None or len(circuit_path) != circuit.qubit_count:
            raise ValueError("Qubits in the given circuit is not sequentially coupled.")
        qubit_path = reliable_coupling_single_stroke_path(
            self._two_qubit_errors, circuit.qubit_count, self._exact
        )
        if not qubit_path:
            raise ValueError(
                "Cannot find single stroke path in coupling map for the given circuit."
            )
        qubit_mapping = {c: q for c, q in zip(circuit_path, qubit_path)}
        return QubitRemappingTranspiler(qubit_mapping)(circuit)
