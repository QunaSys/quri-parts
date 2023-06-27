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

CouplingMapWithCxErrors: TypeAlias = Mapping[tuple[int, int], float]


def qubit_counts_considering_cnot_errors(
    cnot_errors: CouplingMapWithCxErrors, cnot_error_threshold: float
) -> Sequence[int]:
    """Returns a list of the number of qubits in subgraphs consisting only of coupled
    qubits with cnot errors smaller than the specified threshold.
    """
    adjlist = [
        f"{a} {b}" for (a, b), e in cnot_errors.items() if e < cnot_error_threshold
    ]
    graph = nx.parse_adjlist(adjlist)
    return [len(c) for c in nx.connected_components(graph)]


def _sorted_undirected(
    cnot_errors: CouplingMapWithCxErrors,
) -> Sequence[tuple[int, int]]:
    ud = []
    for (a, b), e in sorted(cnot_errors.items()):
        if (a, b) not in ud and (b, a) not in ud:
            ud.append(((a, b), e))
    return next(zip(*sorted(ud, key=lambda x: x[1])))


def _directed(g: Sequence[tuple[int, int]]) -> Sequence[tuple[int, int]]:
    rs = set((b, a) for a, b in g)
    return list(rs | set(g))


def _list_to_graph(coupling_list: Sequence[tuple[int, int]]) -> nx.Graph:
    return nx.parse_adjlist([f"{a} {b}" for a, b in coupling_list])


def approx_cnot_reliable_subgraph(
    cnot_errors: CouplingMapWithCxErrors, qubit_count: int
) -> Sequence[nx.Graph]:
    """Returns a list of subgraphs with relatively small cnot errors that contain
    the specified number or more qubits.
    """
    sorted_edges = _sorted_undirected(cnot_errors)
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


def _approx_cnot_reliable_single_stroke_paths(
    cnot_errors: CouplingMapWithCxErrors, qubit_count: int
) -> Sequence[Sequence[int]]:
    sorted_edges = _sorted_undirected(cnot_errors)
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


def _path_fidelity(cnot_errors: CouplingMapWithCxErrors, path: Sequence[int]) -> float:
    return cast(float, np.prod([1 - cnot_errors[q] for q in zip(path, path[1:])]))


def cnot_reliable_single_stroke_path(
    cnot_errors: CouplingMapWithCxErrors,
    qubit_count: int,
    exact: bool = True,
) -> Sequence[int]:
    """Find the path with the specified number of qubits that has minimal CNOT error
    across the entire path. The cost is calculated by product of 1 - error. If no such
    path is found, an empty list is returned.

    Args:
        cnot_errors: A dictionary representing the coupling of qubits and their CNOT
            error rates.
        qubit_count: Number of qubits in the path.
        exact: Specify whether to search for the optimal solution. Approximate solutions
            can also be selected when the target graph is large and the computational
            cost is high.
    """
    if exact:
        ps = _length_satisfactory_paths(
            _list_to_graph(list(cnot_errors.keys())), qubit_count
        )
    else:
        ps = _approx_cnot_reliable_single_stroke_paths(cnot_errors, qubit_count)
    return max(ps, key=lambda p: _path_fidelity(cnot_errors, p)) if ps else []


class QubitMappingByCxErrorsTranspiler(CircuitTranspilerProtocol):
    """A CircuitTranspiler, that maps qubit indices so that the CNOT error is minimized
    across the entire entangling path.

    If all the qubits of a given circuit are entangled by 2 qubit gates and can be
    arranged in a row, the qubit indices of the circuit are mapped if a path with the
    required number of qubits is found from the qubit coupling map.

    Args:
        cnot_errors: A dictionary representing the coupling of qubits and their CNOT
            error rates.
        exact: Specify whether to search for the optimal solution. Approximate solutions
            can also be selected when the target graph is large and the computational
            cost is high.
    """

    def __init__(self, cnot_errors: CouplingMapWithCxErrors, exact: bool = True):
        self._cnot_errors = cnot_errors
        self._exact = exact

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        circuit_path = extract_qubit_coupling_path(circuit)
        if circuit_path is None or len(circuit_path) != circuit.qubit_count:
            raise ValueError(
                "Qubits in the given circuit is not sequentially entangled."
            )
        qubit_path = cnot_reliable_single_stroke_path(
            self._cnot_errors, circuit.qubit_count, self._exact
        )
        if not qubit_path:
            raise ValueError(
                "Cannot find single stroke path in coupling map for the given circuit."
            )
        qubit_mapping = {c: q for c, q in zip(circuit_path, qubit_path)}
        return QubitRemappingTranspiler(qubit_mapping)(circuit)
