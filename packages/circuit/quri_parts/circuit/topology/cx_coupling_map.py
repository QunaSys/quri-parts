import itertools as it
from collections.abc import Mapping, Sequence
from typing import TypeAlias, cast

import networkx as nx
import numpy as np

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.transpile import (
    CircuitTranspilerProtocol,
    QubitRemappingTranspiler,
    extract_qubit_coupling_path,
)

CouplingMapWithCxErrors: TypeAlias = Mapping[tuple[int, int], float]


def qubit_counts_considering_cx_errors(
    cx_errors: CouplingMapWithCxErrors, cx_error_threshold: float
) -> Sequence[int]:
    adjlist = [f"{a} {b}" for (a, b), e in cx_errors.items() if e < cx_error_threshold]
    graph = nx.parse_adjlist(adjlist)
    return [len(c) for c in nx.connected_components(graph)]


def _sorted_undirected(cx_errors: CouplingMapWithCxErrors) -> Sequence[tuple[int, int]]:
    ud = []
    for (a, b), e in cx_errors.items():
        if (a, b) not in ud and (b, a) not in ud:
            ud.append(((a, b), e))
    return next(zip(*sorted(ud, key=lambda x: x[1])))


def _directed(g: Sequence[tuple[int, int]]) -> Sequence[tuple[int, int]]:
    rs = set((b, a) for a, b in g)
    return list(rs | set(g))


def _list_to_graph(coupling_list: Sequence[tuple[int, int]]) -> nx.Graph:
    return nx.parse_adjlist([f"{a} {b}" for a, b in coupling_list])


def approx_cx_reliable_subgraph(
    cx_errors: CouplingMapWithCxErrors, qubits: int
) -> Sequence[nx.Graph]:
    sorted_edges = _sorted_undirected(cx_errors)
    for i in range(qubits - 1, len(sorted_edges)):
        best_nodes = _list_to_graph(_directed(sorted_edges[:i]))
        enough_nodes = [
            best_nodes.subgraph(c)
            for c in nx.connected_components(best_nodes)
            if len(c) >= qubits
        ]
        if enough_nodes:
            return enough_nodes
    return []


def _length_satisfactory_paths(graph: nx.Graph, qubits: int) -> Sequence[Sequence[int]]:
    nodes = list(graph.nodes)
    ret = []
    for s in nodes:
        for e in nodes:
            ps = nx.all_simple_paths(graph, s, e)
            ret.extend([list(map(int, p)) for p in ps if len(p) == qubits])
    return ret


def _approx_cx_reliable_single_stroke_paths(
    cx_errors: CouplingMapWithCxErrors, qubits: int
) -> Sequence[Sequence[int]]:
    sorted_edges = _sorted_undirected(cx_errors)
    for i in range(qubits - 1, len(sorted_edges)):
        best_nodes = _list_to_graph(_directed(sorted_edges[:i]))
        enough_nodes = [
            best_nodes.subgraph(c)
            for c in nx.connected_components(best_nodes)
            if len(c) >= qubits
        ]
        ret = list(
            it.chain.from_iterable(
                [_length_satisfactory_paths(g, qubits) for g in enough_nodes]
            )
        )
        if ret:
            return ret
    return []


def _path_fidelity(cx_errors: CouplingMapWithCxErrors, path: Sequence[int]) -> float:
    return cast(float, np.prod([1 - cx_errors[q] for q in zip(path, path[1:])]))


def cx_reliable_single_stroke_path(
    cx_errors: CouplingMapWithCxErrors,
    qubits: int,
    exact: bool = True,
) -> Sequence[int]:
    if exact:
        ps = _length_satisfactory_paths(_list_to_graph(list(cx_errors.keys())), qubits)
    else:
        ps = _approx_cx_reliable_single_stroke_paths(cx_errors, qubits)
    return max(ps, key=lambda p: _path_fidelity(cx_errors, p)) if ps else []


class QubitMappingByCxErrorsTranspiler(CircuitTranspilerProtocol):
    def __init__(self, cx_errors: CouplingMapWithCxErrors, exact: bool = True):
        self._cx_errors = cx_errors
        self._exact = exact

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        circuit_path = extract_qubit_coupling_path(circuit)
        if circuit_path is None:
            raise ValueError(
                "Qubits in the given circuit is not sequentially entangled."
            )
        qubit_path = cx_reliable_single_stroke_path(
            self._cx_errors, circuit.qubit_count, self._exact
        )
        if not qubit_path:
            raise ValueError(
                "Cannot find single stroke path in coupling map for the given circuit."
            )
        qubit_mapping = {c: q for c, q in zip(circuit_path, qubit_path)}
        return QubitRemappingTranspiler(qubit_mapping)(circuit)
