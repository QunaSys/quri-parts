import networkx as nx
import numpy as np


def qubit_counts_considering_cx_errors(
    cx_errors: dict[tuple[int, int], float], cx_error_threshold: float
) -> list[int]:
    adjlist = [f"{a} {b}" for (a, b), e in cx_errors.items() if e < cx_error_threshold]
    graph = nx.parse_adjlist(adjlist)
    return [len(c) for c in nx.connected_components(graph)]


def _sorted_undirected(
    cx_errors: dict[tuple[int, int], float]
) -> list[tuple[int, int]]:
    ud = []
    for (a, b), e in cx_errors.items():
        if (a, b) not in ud and (b, a) not in ud:
            ud.append(((a, b), e))
    return next(zip(*sorted(ud, key=lambda x: x[1])))


def _directed(g: list[tuple[int, int]]) -> dict[tuple[int, int], float]:
    rs = set((b, a) for a, b in g)
    return rs | set(g)


def _list_to_graph(coupling_list: list[tuple[int, int]]) -> nx.Graph:
    return nx.parse_adjlist([f"{a} {b}" for a, b in coupling_list])


def approx_cx_reliable_subgraph(
    cx_errors: dict[tuple[int, int], float], qubits: int
) -> list[nx.Graph]:
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


def _length_satisfactory_paths(graph: nx.Graph, qubits: int) -> list[list[int]]:
    nodes = list(graph.nodes)
    ret = []
    for s in nodes:
        for e in nodes:
            ps = nx.all_simple_paths(graph, s, e)
            ret.extend([list(map(int, p)) for p in ps if len(p) == qubits])
    return ret


def _approx_cx_reliable_single_stroke_paths(
    cx_errors: dict[tuple[int, int], float], qubits: int
) -> list[list[int]]:
    sorted_edges = _sorted_undirected(cx_errors)
    for i in range(qubits - 1, len(sorted_edges)):
        best_nodes = _list_to_graph(_directed(sorted_edges[:i]))
        enough_nodes = [
            best_nodes.subgraph(c)
            for c in nx.connected_components(best_nodes)
            if len(c) >= qubits
        ]
        ret = sum([_length_satisfactory_paths(g, qubits) for g in enough_nodes], [])
        if ret:
            return ret
    return []


def _path_fidelity(cx_errors: dict[tuple[int, int], float], path: list[int]) -> float:
    return np.prod([1 - cx_errors[q] for q in zip(path, path[1:])])


def cx_reliable_single_stroke_path(
    cx_errors: dict[tuple[int, int], float],
    qubits: int,
    exact: bool = True,
) -> list[int]:
    if exact:
        ps = _length_satisfactory_paths(_list_to_graph(list(cx_errors.keys())), qubits)
    else:
        ps = _approx_cx_reliable_single_stroke_paths(cx_errors, qubits)
    return max(ps, key=lambda p: _path_fidelity(cx_errors, p)) if ps else []
