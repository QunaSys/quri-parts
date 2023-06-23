import math
from typing import Any, Optional, Union

import networkx as nx
import numpy as np
from qiskit.providers import BackendV1, BackendV2
from z3 import And, Bool, If, Implies, Int, IntVector, Optimize, Or, sat


def device_connectivity_graph(device: Union[BackendV1, BackendV2]) -> nx.Graph:
    if isinstance(device, BackendV1):
        config = device.configuration()
        coupling_map = getattr(config, "coupling_map", None)

        if coupling_map is None:
            adj_list = None
        else:
            adj_list = coupling_map.get_edges()

        num_qubits = getattr(config, "num_qubits", -1)

    else:
        coupling_map = device.coupling_map
        adj_list = coupling_map.get_edges()
        num_qubits = device.num_qubits

    if adj_list is None:
        raise ValueError("Given device does not have a coupling map.")

    if num_qubits == -1:
        raise ValueError("Number of qubits not specified by the backend.")

    lines = [f"{a} {b}" for (a, b) in adj_list]
    lines.extend([str(a) for a in range(num_qubits)])

    return nx.parse_adjlist(lines)


def coupling_map_with_cx_errors(device: BackendV2) -> dict[tuple[int, int], float]:
    coupling_map = device.coupling_map
    two_q_error_map: dict[tuple[Any, ...], Optional[float]] = {}
    cx_errors = {}
    for gate, prop_dict in device.target.items():
        if prop_dict is None or None in prop_dict:
            continue
        for qargs, inst_props in prop_dict.items():
            if inst_props is None:
                continue
            if len(qargs) == 2:
                if inst_props.error is not None:
                    two_q_error_map[qargs] = max(
                        two_q_error_map.get(qargs, 0), inst_props.error
                    )
    if coupling_map:
        for line in coupling_map.get_edges():
            err = two_q_error_map.get(tuple(line), 0)
            cx_errors[tuple(line)] = err
    return cx_errors


def qubit_counts_considering_cx_errors(
    device: BackendV2, cx_error_threshold: float
) -> list[int]:
    cx_errors = coupling_map_with_cx_errors(device)
    adjlist = []
    for (a, b), e in cx_errors.items():
        if e < cx_error_threshold:
            adjlist.append(f"{a} {b}")

    graph = nx.parse_adjlist(adjlist)
    return [len(c) for c in nx.connected_components(graph)]


def _undirected(g):
    rs = {}
    for (a, b), e in g.items():
        if (a, b) not in rs and (b, a) not in rs:
            rs[(a, b)] = e
    return rs


def _directed(g):
    rs = {}
    for (a, b), e in g.items():
        rs[(b, a)] = e
    return rs | g


def _list_to_graph(gl: list[tuple[int, int], float]) -> nx.Graph:
    g = _directed({k: v for k, v in gl})
    adjlist = []
    for (a, b), _ in g.items():
        adjlist.append(f"{a} {b}")
    return nx.parse_adjlist(adjlist)


def cx_reliable_subgraph(device: BackendV2, qubits: int) -> list[nx.Graph]:
    cx_errors = coupling_map_with_cx_errors(device)

    sorted_graph = sorted(_undirected(cx_errors).items(), key=lambda x: x[1])
    for i in range(qubits - 1, len(sorted_graph)):
        sg = _list_to_graph(sorted_graph[:i])
        rs = [sg.subgraph(c) for c in nx.connected_components(sg) if len(c) >= qubits]
        if rs:
            return rs
    return []


def _satisfactory_paths(graph: nx.Graph, qubits: int) -> list[list[str]]:
    nodes = list(graph.nodes)
    ret = []
    for s, e in zip(nodes, nodes[1:]):
        ps = nx.all_simple_paths(graph, s, e)
        ret.extend([p for p in ps if len(p) >= qubits])
    return ret


def _cx_reliable_single_stroke_paths(
    cx_errors: dict[tuple[int, int], float], qubits: int
) -> list[list[str]]:
    sorted_graph = sorted(_undirected(cx_errors).items(), key=lambda x: x[1])
    for i in range(qubits - 1, len(sorted_graph)):
        sg = _list_to_graph(sorted_graph[:i])
        rs = [sg.subgraph(c) for c in nx.connected_components(sg) if len(c) >= qubits]
        print(f"{i}: {[len(c) for c in nx.connected_components(sg)]}")
        ret = []
        for r in rs:
            sp = _satisfactory_paths(r, qubits)
            ret.extend(sp)
        if ret:
            return ret
    return []


def _path_fidelity(cx_errors: dict[tuple[int, int], float], path) -> float:
    return np.prod([1 - cx_errors[q] for q in zip(path, path[1:])])


def cx_reliable_single_stroke_path(device: BackendV2, qubits: int) -> list[int]:
    cx_errors = coupling_map_with_cx_errors(device)
    ps = _cx_reliable_single_stroke_paths(cx_errors, qubits)
    ret = []
    fc = 0.0
    for p in ps:
        path = list(map(int, p))
        f = _path_fidelity(cx_errors, path)
        if f > fc:
            fc = f
            ret = path
    return ret


def optimized_single_stroke_subgraph(
    graph: dict[tuple[int, int], float], qubits: int
) -> Optional[nx.Graph]:
    s = Optimize()

    n = len(set(sum(graph.keys(), ())))
    edge_count = qubits - 1

    adj_mat = [[Int("adj[%d,%d]" % (i, j)) for j in range(n)] for i in range(n)]
    cost = 0.0
    for i in range(n):
        for j in range(n):
            s.add(Implies(adj_mat[i][j] == 1, (i, j) in graph))
            s.add(If(adj_mat[i][j] == 1, adj_mat[j][i] == 1, adj_mat[j][i] == 0))
            s.add(Or(adj_mat[i][j] == 1, adj_mat[i][j] == 0))
            cost = cost + If(adj_mat[i][j] == 1, graph.get((i, j), 0.0), 0.0)

    deg1 = IntVector("deg1", n)
    deg2 = IntVector("deg2", n)
    for i in range(n):
        s.add(sum(adj_mat[i]) <= 2)
        s.add(If(sum(adj_mat[i]) == 1, deg1[i] == 1, deg1[i] == 0))
        s.add(If(sum(adj_mat[i]) == 2, deg2[i] == 1, deg2[i] == 0))
    s.add(sum(deg1) == 2)
    s.add(sum(deg2) == edge_count - 1)

    path = [[Bool("path[%d,%d]" % (i, j)) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            path[i][j] = False
        path[i][i] = True
    for k in range(n):
        for i in range(n):
            for j in range(n):
                path[i][j] = Or(
                    path[j][i],
                    path[i][j],
                    And(path[i][k], Or(adj_mat[k][j] == 1, adj_mat[j][k] == 1)),
                )
    for i in range(n):
        for j in range(n):
            s.add(
                Implies(
                    And(
                        Or(deg1[i] == 1, deg2[i] == 1),
                        Or(deg1[j] == 1, deg2[j] == 1),
                    ),
                    path[i][j],
                )
            )

    s.minimize(cost)

    r = s.check()
    if r != sat:
        # raise RuntimeError("No subgraphs were found by SAT that met the criteria.")
        return None
    m = s.model()
    adj_list = []
    for i in range(n):
        for j in range(n):
            if m[adj_mat[i][j]] == 1:
                adj_list.append(f"{i} {j}")
    return nx.parse_adjlist(adj_list)


def optimized_single_stroke_path(
    device: BackendV2, qubits: int
) -> Optional[list[tuple[int, int]]]:
    cx_errors = coupling_map_with_cx_errors(device)

    cx_lnfidelity = {}
    for q, e in cx_errors.items():
        cx_lnfidelity[q] = -math.log(1.0 - e)

    graph = optimized_single_stroke_subgraph(cx_lnfidelity, qubits)
    if graph is None:
        return None

    path = [(int(a), int(b)) for a, b in nx.eulerian_path(graph)]
    return {q: cx_errors[q] for q in path}
