from typing import Any, Optional, Union

import networkx as nx
import numpy as np
from qiskit.providers import BackendV1, BackendV2


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
    cx_errors: dict[tuple[int, int], float], cx_error_threshold: float
) -> list[int]:
    adjlist = []
    for (a, b), e in cx_errors.items():
        if e < cx_error_threshold:
            adjlist.append(f"{a} {b}")

    graph = nx.parse_adjlist(adjlist)
    return [len(c) for c in nx.connected_components(graph)]


def _undirected(g: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    rs = {}
    for (a, b), e in g.items():
        if (a, b) not in rs and (b, a) not in rs:
            rs[(a, b)] = e
    return rs


def _directed(g: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    rs = {}
    for (a, b), e in g.items():
        rs[(b, a)] = e
    return rs | g


def _list_to_graph(coupling_list: list[tuple[int, int]]) -> nx.Graph:
    adjlist = []
    for a, b in coupling_list:
        adjlist.append(f"{a} {b}")
    return nx.parse_adjlist(adjlist)


def approx_cx_reliable_subgraph(
    cx_errors: dict[tuple[int, int], float], qubits: int
) -> list[nx.Graph]:
    sorted_graph = sorted(_undirected(cx_errors).items(), key=lambda x: x[1])
    for i in range(qubits - 1, len(sorted_graph)):
        sg = _list_to_graph(list(_directed({k: v for k, v in sorted_graph[:i]}).keys()))
        rs = [sg.subgraph(c) for c in nx.connected_components(sg) if len(c) >= qubits]
        if rs:
            return rs
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
    sorted_graph = sorted(_undirected(cx_errors).items(), key=lambda x: x[1])
    for i in range(qubits - 1, len(sorted_graph)):
        sg = _list_to_graph(list(_directed({k: v for k, v in sorted_graph[:i]}).keys()))
        rs = [sg.subgraph(c) for c in nx.connected_components(sg) if len(c) >= qubits]
        ret = sum([_length_satisfactory_paths(r, qubits) for r in rs], [])
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
    return max(ps, key=lambda p: _path_fidelity(cx_errors, p))
