from typing import Union

import numpy as np
import networkx as nx
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


def qubit_counts_considering_cx_errors(device: BackendV2, epsilon: float) -> list[int]:
    coupling_map = device.coupling_map
    two_q_error_map = {}
    cx_errors = []
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
            cx_errors.append(err)

    errors = np.array(cx_errors)
    es = errors < epsilon
    adjlist = []
    for c, e in zip(coupling_map, es):
        if e:
            a, b = c
            adjlist.append(f"{a} {b}")

    graph = nx.parse_adjlist(adjlist)
    return [len(c) for c in nx.connected_components(graph)]
