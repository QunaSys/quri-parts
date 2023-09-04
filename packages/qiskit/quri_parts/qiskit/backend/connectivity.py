from collections.abc import Mapping
from typing import Union, cast

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


def coupling_map_with_2_qubit_gate_errors(
    device: Union[BackendV1, BackendV2], gate_name: str = "cx"
) -> Mapping[tuple[int, int], float]:
    """Extract qubit couplings and their 2 qubit gate error rates from Qiskit
    BackendV1 or BackendV2 instance."""
    if isinstance(device, BackendV2):
        edges = device.coupling_map.get_edges()
        return {
            qs: prop.error
            for qs, prop in device.target[gate_name].items()
            if qs in edges
        }
    elif isinstance(device, BackendV1):
        cmap = device.configuration().coupling_map
        props = device.properties().to_dict()
        gates = [gate for gate in props["gates"] if gate["gate"] == gate_name]
        return {
            cast(tuple[int, int], tuple(gate["qubits"])): gate["parameters"][0]["value"]
            for gate in gates
            if gate["qubits"] in cmap
        }
    else:
        raise ValueError("Unsupported device.")


def qubit_indices_with_readout_errors(
    device: Union[BackendV1, BackendV2]
) -> Mapping[int, float]:
    """Extract readout errors for each qubit from Qiskit BackendV1 or BackendV2
    instance."""
    if isinstance(device, BackendV2):
        return {qs[0]: prop.error for qs, prop in device.target["measure"].items()}
    elif isinstance(device, BackendV1):
        props = device.properties()
        return {
            q: props.readout_error(q) for q in range(device.configuration().n_qubits)
        }
    else:
        raise ValueError("Unsupported device.")
