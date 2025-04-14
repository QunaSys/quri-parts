from collections.abc import Mapping
from typing import Union

import networkx as nx
from qiskit.providers import BackendV1, BackendV2, BackendV2Converter

from quri_parts.circuit import gate_names
from quri_parts.circuit.gate_names import GateNameType
from quri_parts.qiskit.circuit.gate_names import ECR, QiskitGateNameType


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


_qp_qiskit_gate_name_map: Mapping[Union[GateNameType, QiskitGateNameType], str] = {
    gate_names.CNOT: "cx",
    gate_names.CZ: "cz",
    ECR: "ecr",
}


def coupling_map_with_2_qubit_gate_errors(
    device: Union[BackendV1, BackendV2], gate_name: GateNameType = gate_names.CNOT
) -> Mapping[tuple[int, int], float]:
    """Extract qubit couplings and their 2 qubit gate error rates from Qiskit
    BackendV1 or BackendV2 instance."""
    if isinstance(device, BackendV1):
        device = BackendV2Converter(device)

    edges = device.coupling_map.get_edges()
    gate = _qp_qiskit_gate_name_map[gate_name]
    return {qs: prop.error for qs, prop in device.target[gate].items() if qs in edges}


def qubit_indices_with_readout_errors(
    device: Union[BackendV1, BackendV2]
) -> Mapping[int, float]:
    """Extract readout errors for each qubit from Qiskit BackendV1 or BackendV2
    instance."""
    if isinstance(device, BackendV1):
        device = BackendV2Converter(device)

    return {qs[0]: prop.error for qs, prop in device.target["measure"].items()}
