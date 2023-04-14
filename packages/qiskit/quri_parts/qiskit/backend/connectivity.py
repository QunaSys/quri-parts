import networkx as nx
from qiskit.providers.ibmq import IBMQBackend


def device_connectivity_graph(device: IBMQBackend) -> nx.Graph:
    config = device.configuration()
    adj_list = getattr(config, "coupling_map", None)
    num_qubits = getattr(config, "num_qubits", -1)

    if adj_list is None:
        raise ValueError("Given device does not have a coupling map.")

    if num_qubits == -1:
        raise ValueError("Number of qubits not specified by the backend.")

    config = device.configuration()
    adj_list = config.coupling_map
    num_qubits = config.num_qubits
    lines = [f"{a} {b}" for (a, b) in adj_list]
    lines.extend([str(a) for a in range(num_qubits)])

    return nx.parse_adjlist(lines)
