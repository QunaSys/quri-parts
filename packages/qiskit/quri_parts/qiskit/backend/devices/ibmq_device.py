from collections.abc import Mapping

import networkx as nx
from qiskit.providers import BackendV2

from quri_parts.backend.device import DeviceProperty, GateProperty, QubitProperty
from quri_parts.backend.units import TimeUnit, TimeValue
from quri_parts.circuit import gate_names
from quri_parts.circuit.gate_names import GateNameType
from quri_parts.qiskit.circuit.transpile import QiskitTranspiler

_op_gate_name_map: Mapping[str, GateNameType] = {
    "id": gate_names.Identity,
    "x": gate_names.X,
    "sx": gate_names.SqrtX,
    "rz": gate_names.RZ,
    "cz": gate_names.CZ,
    "cx": gate_names.CNOT,
    "ecr": gate_names.ECR,
}


def generate_device_property(backend: BackendV2) -> DeviceProperty:
    qubit_ps = ((i, backend.qubit_properties(i)) for i in range(backend.num_qubits))
    qubit_properties = {
        i: QubitProperty(
            T1=TimeValue(value=p.t1, unit=TimeUnit.SECOND),
            T2=TimeValue(value=p.t2, unit=TimeUnit.SECOND),
            frequency=p.frequency,
        )
        for i, p in qubit_ps
    }

    gate_properties = set()
    for op, ps in backend.target.items():
        gate = _op_gate_name_map[op]
        gate_properties.update(
            {
                GateProperty(
                    gate=gate,
                    qubits=qs,
                    gate_error=p.error,
                    gate_time=TimeValue(value=p.duration, unit=TimeUnit.SECOND),
                )
                for qs, p in ps.items()
                if p is not None
            }
        )

    return DeviceProperty(
        qubit_count=backend.num_qubits,
        qubits=list(qubit_properties.keys()),
        qubit_graph=nx.parse_edgelist(f"{a} {b}" for a, b in backend.coupling_map),
        qubit_properties=qubit_properties,
        native_gates=set(p.gate for p in gate_properties),
        gate_properties=gate_properties,
        physical_qubit_count=backend.num_qubits,
        background_error=None,
        transpiler=QiskitTranspiler(backend=backend),
    )
