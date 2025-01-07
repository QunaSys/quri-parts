import networkx as nx
from braket.aws import AwsDevice

from quri_parts.backend.device import DeviceProperty, GateProperty, QubitProperty
from quri_parts.backend.units import TimeUnit, TimeValue
from quri_parts.circuit import gate_names
from quri_parts.ionq.circuit.transpile import IonQSetTranspiler


def generate_device_property(device: AwsDevice) -> DeviceProperty:
    qubit_count = device.properties.paradigm.qubitCount
    timing = device.properties.provider.timing
    fidelity = device.properties.provider.fidelity

    qubit_properties = {
        i: QubitProperty(
            T1=TimeValue(value=timing["T1"], unit=TimeUnit.SECOND),
            T2=TimeValue(value=timing["T2"], unit=TimeUnit.SECOND),
            frequency=None,
        )
        for i in range(qubit_count)
    }

    gate_properties = {
        GateProperty(
            gate=gate_names.Measurement,
            qubits=(),
            gate_error=1.0 - fidelity["spam"]["mean"],
            gate_time=TimeValue(value=timing["readout"], unit=TimeUnit.SECOND),
        ),
        GateProperty(
            gate=gate_names.GPi,
            qubits=(),
            gate_error=1.0 - fidelity["1Q"]["mean"],
            gate_time=TimeValue(value=timing["1Q"], unit=TimeUnit.SECOND),
        ),
        GateProperty(
            gate=gate_names.GPi2,
            qubits=(),
            gate_error=1.0 - fidelity["1Q"]["mean"],
            gate_time=TimeValue(value=timing["1Q"], unit=TimeUnit.SECOND),
        ),
        GateProperty(
            gate=gate_names.MS,
            qubits=(),
            gate_error=1.0 - fidelity["2Q"]["mean"],
            gate_time=TimeValue(value=timing["2Q"], unit=TimeUnit.SECOND),
        ),
    }

    return DeviceProperty(
        qubit_count=qubit_count,
        qubits=list(qubit_properties.keys()),
        qubit_graph=nx.complete_graph(qubit_count),
        qubit_properties=qubit_properties,
        native_gates=set(p.gate for p in gate_properties),
        gate_properties=gate_properties,
        physical_qubit_count=qubit_count,
        background_error=None,
        transpiler=IonQSetTranspiler(),
    )
