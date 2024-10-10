from collections.abc import Collection
from typing import Optional

import networkx as nx

from quri_parts.backend.device import DeviceProperty, GateProperty, QubitProperty
from quri_parts.backend.units import TimeValue
from quri_parts.circuit import gate_names
from quri_parts.circuit.gate_names import GateNameType
from quri_parts.circuit.transpile import GateSetConversionTranspiler


def generate_device_property(
    qubit_count: int,
    native_gates: Collection[GateNameType],
    gate_error_1q: float,
    gate_error_2q: float,
    gate_error_meas: float,
    gate_time_1q: TimeValue,
    gate_time_2q: TimeValue,
    gate_time_meas: TimeValue,
    background_error: Optional[tuple[float, TimeValue]] = None,
) -> DeviceProperty:
    native_gate_set = set(native_gates)
    gates_1q = native_gate_set & gate_names.SINGLE_QUBIT_GATE_NAMES
    gates_2q = native_gate_set & gate_names.TWO_QUBIT_GATE_NAMES

    if gates_1q & gates_2q != native_gate_set:
        raise ValueError(
            "Only single and two qubit gates are supported as native gates"
        )

    qubits = list(range(qubit_count))
    qubit_properties = {q: QubitProperty() for q in qubits}
    gate_properties = [
        GateProperty(
            gate_names.Measurement,
            (),
            gate_error=gate_error_meas,
            gate_time=gate_time_meas,
        )
    ]
    gate_properties.extend(
        [
            GateProperty(name, (), gate_error=gate_error_1q, gate_time=gate_time_1q)
            for name in gates_1q
        ]
    )
    gate_properties.extend(
        [
            GateProperty(name, (), gate_error=gate_error_2q, gate_time=gate_time_2q)
            for name in gates_2q
        ]
    )

    return DeviceProperty(
        qubit_count=qubit_count,
        qubits=qubits,
        qubit_graph=nx.complete_graph(qubit_count),
        qubit_properties=qubit_properties,
        native_gates=native_gates,
        gate_properties=gate_properties,
        physical_qubit_count=qubit_count,
        background_error=background_error,
        transpiler=GateSetConversionTranspiler(native_gates),
    )
