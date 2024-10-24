from collections.abc import Collection
from typing import Optional

import networkx as nx

from quri_parts.backend.device import DeviceProperty, GateProperty, QubitProperty
from quri_parts.backend.units import TimeValue
from quri_parts.circuit import gate_names
from quri_parts.circuit.gate_names import GateNameType
from quri_parts.circuit.topology import (
    SquareLattice,
    SquareLatticeSWAPInsertionTranspiler,
)
from quri_parts.circuit.transpile import (
    GateSetConversionTranspiler,
    SequentialTranspiler,
)


def generate_device_property(
    lattice: SquareLattice,
    native_gates: Collection[GateNameType],
    gate_error_1q: float,
    gate_error_2q: float,
    gate_error_meas: float,
    gate_time_1q: TimeValue,
    gate_time_2q: TimeValue,
    gate_time_meas: TimeValue,
    t1: Optional[TimeValue] = None,
    t2: Optional[TimeValue] = None,
) -> DeviceProperty:
    """Generate DeviceProperty object for a typical NISQ superconducting qubit
    device.

    Assumes that the device's qubits are connected as square lattice with no defects
    and that a subset of the gates natively supported by QURI Parts can be used as
    the native gates.

    Args:
        lattice: SquareLattice instance representing the device qubits connectivity.
        native_gates: Native gates supported by the device.
        gate_error_1q: Error rate of single qubit gate operations.
        gate_error_2q: Error rate of two qubit gate operations.
        gate_error_meas: Error rate of readout operations.
        gate_time_1q: Latency of single qubit gate operations.
        gate_time_2q: Latency of two qubit gate operations.
        gate_time_meas: Latency of readout operations.
        t1: T1 coherence time.
        t2: T2 coherence time.
    """

    native_gate_set = set(native_gates)
    gates_1q = native_gate_set & gate_names.SINGLE_QUBIT_GATE_NAMES
    gates_2q = native_gate_set & gate_names.TWO_QUBIT_GATE_NAMES

    if gates_1q | gates_2q != native_gate_set:
        raise ValueError(
            "Only single and two qubit gates are supported as native gates"
        )

    qubits = list(lattice.qubits)
    qubit_count = len(qubits)
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

    graph = nx.grid_2d_graph(lattice.xsize, lattice.ysize)
    mapping = {
        (x, y): lattice.get_qubit((x, y))
        for y in range(lattice.ysize)
        for x in range(lattice.xsize)
    }
    graph = nx.relabel_nodes(graph, mapping)

    trans = SequentialTranspiler(
        [
            SquareLatticeSWAPInsertionTranspiler(lattice),
            GateSetConversionTranspiler(native_gates),
        ]
    )

    return DeviceProperty(
        qubit_count=qubit_count,
        qubits=qubits,
        qubit_graph=graph,
        qubit_properties=qubit_properties,
        native_gates=native_gates,
        gate_properties=gate_properties,
        physical_qubit_count=qubit_count,
        # TODO Calculate backgraound error from t1 and t2
        background_error=None,
        transpiler=trans,
    )
