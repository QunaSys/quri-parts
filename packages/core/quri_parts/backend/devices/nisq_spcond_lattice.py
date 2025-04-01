<<<<<<< HEAD
import warnings
from collections.abc import Collection
from typing import Optional, cast
=======
from collections.abc import Collection
from typing import Optional
>>>>>>> main

import networkx as nx

from quri_parts.backend.device import DeviceProperty, GateProperty, QubitProperty
from quri_parts.backend.units import TimeValue
<<<<<<< HEAD
from quri_parts.circuit import gate_names, noise
from quri_parts.circuit.gate_names import GateNameType, NonParametricGateNameType
=======
from quri_parts.circuit import gate_names
from quri_parts.circuit.gate_names import GateNameType
>>>>>>> main
from quri_parts.circuit.topology import (
    SquareLattice,
    SquareLatticeSWAPInsertionTranspiler,
)
from quri_parts.circuit.transpile import (
<<<<<<< HEAD
    CircuitTranspiler,
=======
>>>>>>> main
    GateSetConversionTranspiler,
    SequentialTranspiler,
)


def generate_device_property(
    lattice: SquareLattice,
<<<<<<< HEAD
    native_gates_1q: Collection[str],
    native_gates_2q: Collection[str],
=======
    native_gates: Collection[GateNameType],
>>>>>>> main
    gate_error_1q: float,
    gate_error_2q: float,
    gate_error_meas: float,
    gate_time_1q: TimeValue,
    gate_time_2q: TimeValue,
    gate_time_meas: TimeValue,
    t1: Optional[TimeValue] = None,
    t2: Optional[TimeValue] = None,
<<<<<<< HEAD
    transpiler: Optional[CircuitTranspiler] = None,
=======
>>>>>>> main
) -> DeviceProperty:
    """Generate DeviceProperty object for a typical NISQ superconducting qubit
    device.

    Assumes that the device's qubits are connected as square lattice with no defects
    and that a subset of the gates natively supported by QURI Parts can be used as
    the native gates.

    Args:
        lattice: SquareLattice instance representing the device qubits connectivity.
<<<<<<< HEAD
        native_gates_1q: Single qubit native gates supported by the device.
        native_gates_2q: Two qubit native gates supported by the device.
=======
        native_gates: Native gates supported by the device.
>>>>>>> main
        gate_error_1q: Error rate of single qubit gate operations.
        gate_error_2q: Error rate of two qubit gate operations.
        gate_error_meas: Error rate of readout operations.
        gate_time_1q: Latency of single qubit gate operations.
        gate_time_2q: Latency of two qubit gate operations.
        gate_time_meas: Latency of readout operations.
        t1: T1 coherence time.
        t2: T2 coherence time.
<<<<<<< HEAD
        transpiler: CircuitTranspiler to adapt the circuit to the device. If not
            specified, default transpiler is used.
    """
    if t1 is not None or t2 is not None:
        warnings.warn(
            "The t1 t2 error is not yet supported and is not reflected in the "
            "fidelity estimation or noise model."
        )

    gates_1q = set(native_gates_1q)
    gates_2q = set(native_gates_2q)
    native_gates = gates_1q | gates_2q
    meas = native_gates & {gate_names.Measurement}
=======
    """

    native_gate_set = set(native_gates)
    gates_1q = native_gate_set & gate_names.SINGLE_QUBIT_GATE_NAMES
    gates_2q = native_gate_set & gate_names.TWO_QUBIT_GATE_NAMES

    if gates_1q | gates_2q != native_gate_set:
        raise ValueError(
            "Only single and two qubit gates are supported as native gates"
        )
>>>>>>> main

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
<<<<<<< HEAD
    gate_properties.extend(
        [
            GateProperty(name, (), gate_error=gate_error_meas, gate_time=gate_time_meas)
            for name in meas
        ]
    )
=======
>>>>>>> main

    graph = nx.grid_2d_graph(lattice.xsize, lattice.ysize)
    mapping = {
        (x, y): lattice.get_qubit((x, y))
        for y in range(lattice.ysize)
        for x in range(lattice.xsize)
    }
    graph = nx.relabel_nodes(graph, mapping)

<<<<<<< HEAD
    transpiler = (
        transpiler
        if transpiler is not None
        else SequentialTranspiler(
            [
                GateSetConversionTranspiler(
                    cast(Collection[GateNameType], native_gates)
                ),
                SquareLatticeSWAPInsertionTranspiler(lattice),
                GateSetConversionTranspiler(
                    cast(Collection[GateNameType], native_gates)
                ),
            ]
        )
    )

    noise_model = noise.NoiseModel(
        [
            noise.DepolarizingNoise(
                error_prob=gate_error_1q,
                target_gates=list(cast(set[NonParametricGateNameType], gates_1q)),
            ),
            noise.DepolarizingNoise(
                error_prob=gate_error_2q,
                target_gates=list(cast(set[NonParametricGateNameType], gates_2q)),
            ),
            noise.MeasurementNoise([noise.BitFlipNoise(error_prob=gate_error_meas)]),
=======
    trans = SequentialTranspiler(
        [
            GateSetConversionTranspiler(native_gates),
            SquareLatticeSWAPInsertionTranspiler(lattice),
            GateSetConversionTranspiler(native_gates),
>>>>>>> main
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
<<<<<<< HEAD
        transpiler=transpiler,
        noise_model=noise_model,
=======
        transpiler=trans,
>>>>>>> main
    )
