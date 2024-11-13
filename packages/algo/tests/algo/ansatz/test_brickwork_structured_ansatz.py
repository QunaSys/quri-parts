from typing import Callable, Sequence

import numpy as np

from quri_parts.circuit import (
    ImmutableLinearMappedParametricQuantumCircuit,
    PauliRotation,
    QuantumCircuit,
    QuantumGate,
    RZ,
)


def RXX(target_indices: Sequence[int], angle: float) -> QuantumGate:
    gate = PauliRotation(target_indices, (1, 1), angle)
    return gate


def _build_circuit_qc4_d3(params_list: Sequence[float]) -> QuantumCircuit:
    gates = []
    tz0, tz1 = params_list[0:2]
    gates.extend(
        [RXX((0, 1), np.pi / 2), RZ(0, tz0), RZ(1, tz1), RXX((0, 1), -np.pi / 2)]
    )
    tz2, tz3 = params_list[2:4]
    gates.extend(
        [RXX((2, 3), np.pi / 2), RZ(2, tz2), RZ(3, tz3), RXX((2, 3), -np.pi / 2)]
    )
    tz1, tz2 = params_list[4:6]
    gates.extend(
        [RXX((1, 2), np.pi / 2), RZ(1, tz1), RZ(2, tz2), RXX((1, 2), -np.pi / 2)]
    )

    tz0, tz1 = params_list[6:8]
    gates.extend(
        [RXX((0, 1), np.pi / 2), RZ(0, tz0), RZ(1, tz1), RXX((0, 1), -np.pi / 2)]
    )
    tz2, tz3 = params_list[8:10]
    gates.extend(
        [RXX((2, 3), np.pi / 2), RZ(2, tz2), RZ(3, tz3), RXX((2, 3), -np.pi / 2)]
    )
    tz1, tz2 = params_list[10:12]
    gates.extend(
        [RXX((1, 2), np.pi / 2), RZ(1, tz1), RZ(2, tz2), RXX((1, 2), -np.pi / 2)]
    )

    tz0, tz1 = params_list[12:14]
    gates.extend(
        [RXX((0, 1), np.pi / 2), RZ(0, tz0), RZ(1, tz1), RXX((0, 1), -np.pi / 2)]
    )
    tz2, tz3 = params_list[14:16]
    gates.extend(
        [RXX((2, 3), np.pi / 2), RZ(2, tz2), RZ(3, tz3), RXX((2, 3), -np.pi / 2)]
    )
    tz1, tz2 = params_list[16:18]
    gates.extend(
        [RXX((1, 2), np.pi / 2), RZ(1, tz1), RZ(2, tz2), RXX((1, 2), -np.pi / 2)]
    )

    circuit = QuantumCircuit(4)

    for gate in gates:
        circuit.add_gate(gate)

    return circuit


def _test_circuit(
    ansatz: ImmutableLinearMappedParametricQuantumCircuit,
    test_circuit_builder: Callable[[Sequence[float]], QuantumCircuit],
) -> None:
    params_list = list(2.0 * np.pi * np.random.random(ansatz.parameter_count))
    circuit = ansatz.bind_parameters(params_list)
    test_circuit = test_circuit_builder(params_list)
    assert circuit.gates == test_circuit.gates
