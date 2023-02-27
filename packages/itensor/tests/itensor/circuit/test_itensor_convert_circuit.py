import math
from collections.abc import Mapping
from typing import Callable, Type, cast

import numpy as np
import pytest
import qulacs
from quri_parts.circuit import (
    LinearMappedUnboundParametricQuantumCircuit,
    QuantumCircuit,
    QuantumGate,
    UnboundParametricQuantumCircuit,
    gates,
)
from quri_parts.core.operator.operator import Operator
from quri_parts.core.operator.pauli import PAULI_IDENTITY, pauli_label
from quri_parts.core.state import ComputationalBasisState, ParametricCircuitQuantumState
from quri_parts.core.state.state import CircuitQuantumState

from quri_parts.itensor.estimator import (
    create_itensor_mps_estimator,
    create_itensor_mps_parametric_estimator,
)

single_qubit_gate_mapping: Mapping[
    Callable[[int], QuantumGate], Type[qulacs.QuantumGateBase]
] = {
    gates.Identity: qulacs.gate.Identity,
    gates.X: qulacs.gate.X,
    gates.Y: qulacs.gate.Y,
    gates.Z: qulacs.gate.Z,
    gates.H: qulacs.gate.H,
    gates.S: qulacs.gate.S,
    gates.Sdag: qulacs.gate.Sdag,
    gates.SqrtX: qulacs.gate.sqrtX,
    gates.SqrtXdag: qulacs.gate.sqrtXdag,
    gates.SqrtY: qulacs.gate.sqrtY,
    gates.SqrtYdag: qulacs.gate.sqrtYdag,
    gates.T: qulacs.gate.T,
    gates.Tdag: qulacs.gate.Tdag,
}


two_qubit_gate_mapping: Mapping[
    Callable[[int, int], QuantumGate], Type[qulacs.QuantumGateBase]
] = {
    gates.CNOT: qulacs.gate.CNOT,
    gates.CZ: qulacs.gate.CZ,
    gates.SWAP: qulacs.gate.SWAP,
}

three_qubit_gate_mapping: Mapping[
    Callable[[int, int, int], QuantumGate], Type[qulacs.QuantumGateBase]
] = {
    gates.TOFFOLI: qulacs.gate.TOFFOLI,
}


rotation_gate_mapping: Mapping[
    Callable[[int, float], QuantumGate], Type[qulacs.QuantumGateBase]
] = {
    gates.RX: qulacs.gate.RX,
    gates.RY: qulacs.gate.RY,
    gates.RZ: qulacs.gate.RZ,
}


from quri_parts.qulacs.estimator import (
    create_qulacs_vector_estimator,
    create_qulacs_vector_parametric_estimator,
)


def test_convert_single_qubit_gate() -> None:
    for qp_fac, _ in single_qubit_gate_mapping.items():
        pauli = pauli_label("Z0 Z2 Z5")
        estimator = create_itensor_mps_parametric_estimator()
        qulacs_estimator = create_qulacs_vector_parametric_estimator()
        circuit = UnboundParametricQuantumCircuit(6)
        circuit.add_RX_gate(0, -math.pi / 4)
        circuit.add_RY_gate(2, -math.pi / 4)
        circuit.add_H_gate(5)
        circuit.add_RZ_gate(5, -math.pi / 4)
        circuit.add_gate(qp_fac(0))
        state = ParametricCircuitQuantumState(6, circuit)
        estimate = estimator(pauli, state, [])
        qulacs_estimate = qulacs_estimator(pauli, state, [])
        assert estimate.value == pytest.approx(qulacs_estimate.value)
        assert estimate.error == qulacs_estimate.error


def test_convert_two_qubit_gate() -> None:
    for qp_fac, _ in two_qubit_gate_mapping.items():
        pauli = pauli_label("Z0 Z2 Z5")
        estimator = create_itensor_mps_parametric_estimator()
        qulacs_estimator = create_qulacs_vector_parametric_estimator()
        circuit = UnboundParametricQuantumCircuit(6)
        circuit.add_RX_gate(0, -math.pi / 4)
        circuit.add_RY_gate(2, -math.pi / 4)
        circuit.add_H_gate(5)
        circuit.add_RZ_gate(5, -math.pi / 4)
        circuit.add_gate(qp_fac(0, 1))
        state = ParametricCircuitQuantumState(6, circuit)
        estimate = estimator(pauli, state, [])
        qulacs_estimate = qulacs_estimator(pauli, state, [])
        assert estimate.value == pytest.approx(qulacs_estimate.value)
        assert estimate.error == qulacs_estimate.error


def test_convert_three_qubit_gate() -> None:
    for qp_fac, _ in three_qubit_gate_mapping.items():
        pauli = pauli_label("Z0 Z2 Z5")
        estimator = create_itensor_mps_parametric_estimator()
        qulacs_estimator = create_qulacs_vector_parametric_estimator()
        circuit = UnboundParametricQuantumCircuit(6)
        circuit.add_RX_gate(0, -math.pi / 4)
        circuit.add_RY_gate(2, -math.pi / 4)
        circuit.add_H_gate(5)
        circuit.add_RZ_gate(5, -math.pi / 4)
        circuit.add_gate(qp_fac(0, 1, 2))
        state = ParametricCircuitQuantumState(6, circuit)
        estimate = estimator(pauli, state, [])
        qulacs_estimate = qulacs_estimator(pauli, state, [])
        assert estimate.value == pytest.approx(qulacs_estimate.value)
        assert estimate.error == qulacs_estimate.error


def test_convert_rotation_gate() -> None:
    for qp_fac, _ in rotation_gate_mapping.items():
        pauli = pauli_label("Z0 Z2 Z5")
        estimator = create_itensor_mps_parametric_estimator()
        qulacs_estimator = create_qulacs_vector_parametric_estimator()
        circuit = UnboundParametricQuantumCircuit(6)
        circuit.add_RX_gate(0, -math.pi / 4)
        circuit.add_RY_gate(2, -math.pi / 4)
        circuit.add_H_gate(5)
        circuit.add_RZ_gate(5, -math.pi / 4)
        circuit.add_gate(qp_fac(0, 0.5))
        state = ParametricCircuitQuantumState(6, circuit)
        estimate = estimator(pauli, state, [])
        qulacs_estimate = qulacs_estimator(pauli, state, [0.5])
        assert estimate.value == pytest.approx(qulacs_estimate.value)
        assert estimate.error == qulacs_estimate.error


def test_convert_u_gate() -> None:
    pauli = pauli_label("Z0 Z2 Z5")
    estimator = create_itensor_mps_parametric_estimator()
    qulacs_estimator = create_qulacs_vector_parametric_estimator()
    circuit = UnboundParametricQuantumCircuit(6)
    circuit.add_RX_gate(0, -math.pi / 4)
    circuit.add_RY_gate(2, -math.pi / 4)
    circuit.add_H_gate(5)
    circuit.add_RZ_gate(5, -math.pi / 4)
    circuit.add_gate(gates.U1(0, 0.5))
    state = ParametricCircuitQuantumState(6, circuit)
    estimate = estimator(pauli, state, [])
    qulacs_estimate = qulacs_estimator(pauli, state, [])
    assert estimate.value == pytest.approx(qulacs_estimate.value)
    assert estimate.error == qulacs_estimate.error

    circuit = UnboundParametricQuantumCircuit(6)
    circuit.add_RX_gate(0, -math.pi / 4)
    circuit.add_RY_gate(2, -math.pi / 4)
    circuit.add_H_gate(5)
    circuit.add_RZ_gate(5, -math.pi / 4)
    circuit.add_gate(gates.U2(0, 0.5, 0.5))
    state = ParametricCircuitQuantumState(6, circuit)
    estimate = estimator(pauli, state, [])
    qulacs_estimate = qulacs_estimator(pauli, state, [])
    assert estimate.value == pytest.approx(qulacs_estimate.value)
    assert estimate.error == qulacs_estimate.error

    circuit = UnboundParametricQuantumCircuit(6)
    circuit.add_RX_gate(0, -math.pi / 4)
    circuit.add_RY_gate(2, -math.pi / 4)
    circuit.add_H_gate(5)
    circuit.add_RZ_gate(5, -math.pi / 4)
    circuit.add_gate(gates.U3(0, 0.5, 0.5, 0.5))
    state = ParametricCircuitQuantumState(6, circuit)
    estimate = estimator(pauli, state, [])
    qulacs_estimate = qulacs_estimator(pauli, state, [])
    assert estimate.value == pytest.approx(qulacs_estimate.value)
    assert estimate.error == qulacs_estimate.error
