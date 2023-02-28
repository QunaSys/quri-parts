import math
from typing import Callable
import pytest
from quri_parts.circuit import QuantumGate, UnboundParametricQuantumCircuit, gates
from quri_parts.core.operator.pauli import pauli_label
from quri_parts.core.state import ParametricCircuitQuantumState
from quri_parts.qulacs.estimator import create_qulacs_vector_parametric_estimator

from quri_parts.itensor.estimator import create_itensor_mps_parametric_estimator

single_qubit_gate_list: list[Callable[[int], QuantumGate]] = [
    gates.Identity,
    gates.X,
    gates.Y,
    gates.Z,
    gates.H,
    gates.S,
    gates.Sdag,
    gates.SqrtX,
    gates.SqrtXdag,
    gates.SqrtY,
    gates.SqrtYdag,
    gates.T,
    gates.Tdag,
]

two_qubit_gate_list: list[Callable[[int, int], QuantumGate]] = [
    gates.CNOT,
    gates.CZ,
    gates.SWAP,
]

three_qubit_gate_list: list[Callable[[int, int, int], QuantumGate]] = [
    gates.TOFFOLI,
]


rotation_gate_list: list[Callable[[int, float], QuantumGate]] = [
    gates.RX,
    gates.RY,
    gates.RZ,
]


def test_convert_single_qubit_gate() -> None:
    for qp_fac in single_qubit_gate_list:
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
    for qp_fac in two_qubit_gate_list:
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
    for qp_fac in three_qubit_gate_list:
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
    for qp_fac in rotation_gate_list:
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
