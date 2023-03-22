from numpy import allclose, array, pi

from quri_parts.circuit import (
    LinearMappedUnboundParametricQuantumCircuit,
    QuantumCircuit,
)
from quri_parts.core.state import GeneralCircuitQuantumState, QuantumStateVector
from quri_parts.qulacs.simulator import evalutate_state_to_vector, run_circuit


def test_evalutate_state_to_vector_general_circuit_state_to_vec() -> None:
    """testing evaluating GeneralCircuitQuantumState to
    CircuitQuantumStateWithInitialVector without circuit."""
    n = 2
    # quri-parts operations
    quantum_circuit = QuantumCircuit(n)
    quantum_circuit.add_CNOT_gate(0, 1)
    quantum_circuit.add_X_gate(0)
    quantum_circuit.add_RX_gate(0, pi / 7)
    quantum_circuit.add_RY_gate(0, pi / 8)
    quantum_circuit.add_H_gate(1)

    circuit_quantum_state = GeneralCircuitQuantumState(n, quantum_circuit)
    quantum_state_vector = evalutate_state_to_vector(circuit_quantum_state)

    expect_output_vector = array(
        [
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
        ]
    )

    assert len(quantum_state_vector.circuit.gates) == 0
    assert allclose(quantum_state_vector.vector, expect_output_vector)


def test_evalutate_state_to_vector_vec_to_vec() -> None:
    """testing evaluate CircuitQuantumStateWithInitialVector with circuit to
    CircuitQuantumStateWithInitialVector without circuit."""

    n = 2
    # quri-parts operations
    quantum_circuit = QuantumCircuit(n)
    quantum_circuit.add_CNOT_gate(0, 1)
    quantum_circuit.add_X_gate(0)
    quantum_circuit.add_RX_gate(0, pi / 7)
    quantum_circuit.add_RY_gate(0, pi / 8)
    quantum_circuit.add_H_gate(1)

    circuit_quantum_state = QuantumStateVector(n, array([1, 0, 0, 0]), quantum_circuit)
    quantum_state_vector = evalutate_state_to_vector(circuit_quantum_state)

    expect_output_vector = array(
        [
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
        ]
    )

    assert len(quantum_state_vector.circuit.gates) == 0
    assert allclose(quantum_state_vector.vector, expect_output_vector)


def test_evalutate_state_to_vector_vec_to_vec_2() -> None:
    """testing convert
    CircuitQuantumStateWithInitialVector.with_gates_applied() to
    CircuitQuantumStateWithInitialVector without circuit."""
    n = 2
    # quri-parts operations
    quantum_circuit = QuantumCircuit(n)
    quantum_circuit.add_CNOT_gate(0, 1)
    quantum_circuit.add_X_gate(0)
    quantum_circuit.add_RX_gate(0, pi / 7)
    quantum_circuit.add_RY_gate(0, pi / 8)
    quantum_circuit.add_H_gate(1)

    circuit_quantum_state = QuantumStateVector(n, array([1, 0, 0, 0]))
    circuit_quantum_state = circuit_quantum_state.with_gates_applied(quantum_circuit)
    quantum_state_vector = evalutate_state_to_vector(circuit_quantum_state)

    expect_output_vector = array(
        [
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
        ]
    )

    assert len(quantum_state_vector.circuit.gates) == 0
    assert allclose(quantum_state_vector.vector, expect_output_vector)


def test_evalutate_state_to_vector_genenral_circ_state_to_vec_to_vec() -> None:
    """testing evaluation of state_vector.with_gates_applied."""
    n = 2
    # quri-parts operations
    quantum_circuit = QuantumCircuit(n)
    quantum_circuit.add_CNOT_gate(0, 1)
    quantum_circuit.add_X_gate(0)
    quantum_circuit.add_RX_gate(0, pi / 7)
    quantum_circuit.add_RY_gate(0, pi / 8)
    quantum_circuit.add_H_gate(1)

    circuit_quantum_state = GeneralCircuitQuantumState(n, quantum_circuit)
    quantum_state_vector = evalutate_state_to_vector(circuit_quantum_state)

    in_state_circuit = QuantumCircuit(n)
    in_state_circuit.add_Y_gate(1)
    quantum_state_vector = quantum_state_vector.with_gates_applied(in_state_circuit)
    quantum_state_vector = evalutate_state_to_vector(quantum_state_vector)

    expect_output_vector = array(
        [
            -0.154323 + 0.134491j,
            -0.0306967 - 0.676132j,
            0.154323 - 0.134491j,
            0.0306967 + 0.676132j,
        ]
    )

    assert len(quantum_state_vector.circuit.gates) == 0
    assert allclose(quantum_state_vector.vector, expect_output_vector)


def test_run_circuit_simple() -> None:
    """testing run_circuit."""
    n = 2
    theta_x = pi / 7
    theta_y = pi / 8

    update_circuit = QuantumCircuit(n)
    update_circuit.add_CNOT_gate(0, 1)
    update_circuit.add_X_gate(0)
    update_circuit.add_RX_gate(0, theta_x)
    update_circuit.add_RY_gate(0, theta_y)
    update_circuit.add_H_gate(1)

    updated_quantum_state_vector = run_circuit(update_circuit, array([1, 0, 0, 0]))

    # expectation
    expect_output_vector = array(
        [
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
        ]
    )

    assert allclose(updated_quantum_state_vector, expect_output_vector)


def test_run_cirtuit() -> None:
    """testing run_circuit with circuit."""
    n = 2
    theta_x = pi / 7
    theta_y = pi / 8

    # quri-parts operations
    quantum_circuit = QuantumCircuit(n)
    quantum_circuit.add_H_gate(0)
    quantum_circuit.add_CNOT_gate(0, 1)

    circuit_quantum_state = GeneralCircuitQuantumState(n, quantum_circuit)
    quantum_state_vector = evalutate_state_to_vector(circuit_quantum_state)

    update_circuit = QuantumCircuit(n)
    update_circuit.add_CNOT_gate(0, 1)
    update_circuit.add_X_gate(0)
    update_circuit.add_RX_gate(0, theta_x)
    update_circuit.add_RY_gate(0, theta_y)
    update_circuit.add_H_gate(1)

    updated_quantum_state_vector = run_circuit(
        update_circuit, quantum_state_vector.vector
    )

    # expectation
    expect_output_vector = array(
        [
            0.382998 - 0.0874168j,
            0.573197 - 0.130828j,
            0.382998 - 0.0874168j,
            0.573197 - 0.130828j,
        ]
    )

    assert allclose(updated_quantum_state_vector, expect_output_vector)


def test_updateStateVector_with_param() -> None:
    n = 2
    theta_x = pi / 7
    theta_y = pi / 8

    # quri-parts operations
    quantum_circuit = QuantumCircuit(n)
    quantum_circuit.add_H_gate(0)
    quantum_circuit.add_CNOT_gate(0, 1)

    circuit_quantum_state = GeneralCircuitQuantumState(n, quantum_circuit)
    quantum_state_vector = evalutate_state_to_vector(circuit_quantum_state)

    update_circuit = LinearMappedUnboundParametricQuantumCircuit(n)
    theta, phi = update_circuit.add_parameters("theta", "phi")
    update_circuit.add_CNOT_gate(0, 1)
    update_circuit.add_X_gate(0)
    update_circuit.add_ParametricRX_gate(0, {theta: 1})
    update_circuit.add_ParametricRY_gate(0, {phi: 1})
    update_circuit.add_H_gate(1)
    bounnd_update_circuit = update_circuit.bind_parameters(params=[theta_x, theta_y])
    updated_quantum_state_vector = run_circuit(
        bounnd_update_circuit, quantum_state_vector.vector
    )

    # expectation
    expect_output_vector = array(
        [
            0.382998 - 0.0874168j,
            0.573197 - 0.130828j,
            0.382998 - 0.0874168j,
            0.573197 - 0.130828j,
        ]
    )

    assert allclose(updated_quantum_state_vector, expect_output_vector)
