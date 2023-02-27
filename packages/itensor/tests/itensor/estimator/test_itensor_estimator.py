import math
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import numpy as np
import pytest
from quri_parts.circuit import (
    UnboundParametricQuantumCircuit,
    UnboundParametricQuantumCircuitProtocol,
)
from quri_parts.core.operator import Operator, PauliLabel, pauli_label
from quri_parts.core.state import (
    ComputationalBasisState,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
    QuantumStateVector,
    StateVectorType,
)
from quri_parts.core.state.state_vector import QuantumStateVector, StateVectorType
from quri_parts.qulacs.estimator import (
    _Estimate,
    create_qulacs_vector_concurrent_estimator,
    create_qulacs_vector_concurrent_parametric_estimator,
    create_qulacs_vector_estimator,
    create_qulacs_vector_parametric_estimator,
)

from quri_parts.itensor.estimator import (
    _Estimate,
    create_itensor_mps_concurrent_estimator,
    create_itensor_mps_concurrent_parametric_estimator,
    create_itensor_mps_estimator,
    create_itensor_mps_parametric_estimator,
)


def create_vector(qubit_count: int, bits: int) -> StateVectorType:
    vector: StateVectorType = np.zeros(2**qubit_count, dtype=np.cfloat)
    vector[bits] = 1.0
    return vector


def create_vector_state(qubit_count: int, bits: int) -> QuantumStateVector:
    return QuantumStateVector(qubit_count, create_vector(qubit_count, bits))


# class TestVectorEstimator:
#     def test_estimate_pauli_label(self) -> None:
#         pauli = pauli_label("Z0 Z2 Z5")
#         state = ComputationalBasisState(6, bits=0b110010)
#         estimator = create_itensor_mps_estimator()
#         estimate = estimator(pauli, state)
#         assert estimate.value == -1
#         assert estimate.error == 0
#
#     def test_estimate_operator(self) -> None:
#         operator = Operator(
#             {
#                 pauli_label("Z0 Z2 Z5"): 0.25,
#                 pauli_label("Z1 Z2 Z4"): 0.5j,
#             }
#         )
#         state = ComputationalBasisState(6, bits=0b110010)
#         estimator = create_itensor_mps_estimator()
#         estimate = estimator(operator, state)
#         assert estimate.value == -0.25 + 0.5j
#         assert estimate.error == 0
#
class TestVectorConcurrentEstimator:
    def test_invalid_arguments(self) -> None:
        pauli = pauli_label("Z0 Z2 Z5")
        state = ComputationalBasisState(6, bits=0b110010)
        estimator = create_itensor_mps_concurrent_estimator()
        with pytest.raises(ValueError):
            estimator([], [state])
        with pytest.raises(ValueError):
            estimator([pauli], [])
        with pytest.raises(ValueError):
            estimator([pauli] * 3, [state] * 2)

    def test_concurrent_estimate(self) -> None:
        operators: list[Union[PauliLabel, Operator]] = [
            pauli_label("Z0 Z2 Z5"),
            Operator(
                {
                    pauli_label("Z0 Z2 Z5"): 0.25,
                    pauli_label("Z1 Z2 Z4"): 0.5j,
                }
            ),
        ]
        states = [
            ComputationalBasisState(6, bits=0b110000),
            ComputationalBasisState(6, bits=0b110010),
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            estimator = create_itensor_mps_concurrent_estimator(executor, concurrency=2)
            result = estimator(operators, states)
        assert result == [
            _Estimate(value=-1, error=0),
            _Estimate(value=-0.25 + 0.5j, error=0),
        ]

    def test_concurrent_estimate_vector(self) -> None:
        operators: list[Union[PauliLabel, Operator]] = [
            pauli_label("Z0 Z2 Z5"),
            Operator(
                {
                    pauli_label("Z0 Z2 Z5"): 0.25,
                    pauli_label("Z1 Z2 Z4"): 0.5j,
                }
            ),
        ]
        states = [
            create_vector_state(6, 0b110000),
            create_vector_state(6, 0b110010),
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            estimator = create_itensor_mps_concurrent_estimator(executor, concurrency=2)
            result = estimator(operators, states)
        assert result == [
            _Estimate(value=-1, error=0),
            _Estimate(value=-0.25 + 0.5j, error=0),
        ]

    def test_concurrent_estimate_single_state(self) -> None:
        operators: list[Union[PauliLabel, Operator]] = [
            pauli_label("Z0 Z2 Z5"),
            Operator(
                {
                    pauli_label("Z0 Z2 Z5"): 0.25,
                    pauli_label("Z1 Z2 Z4"): 0.5j,
                }
            ),
        ]
        states = [
            ComputationalBasisState(6, bits=0b110010),
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            estimator = create_itensor_mps_concurrent_estimator(executor, concurrency=2)
            result = estimator(operators, states)
        assert result == [
            _Estimate(value=-1, error=0),
            _Estimate(value=-0.25 + 0.5j, error=0),
        ]

    def test_concurrent_estimate_single_operator(self) -> None:
        operators: list[Union[PauliLabel, Operator]] = [
            Operator(
                {
                    pauli_label("Z0 Z2 Z5"): 0.25,
                    pauli_label("Z1 Z2 Z4"): 0.5j,
                }
            ),
        ]
        states = [
            ComputationalBasisState(6, bits=0b110010),
            ComputationalBasisState(6, bits=0b110011),
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            estimator = create_itensor_mps_concurrent_estimator(executor, concurrency=2)
            result = estimator(operators, states)
        assert result == [
            _Estimate(value=-0.25 + 0.5j, error=0),
            _Estimate(value=0.25 + 0.5j, error=0),
        ]


def parametric_circuit() -> UnboundParametricQuantumCircuitProtocol:
    circuit = UnboundParametricQuantumCircuit(6)
    circuit.add_RX_gate(0, -math.pi / 4)
    circuit.add_ParametricRX_gate(0)

    circuit.add_RY_gate(2, -math.pi / 4)
    circuit.add_ParametricRY_gate(2)

    circuit.add_H_gate(5)
    circuit.add_RZ_gate(5, -math.pi / 4)
    circuit.add_ParametricRZ_gate(5)

    return circuit


def create_parametric_vector_state(
    qubit_count: int, circuit: UnboundParametricQuantumCircuitProtocol, bits: int
) -> ParametricQuantumStateVector:
    return ParametricQuantumStateVector(
        qubit_count, circuit, create_vector(qubit_count, bits)
    )


# class TestVectorParametricEstimator:
#     def test_estimate_pauli_label(self) -> None:
#         pauli = pauli_label("Y0 X2 Y5")
#         state = ParametricCircuitQuantumState(6, parametric_circuit())
#         estimator = create_itensor_mps_parametric_estimator()
#         qulacs_estimator = create_qulacs_vector_parametric_estimator()
#         params = [0.0, 0.0, 0.0]
#         estimate = estimator(pauli, state, params)
#         qulacs_estimate = qulacs_estimator(pauli, state, params)
#         assert estimate.value == pytest.approx(qulacs_estimate.value)
#         assert estimate.error == 0
#
#         params = [-math.pi / 4, 0.0, 0.0]
#         estimate = estimator(pauli, state, params)
#         qulacs_estimate = qulacs_estimator(pauli, state, params)
#         assert estimate.value == pytest.approx(qulacs_estimate.value)
#         assert estimate.error == 0
#
#         params = [0.0, -math.pi / 4, 0.0]
#         estimate = estimator(pauli, state, params)
#         qulacs_estimate = qulacs_estimator(pauli, state, params)
#         assert estimate.value == pytest.approx(qulacs_estimate.value)
#         assert estimate.error == 0
#
#         params = [0.0, 0.0, -math.pi / 4]
#         estimate = estimator(pauli, state, params)
#         qulacs_estimate = qulacs_estimator(pauli, state, params)
#         assert estimate.value == pytest.approx(qulacs_estimate.value)
#         assert estimate.error == 0
#
#     def test_estimate_operator(self) -> None:
#         operator = Operator(
#             {
#                 pauli_label("Y0 X2 Y5"): 0.25,
#                 pauli_label("Z1 X2 Z4"): 0.5j,
#             }
#         )
#         state = ParametricCircuitQuantumState(6, parametric_circuit())
#         estimator = create_itensor_mps_parametric_estimator()
#         qulacs_estimator = create_qulacs_vector_parametric_estimator()
#         params = [0.0, 0.0, 0.0]
#         estimate = estimator(operator, state, params)
#         qulacs_estimate = qulacs_estimator(operator, state, params)
#         assert estimate.value == pytest.approx(qulacs_estimate.value)
#         assert estimate.error == 0
#
#         params = [-math.pi / 4, 0.0, 0.0]
#         estimate = estimator(operator, state, params)
#         qulacs_estimate = qulacs_estimator(operator, state, params)
#         assert estimate.value == pytest.approx(qulacs_estimate.value)
#         assert estimate.error == 0
#
#         params = [0.0, -math.pi / 4, 0.0]
#         estimate = estimator(operator, state, params)
#         qulacs_estimate = qulacs_estimator(operator, state, params)
#         assert estimate.value == pytest.approx(qulacs_estimate.value)
#         assert estimate.error == 0
#
#         params = [0.0, 0.0, -math.pi / 4]
#         estimate = estimator(operator, state, params)
#         qulacs_estimate = qulacs_estimator(operator, state, params)
#         assert estimate.value == pytest.approx(qulacs_estimate.value)
#         assert estimate.error == 0

# class TestVectorConcurrentParametricEstimator:
#     def test_concurrent_estimate(self) -> None:
#         operator = Operator(
#             {
#                 pauli_label("Y0 X2 Y5"): 0.25,
#                 pauli_label("Z1 X2 Z4"): 0.5j,
#             }
#         )
#         state = ParametricCircuitQuantumState(6, parametric_circuit())
#         params = [
#             [0.0, 0.0, 0.0, 0.0],
#             [-math.pi / 4, 0, 0, 0],
#             [0, -math.pi / 4, 0, 0],
#             [0, 0, -math.pi / 4, 0],
#             [0, 0, 0, -math.pi / 4],
#         ]
#
#         with ThreadPoolExecutor(max_workers=2) as executor:
#             estimator = create_itensor_mps_concurrent_parametric_estimator(
#                 executor, concurrency=2
#             )
#             result = list(estimator(operator, state, params))
#
#         assert len(result) == 5
#         for r in result:
#             assert r.error == 0
#         assert result[0].value == pytest.approx(
#             0.25 * ((1 / math.sqrt(2)) ** 3) + 0.5j * (-1 / math.sqrt(2))
#         )
#         assert result[1].value == pytest.approx(0.25 * 0.5 + 0.5j * (-1 / math.sqrt(2)))
#         assert result[2].value == pytest.approx(0.25 * 0.5 + 0.5j * (-1))
#         assert result[3].value == pytest.approx(0.25 * 0.5 + 0.5j * (-1 / math.sqrt(2)))
#         assert result[4].value == pytest.approx(0 + 0.5j * (-0.5))
#
#     def test_concurrent_estimate_vector(self) -> None:
#         operator = Operator(
#             {
#                 pauli_label("Y0 X2 Y5"): 0.25,
#                 pauli_label("Z1 X2 Z4"): 0.5j,
#             }
#         )
#         state = create_parametric_vector_state(6, parametric_circuit(), 0b100000)
#         params = [
#             [0.0, 0.0, 0.0, 0.0],
#             [-math.pi / 4, 0, 0, 0],
#             [0, -math.pi / 4, 0, 0],
#             [0, 0, -math.pi / 4, 0],
#             [0, 0, 0, -math.pi / 4],
#         ]
#
#         with ThreadPoolExecutor(max_workers=2) as executor:
#             estimator = create_itensor_mps_concurrent_parametric_estimator(
#                 executor, concurrency=2
#             )
#             result = list(estimator(operator, state, params))
#
#         assert len(result) == 5
#         for r in result:
#             assert r.error == 0
#         assert result[0].value == pytest.approx(
#             0.25 * (-((1 / math.sqrt(2)) ** 3)) + 0.5j * (-1 / math.sqrt(2))
#         )
#         assert result[1].value == pytest.approx(
#             0.25 * (-0.5) + 0.5j * (-1 / math.sqrt(2))
#         )
#         assert result[2].value == pytest.approx(0.25 * (-0.5) + 0.5j * (-1))
#         assert result[3].value == pytest.approx(
#             0.25 * (-0.5) + 0.5j * (-1 / math.sqrt(2))
#         )
#         assert result[4].value == pytest.approx(0 + 0.5j * (-0.5))
#
