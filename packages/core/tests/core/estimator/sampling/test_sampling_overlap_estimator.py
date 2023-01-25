# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable
from math import sqrt
from typing import Any, Sequence, Union, cast
from unittest.mock import Mock

import pytest

from quri_parts.circuit import (
    H,
    NonParametricQuantumCircuit,
    QuantumCircuit,
    inverse_circuit,
)
from quri_parts.core.estimator import Estimate
from quri_parts.core.estimator.sampling import (
    concurrent_sampling_overlap_estimate,
    create_sampling_concurrent_overlap_estimator,
    create_sampling_overlap_estimator,
    sampling_overlap_estimate,
)
from quri_parts.core.state import GeneralCircuitQuantumState, CircuitQuantumState
from quri_parts.core.sampling import MeasurementCounts
from quri_parts.core.sampling.weighted_shots_allocator import (
    create_equipartition_weighted_shots_allocator,
)
from quri_parts.core.state import ComputationalBasisState

n_qubits = 3


def initial_state() -> ComputationalBasisState:
    return ComputationalBasisState(n_qubits, bits=0b101)


def state_prep_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits)
    circuit.add_X_gate(0)
    circuit.add_X_gate(2)
    return circuit


def counts() -> Iterable[MeasurementCounts]:
    return [
        {
            0b000: 4,
            0b001: 1,
            0b100: 2,
            0b111: 1,
        },
        {
            0b000: 0,
            0b001: 1,
            0b100: 2,
            0b111: 1,
        },
        {
            0b000: 3,
            0b001: 1,
            0b100: 2,
            0b111: 1,
        },
        {
            0b000: 7,
            0b001: 1,
            0b100: 2,
            0b111: 1,
        },
    ]


def weights() -> list[complex]:
    return [
        1.0j,
        2.0,
        3.0,
        4.0,
    ]


def total_shots() -> int:
    return cast(int, sum(sum(count.values()) for count in counts()))


def sampler(
    _: Iterable[tuple[NonParametricQuantumCircuit, int]]
) -> Iterable[MeasurementCounts]:
    return counts()


def mock_sampler() -> Mock:
    mock = Mock()
    mock.side_effect = sampler
    return mock


def state_list() -> list[CircuitQuantumState]:
    return [
        GeneralCircuitQuantumState(n_qubits, state_prep_circuit()),
        GeneralCircuitQuantumState(n_qubits, state_prep_circuit() + [H(0)]),
        GeneralCircuitQuantumState(n_qubits, state_prep_circuit() + [H(0), H(1)]),
        GeneralCircuitQuantumState(n_qubits, state_prep_circuit() + [H(1), H(2)]),
    ]


def sampled_circuits() -> list[QuantumCircuit]:
    circuits = [
        bra.circuit.get_mutable_copy() + inverse_circuit(ket.circuit.get_mutable_copy())
        for bra, ket in zip(state_list(), state_list())
    ]

    return circuits


allocator = create_equipartition_weighted_shots_allocator()


def assert_sampler_args(s: Mock) -> None:
    s.assert_called_once()
    args: list[Any] = list(*s.call_args.args)
    assert len(args) == 4
    expected_circuits = sampled_circuits()
    for circuit, shots in args:
        assert shots == total_shots() // 4
        assert circuit in expected_circuits


def assert_sample(estimate: Estimate[complex]) -> None:
    counts = [
        [4, 1, 2, 1],
        [0, 1, 2, 1],
        [3, 1, 2, 1],
        [7, 1, 2, 1],
    ]

    expected_exp = sum(
        weight * count[0] / sum(count) for weight, count in zip(weights(), counts)
    )
    assert estimate.value == expected_exp

    expected_err = sqrt(
        sum(
            abs(weight) ** 2 * count[0] / sum(count) ** 2
            for weight, count in zip(weights(), counts)
        )
    )
    assert estimate.error == expected_err


class TestSamplingEstimate:
    def test_zero_op(self) -> None:
        estimate = sampling_overlap_estimate(
            [], [], [], total_shots(), sampler, allocator
        )
        assert estimate.value == 0.0
        assert estimate.error == 0.0

    def test_sampling_estimate(self) -> None:
        s = mock_sampler()
        estimate = sampling_overlap_estimate(
            state_list(), state_list(), weights(), total_shots(), s, allocator
        )
        assert_sampler_args(s)
        assert_sample(estimate)

    def test_sampling_estimate_zero_shots(self) -> None:
        def sampler(
            shot_circuit_pairs: Iterable[tuple[NonParametricQuantumCircuit, int]]
        ) -> Iterable[MeasurementCounts]:
            return [
                {} if shot_circuit_pair[1] == 0 else count
                for shot_circuit_pair, count in zip(shot_circuit_pairs, counts())
            ]

        def allocator(
            weights: Sequence[Union[float, complex]], total_shots: int
        ) -> Sequence[int]:
            s = total_shots // len(weights)
            return [s for _ in weights]

        estimate = sampling_overlap_estimate(
            state_list(), state_list(), weights(), total_shots(), sampler, allocator
        )
        assert isinstance(estimate.value, complex)


class TestSamplingEstimator:
    def test_sampling_estimator(self) -> None:
        s = mock_sampler()
        estimator = create_sampling_overlap_estimator(total_shots(), s, allocator)
        estimate = estimator(state_list(), state_list(), weights())
        assert_sampler_args(s)
        assert_sample(estimate)


class TestConcurrentSamplingEstimate:
    def test_invalid_arguments(self) -> None:
        s = mock_sampler()

        with pytest.raises(ValueError):
            concurrent_sampling_overlap_estimate(
                [], [[initial_state()]], [[1.0]], total_shots(), s, allocator
            )

        with pytest.raises(ValueError):
            concurrent_sampling_overlap_estimate(
                [[initial_state()]], [], [[1.0]], total_shots(), s, allocator
            )

        with pytest.raises(ValueError):
            concurrent_sampling_overlap_estimate(
                [[initial_state()]],
                [[initial_state()]],
                [],
                total_shots(),
                s,
                allocator,
            )

        with pytest.raises(ValueError):
            concurrent_sampling_overlap_estimate(
                [[initial_state()]] * 3,
                [[initial_state()]] * 2,
                [[2.0]] * 4,
                total_shots(),
                s,
                allocator,
            )

        with pytest.raises(ValueError):
            concurrent_sampling_overlap_estimate(
                [[initial_state()] * 3],
                [[initial_state()] * 2],
                [[2.0] * 4],
                total_shots(),
                s,
                allocator,
            )

    def test_concurrent_estimate(self) -> None:
        s = mock_sampler()

        estimates = concurrent_sampling_overlap_estimate(
            [state_list(), state_list()],
            [state_list(), state_list()],
            [weights(), weights()],
            total_shots(),
            s,
            allocator,
        )

        estimate_list = list(estimates)
        assert len(estimate_list) == 2
        assert_sample(estimate_list[0])

    def test_concurrent_estimate_single_bra(self) -> None:
        s = mock_sampler()

        estimates = concurrent_sampling_overlap_estimate(
            [state_list(), state_list()],
            [state_list()],
            [weights(), weights()],
            total_shots(),
            s,
            allocator,
        )

        estimate_list = list(estimates)
        assert len(estimate_list) == 2
        assert_sample(estimate_list[0])

    def test_concurrent_estimate_single_weight(self) -> None:
        s = mock_sampler()

        estimates = concurrent_sampling_overlap_estimate(
            [state_list(), state_list()],
            [state_list(), state_list()],
            [weights()],
            total_shots(),
            s,
            allocator,
        )

        estimate_list = list(estimates)
        assert len(estimate_list) == 2
        assert_sample(estimate_list[0])


class TestSamplingConcurrentEstimator:
    def test_sampling_concurrent_estimator(self) -> None:
        s = mock_sampler()

        estimator = create_sampling_concurrent_overlap_estimator(
            total_shots(),
            s,
            allocator,
        )
        estimates = estimator(
            [state_list(), state_list()],
            [state_list(), state_list()],
            [weights(), weights()],
        )

        estimate_list = list(estimates)
        assert len(estimate_list) == 2
        assert_sample(estimate_list[0])
