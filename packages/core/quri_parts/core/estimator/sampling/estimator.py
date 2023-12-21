# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from math import sqrt
from typing import Union

from quri_parts.core.estimator import (
    ConcurrentQuantumEstimator,
    Estimatable,
    Estimate,
    QuantumEstimator,
    parse_concurrent_estimator_arguments,
)
from quri_parts.core.estimator.sampling.pauli import (
    general_pauli_sum_expectation_estimator,
    general_pauli_sum_sample_variance,
)
from quri_parts.core.measurement import (
    CommutablePauliSetMeasurement,
    CommutablePauliSetMeasurementFactory,
    PauliReconstructorFactory,
)
from quri_parts.core.operator import PAULI_IDENTITY, CommutablePauliSet, Operator
from quri_parts.core.sampling import (
    ConcurrentSampler,
    MeasurementCounts,
    PauliSamplingShotsAllocator,
)
from quri_parts.core.state import CircuitQuantumState

from .estimator_helpers import CircuitShotPairPreparationFunction
from .estimator_helpers import circuit_shot_pairs_preparation_fn as default_prep
from .estimator_helpers import (
    distribute_shots_among_pauli_sets,
    get_constant_seperated_measurement_group,
)


class _Estimate:
    def __init__(
        self,
        op: Operator,
        const: complex,
        pauli_sets: Sequence[CommutablePauliSet],
        pauli_recs: Sequence[PauliReconstructorFactory],
        sampling_counts: Sequence[MeasurementCounts],
    ):
        self._op = op
        self._const = const
        self._pauli_sets = pauli_sets
        self._pauli_recs = pauli_recs
        self._sampling_counts = sampling_counts

    @cached_property
    def value(self) -> complex:
        val = self._const
        for pauli_set, pauli_rec, counts in zip(
            self._pauli_sets, self._pauli_recs, self._sampling_counts
        ):
            val += general_pauli_sum_expectation_estimator(
                counts, pauli_set, self._op, pauli_rec
            )
        return val

    @cached_property
    def error(self) -> float:
        square_err: float = 0.0
        for pauli_set, pauli_rec, counts in zip(
            self._pauli_sets, self._pauli_recs, self._sampling_counts
        ):
            total_counts = sum(counts.values())
            var = general_pauli_sum_sample_variance(
                counts, pauli_set, self._op, pauli_rec
            )
            square_err += var / total_counts
        return sqrt(square_err)


@dataclass
class _ConstEstimate:
    value: complex
    error: float = 0.0


def get_estimate_from_sampling_result(
    op: Operator,
    measurement_groups: Iterable[CommutablePauliSetMeasurement],
    const: complex,
    sampling_counts: Iterable[MeasurementCounts],
) -> Estimate[complex]:
    """Converts sampling counts into the estimation of the operator's
    expectation value."""
    pauli_sets = tuple(m.pauli_set for m in measurement_groups)
    pauli_recs = tuple(m.pauli_reconstructor_factory for m in measurement_groups)
    return _Estimate(op, const, pauli_sets, pauli_recs, tuple(sampling_counts))


def sampling_estimate(
    op: Estimatable,
    state: CircuitQuantumState,
    total_shots: int,
    sampler: ConcurrentSampler,
    measurement: Union[
        CommutablePauliSetMeasurementFactory, Iterable[CommutablePauliSetMeasurement]
    ],
    shots_allocator: PauliSamplingShotsAllocator,
    circuit_shot_pair_prep_fn: CircuitShotPairPreparationFunction = default_prep,
) -> Estimate[complex]:
    """Estimate expectation value of a given operator with a given state by
    sampling measurement.

    The sampling measurements are configured with arguments as follows.

    Args:
        op: An operator of which expectation value is estimated.
        state: A quantum state on which the operator expectation is evaluated.
        total_shots: Total number of shots available for sampling measurements.
        sampler: A Sampler that actually performs the sampling measurements.
        measurement: This can be either one of:
            - A function that performs Pauli grouping and returns a measurement
                scheme for Pauli operators constituting the original operator.
            - An iterable collection of :class:`~CommutablePauliSetMeasurement`
                that corresponds to the grouping result of the input op.
        shots_allocator: A function that allocates the total shots to Pauli groups to
            be measured.
        circuit_shot_pair_prep_fn: A :class:`~CircuitShotPairPreparationFunction` that
            prepares the set of circuits to perform measurement with. It is default to
            a function that concatenates the measurement circuits after tghe state
            preparation circuit.
    Returns:
        The estimated value (can be accessed with :attr:`.value`) with standard error
            of estimation (can be accessed with :attr:`.error`).
    """
    if not isinstance(op, Operator):
        op = Operator({op: 1.0})

    if len(op) == 0:
        return _ConstEstimate(0.0)

    if len(op) == 1 and PAULI_IDENTITY in op:
        return _ConstEstimate(op[PAULI_IDENTITY])

    # Support caching
    if isinstance(measurement, Iterable):
        measurement_groups = measurement
    else:
        measurement_groups = measurement(op)

    measurements, const = get_constant_seperated_measurement_group(
        op, measurement_groups
    )
    shots_map = distribute_shots_among_pauli_sets(
        op, measurements, shots_allocator, total_shots
    )
    circuit_and_shots = circuit_shot_pair_prep_fn(state, measurements, shots_map)
    sampling_counts = sampler(circuit_and_shots)
    return get_estimate_from_sampling_result(op, measurements, const, sampling_counts)


def create_sampling_estimator(
    total_shots: int,
    sampler: ConcurrentSampler,
    measurement_factory: CommutablePauliSetMeasurementFactory,
    shots_allocator: PauliSamplingShotsAllocator,
) -> QuantumEstimator[CircuitQuantumState]:
    """Create a :class:`QuantumEstimator` that estimates operator expectation
    value by sampling measurement.

    The sampling measurements are configured with arguments as follows.

    Args:
        total_shots: Total number of shots available for sampling measurements.
        sampler: A Sampler that actually performs the sampling measurements.
        measurement_factory: A function that performs Pauli grouping and returns
            a measurement scheme for Pauli operators constituting the original operator.
        shots_allocator: A function that allocates the total shots to Pauli groups to
            be measured.
    """

    def estimator(op: Estimatable, state: CircuitQuantumState) -> Estimate[complex]:
        return sampling_estimate(
            op, state, total_shots, sampler, measurement_factory, shots_allocator
        )

    return estimator


def concurrent_sampling_estimate(
    operators: Collection[Estimatable],
    states: Collection[CircuitQuantumState],
    total_shots: int,
    sampler: ConcurrentSampler,
    measurements: Union[
        CommutablePauliSetMeasurementFactory,
        Iterable[Iterable[CommutablePauliSetMeasurement]],
    ],
    shots_allocator: PauliSamplingShotsAllocator,
    circuit_shot_pair_prep_fn: CircuitShotPairPreparationFunction = default_prep,
) -> Iterable[Estimate[complex]]:
    """Estimate expectation value of given operators with given states by
    sampling measurement.

    The sampling measurements are configured with arguments as follows.

    Args:
        operators: Operators of which expectation value is estimated.
        states: Quantum states on which the operator expectation is evaluated.
        total_shots: Total number of shots available for sampling measurements.
        sampler: A Sampler that actually performs the sampling measurements.
        measurements: This can be either one of:
            - A function that performs Pauli grouping and returns a measurement
                scheme for Pauli operators constituting the original operator.
            - An iterable collection of Iterable[CommutablePauliSetMeasurement]
                where each element cooresponds to the grouping result of an input
                operator in ops.
        shots_allocator: A function that allocates the total shots to Pauli groups to
            be measured.

    Returns:
        The estimated values (can be accessed with :attr:`.value`) with standard errors
            of estimation (can be accessed with :attr:`.error`).
    """
    operators, states = parse_concurrent_estimator_arguments(operators, states)

    if isinstance(measurements, Iterable):
        measurements = list(measurements)
        if len(operators) != len(measurements):
            assert len(measurements) == 1
            measurements = [next(iter(measurements))] * len(operators)
        return [
            sampling_estimate(
                op,
                state,
                total_shots,
                sampler,
                m,
                shots_allocator,
                circuit_shot_pair_prep_fn,
            )
            for op, state, m in zip(operators, states, measurements)
        ]

    return [
        sampling_estimate(
            op,
            state,
            total_shots,
            sampler,
            measurements,
            shots_allocator,
            circuit_shot_pair_prep_fn,
        )
        for op, state in zip(operators, states)
    ]


def create_sampling_concurrent_estimator(
    total_shots: int,
    sampler: ConcurrentSampler,
    measurement_factory: CommutablePauliSetMeasurementFactory,
    shots_allocator: PauliSamplingShotsAllocator,
) -> ConcurrentQuantumEstimator[CircuitQuantumState]:
    """Create a :class:`ConcurrentQuantumEstimator` that estimates operator
    expectation value by sampling measurement.

    The sampling measurements are configured with arguments as follows.

    Args:
        total_shots: Total number of shots available for sampling measurements.
        sampler: A Sampler that actually performs the sampling measurements.
        measurement_factory: A function that performs Pauli grouping and returns
            a measurement scheme for Pauli operators constituting the original operator.
        shots_allocator: A function that allocates the total shots to Pauli groups to
            be measured.
    """

    def estimator(
        operators: Collection[Estimatable],
        states: Collection[CircuitQuantumState],
    ) -> Iterable[Estimate[complex]]:
        return concurrent_sampling_estimate(
            operators,
            states,
            total_shots,
            sampler,
            measurement_factory,
            shots_allocator,
        )

    return estimator


def create_fixed_operator_sampling_esimator(
    fixed_op: Estimatable,
    total_shots: int,
    sampler: ConcurrentSampler,
    measurement_factory: CommutablePauliSetMeasurementFactory,
    shots_allocator: PauliSamplingShotsAllocator,
) -> QuantumEstimator[CircuitQuantumState]:
    """Returns an estimator that performs sampling estimation for the specified
    operator.

    Args:
        fixed_op: The operator to be grouped and cached.
        total_shots: Total number of shots available for sampling measurements.
        sampler: A Sampler that actually performs the sampling measurements.
        measurement_factory: A function that performs Pauli grouping and returns
            a measurement scheme for Pauli operators constituting the original operator.
        shots_allocator: A function that allocates the total shots to Pauli groups to
            be measured.
    """
    if not isinstance(fixed_op, Operator):
        fixed_op = Operator({fixed_op: 1.0})

    measurement_groups = measurement_factory(fixed_op)

    def estimator(op: Estimatable, state: CircuitQuantumState) -> Estimate[complex]:
        if not isinstance(fixed_op, Operator):
            op = Operator({op: 1.0})
        assert op == fixed_op, (
            "The input operator is not consistent with the "
            "fixed operator used to create the estimator."
        )
        return sampling_estimate(
            op, state, total_shots, sampler, measurement_groups, shots_allocator
        )

    return estimator


def create_fixed_operator_sampling_concurrent_esimator(
    fixed_ops: Sequence[Estimatable],
    total_shots: int,
    sampler: ConcurrentSampler,
    measurement_factory: CommutablePauliSetMeasurementFactory,
    shots_allocator: PauliSamplingShotsAllocator,
) -> ConcurrentQuantumEstimator[CircuitQuantumState]:
    """Returns an estimator that performs sampling estimation for the specified
    operator.

    Args:
        fixed_ops: The operators to be grouped and cached.
        total_shots: Total number of shots available for sampling measurements.
        sampler: A Sampler that actually performs the sampling measurements.
        measurement_factory: A function that performs Pauli grouping and returns
            a measurement scheme for Pauli operators constituting the original operator.
        shots_allocator: A function that allocates the total shots to Pauli groups to
            be measured.
    """
    measurements = [
        measurement_factory(fixed_op)
        if isinstance(fixed_op, Operator)
        else measurement_factory(Operator({fixed_op: 1}))
        for fixed_op in fixed_ops
    ]

    def concurrent_estimator(
        ops: Sequence[Estimatable], states: Sequence[CircuitQuantumState]
    ) -> Iterable[Estimate[complex]]:
        if len(fixed_ops) < len(ops):
            assert len(fixed_ops) == 1
            _measurements = measurements * len(ops)
            for op in ops:
                assert op == fixed_ops[0]
        elif len(fixed_ops) > len(ops):
            _measurements = measurements
            assert len(ops) == 1
            for op in fixed_ops:
                assert op == ops[0]
        else:
            _measurements = measurements
            for op, fixed_op in zip(ops, fixed_ops):
                assert op == fixed_op

        return concurrent_sampling_estimate(
            ops,
            states,
            total_shots,
            sampler,
            _measurements,
            shots_allocator,
        )

    return concurrent_estimator
