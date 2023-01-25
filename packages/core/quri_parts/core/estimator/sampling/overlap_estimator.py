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

from quri_parts.circuit import inverse_circuit
from quri_parts.core.estimator import (
    ConcurrentOverlapEstimator,
    Estimate,
    OverlapEstimator,
)
from quri_parts.core.sampling import (
    ConcurrentSampler,
    MeasurementCounts,
    WeightedSamplingShotsAllocator,
)
from quri_parts.core.state import CircuitQuantumState


class _OverlapEstimate:
    def __init__(
        self, sampling_counts: Sequence[MeasurementCounts], weights: Sequence[complex]
    ):
        self._sampling_counts = sampling_counts
        self._total_counts = [sum(counts.values()) for counts in self._sampling_counts]
        self._weights = weights

    @cached_property
    def value(self) -> complex:
        # should be calculated with weights (if direct answer is desired use weights =
        # [1.0]) penalty terms can be calculated with weights = [1.0, 2.0, ...]. Users
        # can decide whether to use this for penalty terms or pure overlap magnitude
        # squared
        val = sum(
            weight * counts[0] / total_counts
            for weight, counts, total_counts in zip(
                self._weights, self._sampling_counts, self._total_counts
            )
        )
        return val

    @cached_property
    def error(self) -> float:
        square_err = sum(
            abs(weight) ** 2 * counts[0] / total_counts**2
            for weight, counts, total_counts in zip(
                self._weights, self._sampling_counts, self._total_counts
            )
        )
        return sqrt(square_err)


@dataclass
class _ConstOverlapEstimate:
    value: complex
    error: float = 0.0


def sampling_overlap_estimate(
    bras: Sequence[CircuitQuantumState],
    kets: Sequence[CircuitQuantumState],
    weights: Sequence[complex],
    total_shots: int,
    sampler: ConcurrentSampler,
    shots_allocator: WeightedSamplingShotsAllocator[complex],
) -> Estimate[complex]:
    """Estimate the overlap of two quantum states by a sampling measurement

    The sampling measurements are configured with arguments as follows.

    Args:
        bras: The states whose inverse circuit is used
        kets: The states whose direct circuit is used
        weights: Weights used in summing the overlaps
        total_shots: Nmber of shots used for sampling measurements.
        sampler: A Sampler that actually performs the sampling measurements.
        shots_allocator: shot allocator which takes weights as an argument.

    Returns:
        The estimated value (can be accessed with :attr:`.value`) with standard error
            of estimation (can be accessed with :attr:`.error`).
    """
    len_kets = len(kets)
    len_bras = len(bras)
    len_weights = len(weights)

    if any(len_kets != x for x in {len_bras, len_weights}):
        raise ValueError(
            f"Number of kets ({len_kets}) and "
            f"number of bras ({len_bras}) and "
            f"number of weights ({len_weights}) does not match."
        )

    if len_kets == 0:
        return _ConstOverlapEstimate(0.0)

    shots = shots_allocator(weights, total_shots)

    circuits_and_shots = [
        (
            ket.circuit.get_mutable_copy()
            + inverse_circuit(bra.circuit.get_mutable_copy()),
            shot_count,
        )
        for ket, bra, shot_count in zip(kets, bras, shots)
    ]
    counts = list(sampler(circuits_and_shots))

    return _OverlapEstimate(counts, weights)


def create_sampling_overlap_estimator(
    total_shots: int,
    sampler: ConcurrentSampler,
    shots_allocator: WeightedSamplingShotsAllocator[complex],
) -> OverlapEstimator[CircuitQuantumState, complex]:
    """Create a :class:`OverlapEstimator` that estimates the squared magnitude of the
    overlap of pairs of wavefunctions by sampling measurement and returns a weighted
    sum.

    The sampling measurements are configured with arguments as follows.

    Args:
        total_shots: Total number of shots available for sampling measurements.
        sampler: A Sampler that actually performs the sampling measurements.
        shots_allocator: shot allocator which takes weights as an argument.
    """

    def estimator(
        kets: Sequence[CircuitQuantumState],
        bras: Sequence[CircuitQuantumState],
        weights: Sequence[complex],
    ) -> Estimate[complex]:
        return sampling_overlap_estimate(
            kets, bras, weights, total_shots, sampler, shots_allocator
        )

    return estimator


def concurrent_sampling_overlap_estimate(
    kets: Collection[Sequence[CircuitQuantumState]],
    bras: Collection[Sequence[CircuitQuantumState]],
    weights: Collection[Sequence[complex]],
    total_shots: int,
    sampler: ConcurrentSampler,
    shots_allocator: WeightedSamplingShotsAllocator[complex],
) -> Iterable[Estimate[complex]]:
    """This function estimates overlaps of pairs organized in weighted sequences,
    collections of which are estimated concurrently.

    The sampling measurements are configured with arguments as follows.

    Args:
        kets: The states whose direct circuit is used
        bras: The states whose inverse circuit is used
        weights: Weights used in summing the overlaps
        total_shots: Nmber of shots used for sampling measurements.
        sampler: A Sampler that actually performs the sampling measurements.
        shots_allocator: shot allocator which takes weights as an argument.

    Returns:
        The estimated value (can be accessed with :attr:`.value`) with standard error
            of estimation (can be accessed with :attr:`.error`).
    """
    num_kets = len(kets)
    num_bras = len(bras)
    num_weights = len(weights)

    if num_kets == 0 or num_bras == 0:
        raise ValueError("No states specified.")

    num_set = [num_kets, num_bras, num_weights]
    multiplier = max(num_set)

    def num_pop() -> Iterable[int]:
        while len(num_set) > 1:
            yield num_set.pop()

    if any([x != 1 and x not in num_set for x in num_pop()]):
        raise ValueError(
            f"Number of sequences of bras ({num_bras}) and"
            f"number of sequences of kets ({num_kets}) and"
            f"number of sequences of weights ({num_kets}) do not match."
        )

    if num_kets == 1:
        kets = [next(iter(kets))] * multiplier
    if num_bras == 1:
        bras = [next(iter(bras))] * multiplier
    if num_weights == 1:
        weights = [next(iter(weights))] * multiplier
    return [
        sampling_overlap_estimate(
            ket, bra, weight, total_shots, sampler, shots_allocator
        )
        for ket, bra, weight in zip(kets, bras, weights)
    ]


def create_sampling_concurrent_overlap_estimator(
    total_shots: int,
    sampler: ConcurrentSampler,
    shots_allocator: WeightedSamplingShotsAllocator[complex],
) -> ConcurrentOverlapEstimator[CircuitQuantumState, complex]:
    """Create a :class:`ConcurrentOverlapEstimator` that estimates weighted magnitude
    squared overlap values by sampling measurement.

    The sampling measurements are configured with arguments as follows.

    Args:
        total_shots: Total number of shots available for sampling measurements.
        sampler: A Sampler that actually performs the sampling measurements.
        shots_allocator: A function that allocates the total shots to Pauli groups to
            be measured.
    """

    def estimator(
        kets: Sequence[Sequence[CircuitQuantumState]],
        bras: Sequence[Sequence[CircuitQuantumState]],
        weights: Sequence[Sequence[complex]],
    ) -> Iterable[Estimate[complex]]:
        return concurrent_sampling_overlap_estimate(
            kets,
            bras,
            weights,
            total_shots,
            sampler,
            shots_allocator,
        )

    return estimator
