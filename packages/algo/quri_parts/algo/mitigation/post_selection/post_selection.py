# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Iterable

from typing_extensions import TypeAlias

from quri_parts.circuit import ImmutableQuantumCircuit
from quri_parts.core.sampling import ConcurrentSampler, MeasurementCounts, Sampler

#: Represents a filter function for post-selection.
PostSelectionFilterFunction: TypeAlias = Callable[[int], bool]


def post_selection(
    filter_fn: PostSelectionFilterFunction, meas_counts: MeasurementCounts
) -> MeasurementCounts:
    """Functions that filter out unwanted measurement results from given
    ``meas_counts``."""
    return {k: v for k, v in meas_counts.items() if filter_fn(k)}


def create_general_post_selection_sampler(
    sampler: Sampler, filter_fn: PostSelectionFilterFunction
) -> Sampler:
    """Returns :class:`Sampler` that performs post-selection after sampling.

    Note that the effective shots might decrease by discarding the
    measurement results which do not pass ``filter_fn``.
    """

    def post_selection_sampler(
        circuit: ImmutableQuantumCircuit, shots: int
    ) -> MeasurementCounts:
        meas_counts = sampler(circuit, shots)
        filtered_counts = post_selection(filter_fn, meas_counts)
        return filtered_counts

    return post_selection_sampler


def create_general_post_selection_concurrent_sampler(
    concurrent_sampler: ConcurrentSampler, filter_fn: PostSelectionFilterFunction
) -> ConcurrentSampler:
    """Returns :class:`ConcurrentSampler` that performs post-selection after
    sampling.

    Note that the effective shots might decrease by discarding the
    measurement results which do not pass ``filter_fn``.
    """

    def post_selection_concurrent_sampler(
        circuit_shots_tuples: Iterable[tuple[ImmutableQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        meas_counts = concurrent_sampler(circuit_shots_tuples)
        filtered_counts = []
        for meas_count in meas_counts:
            post_selected = post_selection(filter_fn, meas_count)
            filtered_counts.append(post_selected)

        return filtered_counts

    return post_selection_concurrent_sampler
