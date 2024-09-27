# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Collection,
    Generic,
    Iterable,
    Mapping,
    NamedTuple,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

from quri_parts.backend import SamplingBackend
from quri_parts.circuit import (
    ImmutableQuantumCircuit,
    UnboundParametricQuantumCircuitProtocol,
)
from quri_parts.core.operator import CommutablePauliSet, Operator
from quri_parts.core.state import (
    CircuitQuantumState,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
    QuantumStateVector,
)

#: A type variable represents *any* non-parametric quantum state classes.
#: This is different from :class:`quri_parts.core.state.QuantumStateT`;
#: ``QuantumStateT`` represents *either one of* the classes, while ``_StateT`` also
#: covers *a union of* multiple state classes.
_StateT = TypeVar("_StateT", bound=Union[CircuitQuantumState, QuantumStateVector])

#: A type variable represents *any* parametric quantum state classes.
#: This is different from :class:`quri_parts.core.state.ParametricQuantumStateT`;
#: ``ParametricQuantumStateT`` represents *either one of* the classes, while
#: ``_ParametricStateT`` also covers *a union of* multiple state classes.
_ParametricStateT = TypeVar(
    "_ParametricStateT",
    bound=Union[ParametricCircuitQuantumState, ParametricQuantumStateVector],
)

#: MeasurementCounts represents count statistics of repeated measurements of a quantum
#: circuit. Keys are observed bit patterns encoded in integers and values are counts
#: of observation of the corresponding bit patterns.
MeasurementCounts: TypeAlias = Mapping[int, Union[int, float]]

#: Sampler represents a function that samples a specified (non-parametric) circuit by
#: a specified times and returns the count statistics. In the case of an ideal Sampler,
# the return value corresponds to probabilities multiplied by shot count.
Sampler: TypeAlias = Callable[[ImmutableQuantumCircuit, int], MeasurementCounts]

#: ConcurrentSampler represents a function that samples specified (non-parametric)
#: circuits concurrently.
ConcurrentSampler: TypeAlias = Callable[
    [Iterable[tuple[ImmutableQuantumCircuit, int]]], Iterable[MeasurementCounts]
]


#: ParametricSampler represents a sampler that samples from a parametric circuit with
#: a fixed set of circuit parameters.
ParametricSampler: TypeAlias = Callable[
    [UnboundParametricQuantumCircuitProtocol, int, Sequence[float]], MeasurementCounts
]


#: ConcurrentParametricSampler represents a sampler that samples from a parametric
#: circuit with a sequence of circuit parameter list.
ConcurrentParametricSampler: TypeAlias = Callable[
    [Iterable[tuple[UnboundParametricQuantumCircuitProtocol, int, Sequence[float]]]],
    Iterable[MeasurementCounts],
]


#: StateSampler representes a function that samples a specific (non-parametric) state by
#: specified times and returns the count statistics. In the case of an ideal
#: StateSampler, the return value corresponds to probabilities multiplied by shot count.
StateSampler: TypeAlias = Callable[[_StateT, int], MeasurementCounts]

#: ConcurrentSampler represents a function that samples specified (non-parametric)
#: state by specified times and returns the count statistics concurrently.
ConcurrentStateSampler: TypeAlias = Callable[
    [Iterable[tuple[_StateT, int]]], Iterable[MeasurementCounts]
]


#: ParametricStateSampler represents a state sampler that samples from a
#: parametric state with a fixed set of circuit parameters.
ParametricStateSampler: TypeAlias = Callable[
    [_ParametricStateT, int, Sequence[float]], MeasurementCounts
]


#: ConcurrentParametricStateSampler represents a state sampler that samples from a
#: parametric state with a sequence of circuit parameter list.
ConcurrentParametricStateSampler: TypeAlias = Callable[
    [Iterable[tuple[_ParametricStateT, int, Sequence[float]]]],
    Iterable[MeasurementCounts],
]


@dataclass
class GeneralSampler(Generic[_StateT, _ParametricStateT]):
    sampler: Sampler
    state_sampler: StateSampler[_StateT]
    parametric_sampler: ParametricSampler = field(init=False)
    parametric_state_sampler: ParametricStateSampler[_ParametricStateT] = field(
        init=False
    )

    def __post_init__(self) -> None:
        self.parametric_sampler = create_parametric_sampler_from_sampler(self.sampler)
        self.parametric_state_sampler = (
            create_parametric_state_sampler_from_state_sampler(self.state_sampler)
        )

    def __call__(
        self, *sampler_input: Any
    ) -> Union[MeasurementCounts, Iterable[MeasurementCounts]]:
        if isinstance(sampler_input[0], NonParametricQuantumCircuit):
            assert isinstance(sampler_input[1], int), "Shot is not specified properly."
            return self.sampler(*sampler_input)
        if isinstance(sampler_input[0], UnboundParametricQuantumCircuitProtocol):
            assert isinstance(sampler_input[1], int), "Shot is not specified properly."
            assert isinstance(
                sampler_input[2], Iterable
            ), "Circuit parameter is not specified properly."
            return self.parametric_sampler(*sampler_input)
        if isinstance(sampler_input[0], (CircuitQuantumState, QuantumStateVector)):
            assert isinstance(sampler_input[1], int), "Shot is not specified properly."
            return self.state_sampler(*sampler_input)
        if isinstance(
            sampler_input[0],
            (ParametricCircuitQuantumState, ParametricQuantumStateVector),
        ):
            assert isinstance(sampler_input[1], int), "Shot is not specified properly."
            return self.parametric_state_sampler(*sampler_input)

        if isinstance(sampler_input[0][0], Iterable):
            return self(*sampler_input[0])

        return [cast(MeasurementCounts, self(*comb)) for comb in sampler_input]


def create_parametric_sampler_from_sampler(sampler: Sampler) -> ParametricSampler:
    """Create a :class:`ParametricSampler` from a :class:`Sampler`."""

    def _parametric_sampler(
        param_circuit: UnboundParametricQuantumCircuitProtocol,
        measurement_cnt: int,
        param: Sequence[float],
    ) -> MeasurementCounts:
        bound_circuit = param_circuit.bind_parameters(param)
        return sampler(bound_circuit, measurement_cnt)

    return _parametric_sampler


def create_concurrent_parametric_sampler_from_concurrent_sampler(
    concurrent_sampler: ConcurrentSampler,
) -> ConcurrentParametricSampler:
    """Create a :class:`ConcurrentParametricSampler` from a
    :class:`ConcurrentSampler`."""

    def _concurrent_parametric_sampler(
        circuit_shot_param_tuples: Iterable[
            tuple[UnboundParametricQuantumCircuitProtocol, int, Sequence[float]]
        ]
    ) -> Iterable[MeasurementCounts]:
        circuit_shot_tuples = [
            (circuit.bind_parameters(param), shot)
            for circuit, shot, param in circuit_shot_param_tuples
        ]
        return concurrent_sampler(circuit_shot_tuples)

    return _concurrent_parametric_sampler


def create_parametric_state_sampler_from_state_sampler(
    state_sampler: StateSampler[_StateT],
) -> ParametricStateSampler[_ParametricStateT]:
    """Create a :class:`ParametricStateSampler` from a
    :class:`StateSampler`."""

    def _parametric_state_sampler(
        param_state: _ParametricStateT,
        measurement_cnt: int,
        param: Sequence[float],
    ) -> MeasurementCounts:
        bound_state = cast(_StateT, param_state.bind_parameters(param))
        return state_sampler(bound_state, measurement_cnt)

    return _parametric_state_sampler


def create_concurrent_parametric_state_sampler_from_concurrent_state_sampler(
    concurrent_state_sampler: ConcurrentStateSampler[_StateT],
) -> ConcurrentParametricStateSampler[_ParametricStateT]:
    """Create a :class:`ConcurrentParametricStateSampler` from a
    :class:`ConcurrentStateSampler`.
    """

    def _concurrent_parametric_state_sampler(
        state_shot_param_tuples: Iterable[
            tuple[_ParametricStateT, int, Sequence[float]]
        ]
    ) -> Iterable[MeasurementCounts]:
        state_shot_tuples = [
            (cast(_StateT, state.bind_parameters(param)), shot)
            for state, shot, param in state_shot_param_tuples
        ]
        return concurrent_state_sampler(state_shot_tuples)

    return _concurrent_parametric_state_sampler


def sample_from_state_vector(
    state_vector: npt.NDArray[np.complex128], n_shots: int
) -> MeasurementCounts:
    """Perform sampling from a state vector."""
    n_qubits: float = np.log2(state_vector.shape[0])
    assert n_qubits.is_integer(), "Length of the state vector must be a power of 2."
    if not np.isclose(np.linalg.norm(state_vector), 1):
        raise ValueError("probabilities do not sum to 1")
    probs = np.abs(state_vector) ** 2
    rng = np.random.default_rng()
    counts = rng.multinomial(n_shots, probs)
    return Counter(dict(((i, count) for i, count in enumerate(counts) if count > 0)))


def ideal_sample_from_state_vector(
    state_vector: npt.NDArray[np.complex128], n_shots: int
) -> MeasurementCounts:
    """Perform ideal sampling from a state vector."""
    n_qubits: float = np.log2(state_vector.shape[0])
    assert n_qubits.is_integer(), "Length of the state vector must be a power of 2."
    if not np.isclose(np.linalg.norm(state_vector), 1):
        raise ValueError("probabilities do not sum to 1")

    probs = np.abs(state_vector) ** 2
    return {i: prob * n_shots for i, prob in enumerate(probs)}


def create_sampler_from_sampling_backend(backend: SamplingBackend) -> Sampler:
    """Create a simple :class:`~Sampler` using a :class:`~SamplingBackend`."""

    def sampler(circuit: ImmutableQuantumCircuit, n_shots: int) -> MeasurementCounts:
        job = backend.sample(circuit, n_shots)
        return job.result().counts

    return sampler


def create_concurrent_sampler_from_sampling_backend(
    backend: SamplingBackend,
) -> ConcurrentSampler:
    """Create a simple :class:`~ConcurrentSampler` using a
    :class:`~SamplingBackend`."""

    def sampler(
        shot_circuit_pairs: Iterable[tuple[ImmutableQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        jobs = [
            backend.sample(circuit, n_shots) for circuit, n_shots in shot_circuit_pairs
        ]
        return map(lambda j: j.result().counts, jobs)

    return sampler


def create_sampler_from_concurrent_sampler(
    concurrent_sampler: ConcurrentSampler,
) -> Sampler:
    def sampler(circuit: ImmutableQuantumCircuit, shots: int) -> MeasurementCounts:
        return next(iter(concurrent_sampler([(circuit, shots)])))

    return sampler


class PauliSamplingSetting(NamedTuple):
    pauli_set: CommutablePauliSet
    n_shots: int


#: PauliSamplingShotsAllocator represents a function that distributes
#: a given number of sampling shots to each :class:`~CommutablePauliSet`.
PauliSamplingShotsAllocator: TypeAlias = Callable[
    [Operator, Collection[CommutablePauliSet], int], Collection[PauliSamplingSetting]
]


#: WeightedSamplingShotsAllocator represents a function that distributes
#: a given number of sampling shots based on a set of weights.
WeightedSamplingShotsAllocator: TypeAlias = Callable[
    [Sequence[complex], int], Sequence[int]
]


__all__ = [
    "MeasurementCounts",
    "Sampler",
    "ConcurrentSampler",
    "create_sampler_from_sampling_backend",
    "create_concurrent_sampler_from_sampling_backend",
    "create_sampler_from_concurrent_sampler",
    "PauliSamplingSetting",
    "PauliSamplingShotsAllocator",
    "WeightedSamplingShotsAllocator",
]
