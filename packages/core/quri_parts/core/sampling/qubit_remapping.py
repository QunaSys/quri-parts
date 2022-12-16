# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable, Mapping
from typing import Union

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.transpile import QubitRemappingTranspiler

from . import ConcurrentSampler, MeasurementCounts, Sampler


def _create_reverse_map(qubit_remapping: Mapping[int, int]) -> Mapping[int, int]:
    return {1 << v: 1 << k for k, v in qubit_remapping.items()}


def _reverse_map_bits(x: int, reverse_bit_map: Mapping[int, int]) -> int:
    result = 0
    for mapped_bits, original_bits in reverse_bit_map.items():
        if (x & mapped_bits) != 0:
            result += original_bits
    return result


def _reverse_map_counts(
    m: MeasurementCounts, reverse_bit_map: Mapping[int, int]
) -> MeasurementCounts:
    d: dict[int, Union[int, float]] = {}
    for b, count in m.items():
        reversed_bits = _reverse_map_bits(b, reverse_bit_map)
        d[reversed_bits] = d.get(reversed_bits, 0) + count
    return d


def create_qubit_remapping_sampler(
    sampler: Sampler, qubit_remapping: Mapping[int, int]
) -> Sampler:
    """Create a :class:`~Sampler` that remap qubits in the circuit before
    sampling.

    The keys (measured bits) in the measurement counts returned by the
    Sampler are represented in terms of qubits before the remapping, so
    you do not need to take the remapping into consideration when
    handling them. The mapping ``qubit_mapping`` should be specified
    with "from" qubit indices as keys and "to" qubit indices as values.
    For example, if you want to convert a circuit using qubits 0, 1, 2,
    3 by mapping them as 0 → 4, 1 → 2, 2 → 5, 3 → 0, then the
    ``qubit_mapping`` should be ``{0: 4, 1: 2, 2: 5, 3: 0}``.
    """
    transpiler = QubitRemappingTranspiler(qubit_remapping)
    reverse_bit_map = _create_reverse_map(qubit_remapping)

    def remapping_sampler(
        circuit: NonParametricQuantumCircuit, n_shots: int
    ) -> MeasurementCounts:
        transpiled_circuit = transpiler(circuit)
        m = sampler(transpiled_circuit, n_shots)
        return _reverse_map_counts(m, reverse_bit_map)

    return remapping_sampler


def create_qubit_remapping_concurrent_sampler(
    sampler: ConcurrentSampler, qubit_remapping: Mapping[int, int]
) -> ConcurrentSampler:
    """Create a :class:`~ConcurrentSampler` that remap qubits in the circuit
    before sampling.

    The keys (measured bits) in the measurement counts returned by the
    Sampler are represented in terms of qubits before the remapping, so
    you do not need to take the remapping into consideration when
    handling them. The mapping ``qubit_mapping`` should be specified
    with "from" qubit indices as keys and "to" qubit indices as values.
    For example, if you want to convert a circuit using qubits 0, 1, 2,
    3 by mapping them as 0 → 4, 1 → 2, 2 → 5, 3 → 0, then the
    ``qubit_mapping`` should be ``{0: 4, 1: 2, 2: 5, 3: 0}``.
    """
    transpiler = QubitRemappingTranspiler(qubit_remapping)
    reverse_bit_map = _create_reverse_map(qubit_remapping)

    def remapping_sampler(
        shot_circuit_pairs: Iterable[tuple[NonParametricQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        transpiled = map(lambda t: (transpiler(t[0]), t[1]), shot_circuit_pairs)
        m = sampler(transpiled)
        return map(lambda m: _reverse_map_counts(m, reverse_bit_map), m)

    return remapping_sampler
