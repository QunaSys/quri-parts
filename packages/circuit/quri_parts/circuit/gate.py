# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from typing import NamedTuple


class QuantumGate(NamedTuple):
    """Non-parametric quantum gate.

    Not intended for direct use. Every gate is created by factory
    methods. A QuantumGate object contains information of gate name,
    control qubit, target qubit, classical bits, parameters, and pauli
    ids.
    """

    name: str
    target_indices: Sequence[int]
    control_indices: Sequence[int] = ()
    classical_indices: Sequence[int] = ()
    params: Sequence[float] = ()
    pauli_ids: Sequence[int] = ()
    unitary_matrix: Sequence[Sequence[complex]] = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuantumGate):
            return False
        if self.name != other.name:
            return False
        if set(self.control_indices) != set(other.control_indices):
            return False
        if self.params != other.params:
            return False
        if set(zip(self.target_indices, self.pauli_ids)) != set(
            zip(other.target_indices, other.pauli_ids)
        ):
            return False
        if self.name == "Measurement":
            return set(zip(self.target_indices, self.classical_indices)) == set(
                zip(other.target_indices, other.classical_indices)
            )
        return self.unitary_matrix == other.unitary_matrix


class ParametricQuantumGate(NamedTuple):
    """Parametric quantum gate.

    Not intended for direct use. Every gate is created through factory
    methods. A ParametricQuantumGate object contains information of gate
    name, control qubit, target qubit, and pauli ids.
    """

    name: str
    target_indices: Sequence[int]
    control_indices: Sequence[int] = ()
    pauli_ids: Sequence[int] = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParametricQuantumGate):
            return False
        if self.name != other.name:
            return False
        if set(self.control_indices) != set(other.control_indices):
            return False
        return set(zip(self.target_indices, self.pauli_ids)) == set(
            zip(other.target_indices, other.pauli_ids)
        )
