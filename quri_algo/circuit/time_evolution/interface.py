# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from quri_parts.circuit import (
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    NonParametricQuantumCircuit,
)
from quri_parts.core.operator import Operator, pauli_label

from quri_algo.circuit.interface import ProblemCircuitFactory
from quri_algo.circuit.utils.transpile import apply_transpiler
from quri_algo.problem import HamiltonianInput, HamiltonianT, QubitHamiltonianInput


@dataclass
class TimeEvolutionCircuitFactory(ProblemCircuitFactory[HamiltonianT], ABC):
    """Encode a Hamiltonian to a time evolution circuit."""

    def __post_init__(self) -> None:
        assert isinstance(self.encoded_problem, HamiltonianInput)

    @apply_transpiler  # type: ignore
    @abstractmethod
    def __call__(
        self, evolution_time: float, *args: Any, **kwds: Any
    ) -> NonParametricQuantumCircuit:
        ...


@dataclass
class ControlledTimeEvolutionCircuitFactory(ProblemCircuitFactory[HamiltonianT], ABC):
    """Encode a Hamiltonian to a controlled-time evolution circuit."""

    def __post_init__(self) -> None:
        assert isinstance(self.encoded_problem, HamiltonianInput)

    @apply_transpiler  # type: ignore
    @abstractmethod
    def __call__(self, evolution_time: float) -> NonParametricQuantumCircuit:
        ...


@dataclass
class PartialTimeEvolutionCircuitFactory(
    TimeEvolutionCircuitFactory[QubitHamiltonianInput], ABC
):
    def get_local_hamiltonian_input(
        self, idx0: int, idx1: int
    ) -> QubitHamiltonianInput:
        def within_range(j: int) -> bool:
            return j >= idx0 and j <= idx1

        local_hamiltonian = Operator()
        for op, coef in self.encoded_problem.qubit_hamiltonian.items():
            if all(map(within_range, op.index_and_pauli_id_list[0])):
                idx_id_iterable = [
                    (idx - idx0, id) for idx, id in zip(*op.index_and_pauli_id_list)
                ]
                local_hamiltonian.add_term(pauli_label(idx_id_iterable), coef)

        return QubitHamiltonianInput(idx1 - idx0 + 1, local_hamiltonian)

    @abstractmethod
    def get_partial_time_evolution_circuit(
        self, idx0: int, idx1: int
    ) -> ImmutableLinearMappedUnboundParametricQuantumCircuit:
        pass

    @apply_transpiler  # type: ignore
    @abstractmethod
    def __call__(
        self, evolution_time: float, idx0: int, idx1: int
    ) -> NonParametricQuantumCircuit:
        ...
