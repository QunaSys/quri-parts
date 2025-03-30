# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

from quri_parts.circuit import (
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    NonParametricQuantumCircuit,
)
from quri_parts.core.operator import Operator, pauli_label

from quri_algo.circuit.interface import CircuitFactory
from quri_algo.circuit.utils.transpile import apply_transpiler
from quri_algo.problem import HamiltonianT, QubitHamiltonianInput


@runtime_checkable
class TimeEvolutionCircuitFactory(CircuitFactory, Protocol):
    """Encode a Hamiltonian to a time evolution circuit."""

    @apply_transpiler  # type: ignore
    @abstractmethod
    def __call__(self, evolution_time: float) -> NonParametricQuantumCircuit:
        ...


@runtime_checkable
class ControlledTimeEvolutionCircuitFactory(CircuitFactory, Protocol):
    """Encode a Hamiltonian to a controlled-time evolution circuit."""

    @apply_transpiler  # type: ignore
    @abstractmethod
    def __call__(self, evolution_time: float) -> NonParametricQuantumCircuit:
        ...


# class PartialTimeEvolutionCircuitFactory(CircuitFactory, Protocol):
#     def get_local_hamiltonian_input(
#         self, idx0: int, idx1: int
#     ) -> QubitHamiltonianInput:
#         def within_range(j: int) -> bool:
#             return j >= idx0 and j <= idx1

#         local_hamiltonian = Operator()
#         for op, coef in self.encoded_problem.qubit_hamiltonian.items():
#             if all(map(within_range, op.index_and_pauli_id_list[0])):
#                 idx_id_iterable = [
#                     (idx - idx0, id) for idx, id in zip(*op.index_and_pauli_id_list)
#                 ]
#                 local_hamiltonian.add_term(pauli_label(idx_id_iterable), coef)

#         return QubitHamiltonianInput(idx1 - idx0 + 1, local_hamiltonian)

#     @abstractmethod
#     def get_partial_time_evolution_circuit(
#         self, idx0: int, idx1: int
#     ) -> ImmutableLinearMappedUnboundParametricQuantumCircuit:
#         pass

#     @apply_transpiler  # type: ignore
#     @abstractmethod
#     def __call__(
#         self, evolution_time: float, idx0: int, idx1: int
#     ) -> NonParametricQuantumCircuit:
#         ...
