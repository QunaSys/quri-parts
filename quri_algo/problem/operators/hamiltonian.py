# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Protocol, TypeVar

from openfermion import FermionOperator
from quri_parts.core.operator import Operator

from .interface import OperatorProtocol


class HamiltonianInput(OperatorProtocol, Protocol):
    """Represents an encoded Hamiltonian."""

    ...


HamiltonianT = TypeVar("HamiltonianT", bound="HamiltonianInput")


class QubitHamiltonianInput(HamiltonianInput):
    """Represents the Hamiltonian of a fixed qubit size in its qubit
    Hamiltonian form."""

    def __init__(self, n_state_qubit: int, qubit_hamiltonian: Operator):
        self.n_state_qubit = n_state_qubit
        self._qubit_hamiltonian = qubit_hamiltonian

    @property
    def qubit_hamiltonian(self) -> Operator:
        return self._qubit_hamiltonian.copy()


class FermionicHamiltonianInput(HamiltonianInput):
    def __init__(self, n_spin_orbital: int, fermionic_hamiltonian: FermionOperator):
        self.n_spin_orbital = n_spin_orbital
        self._fermionic_hamiltonian = fermionic_hamiltonian

    @property
    def fermionic_hamiltonian(self) -> FermionOperator:
        return self._fermionic_hamiltonian
