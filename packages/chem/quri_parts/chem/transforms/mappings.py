# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractproperty
from collections.abc import Collection
from typing import Callable, Optional, Protocol

from typing_extensions import TypeAlias

from quri_parts.core.state import ComputationalBasisState

#: Interface for a function that maps a collection of occupied spin orbital indices to
#: a computational basis state of qubits.
#: Note that the mapping does not depend on the order of the occupied indices.
#: A computational basis state with a positive sign should always be returned.
FermionQubitStateMapper: TypeAlias = Callable[
    [Collection[int]], ComputationalBasisState
]

#: Interface for a function that maps a computational basis state of qubits to
#: a collection of occupied spin orbital indices.
QubitFermionStateMapper: TypeAlias = Callable[
    [ComputationalBasisState], Collection[int]
]


class FermionQubitMapping(Protocol):
    @abstractproperty
    def n_spin_orbitals(self) -> Optional[int]:
        ...

    @abstractproperty
    def n_qubits(self) -> Optional[int]:
        ...

    @abstractproperty
    def state_mapper(self) -> "FermionQubitStateMapper":
        ...

    @abstractproperty
    def inv_state_mapper(self) -> "QubitFermionStateMapper":
        ...


def jordan_wigner_n_qubits_required(n_spin_orbtals: int) -> int:
    return n_spin_orbtals


def jordan_wigner_n_spin_orbitals(n_qubits: int) -> int:
    return n_qubits


class JordanWigner(FermionQubitMapping, ABC):
    @property
    def n_qubits(self) -> Optional[int]:
        if self.n_spin_orbitals is None:
            return None
        return jordan_wigner_n_qubits_required(self.n_spin_orbitals)


def bravyi_kitaev_n_qubits_required(n_spin_orbtals: int) -> int:
    return n_spin_orbtals


def bravyi_kitaev_n_spin_orbitals(n_qubits: int) -> int:
    return n_qubits


class BravyiKitaev(FermionQubitMapping, ABC):
    @property
    def n_qubits(self) -> Optional[int]:
        if self.n_spin_orbitals is None:
            return None
        return bravyi_kitaev_n_qubits_required(self.n_spin_orbitals)


def symmetry_conserving_bravyi_kitaev_n_qubits_required(n_spin_orbtals: int) -> int:
    return n_spin_orbtals - 2


def symmetry_conserving_bravyi_kitaev_n_spin_orbitals(n_qubits: int) -> int:
    return n_qubits + 2


class SymmetryConservingBravyiKitaev(FermionQubitMapping, ABC):
    @property
    def n_qubits(self) -> Optional[int]:
        if self.n_spin_orbitals is None:
            return None
        return symmetry_conserving_bravyi_kitaev_n_qubits_required(self.n_spin_orbitals)
