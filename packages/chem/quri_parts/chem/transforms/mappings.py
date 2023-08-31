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
from typing import Optional, Protocol

from . import (
    BravyiKitaevMapperFactory,
    FermionQubitMapperFactory,
    FermionQubitStateMapper,
    JordanWignerMapperFactory,
    QubitFermionStateMapper,
    SymmetryConservingBravyiKitaevMapperFactory,
)


class FermionQubitMapping(Protocol):
    mapping_method: type[FermionQubitMapperFactory]
    _n_spin_orbitals: Optional[int]

    @property
    def n_spin_orbitals(self) -> Optional[int]:
        return self._n_spin_orbitals

    @property
    def n_qubits(self) -> Optional[int]:
        if self._n_spin_orbitals is None:
            return None
        return self.mapping_method.n_qubits_required(self._n_spin_orbitals)

    @abstractproperty
    def state_mapper(self) -> FermionQubitStateMapper:
        ...

    @abstractproperty
    def inv_state_mapper(self) -> QubitFermionStateMapper:
        ...


class JordanWigner(FermionQubitMapping, ABC):
    mapping_method = JordanWignerMapperFactory


class BravyiKitaev(FermionQubitMapping, ABC):
    mapping_method = BravyiKitaevMapperFactory


class SymmetryConservingBravyiKitaev(FermionQubitMapping, ABC):
    mapping_method = SymmetryConservingBravyiKitaevMapperFactory
