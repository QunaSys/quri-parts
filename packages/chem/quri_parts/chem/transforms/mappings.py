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
    _mapping_method: type[FermionQubitMapperFactory]

    @abstractproperty
    def n_spin_orbitals(self) -> Optional[int]:
        ...

    @abstractproperty
    def n_qubits(self) -> Optional[int]:
        ...

    @abstractproperty
    def state_mapper(self) -> FermionQubitStateMapper:
        ...

    @abstractproperty
    def inv_state_mapper(self) -> QubitFermionStateMapper:
        ...


class JordanWigner(FermionQubitMapping, ABC):
    _mapping_method = JordanWignerMapperFactory


class BravyiKitaev(FermionQubitMapping, ABC):
    _mapping_method = BravyiKitaevMapperFactory


class SymmetryConservingBravyiKitaev(FermionQubitMapping, ABC):
    _mapping_method = SymmetryConservingBravyiKitaevMapperFactory
