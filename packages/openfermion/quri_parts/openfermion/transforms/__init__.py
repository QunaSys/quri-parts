# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from quri_parts.chem.transforms import (
    BravyiKitaevMapperFactory,
    FermionQubitMapperFactory,
    FermionQubitStateMapper,
    JordanWignerMapperFactory,
    QubitFermionStateMapper,
    SymmetryConservingBravyiKitaevMapperFactory,
)

from .mappings import (
    OpenFermionBravyiKitaev,
    OpenFermionJordanWigner,
    OpenFermionQubitMapping,
    OpenFermionQubitOperatorMapper,
    OpenFermionSymmetryConservingBravyiKitaev,
)


class OpenFermionQubitMapperFactory(FermionQubitMapperFactory):
    """Mapping from Fermionic operators and states to :class:`Operator`s and
    states using OpenFermion."""

    _mapping_method: type[OpenFermionQubitMapping]

    def __call__(
        self,
        n_spin_orbitals: Optional[int] = None,
        n_fermions: Optional[int] = None,
        sz: Optional[float] = None,
    ) -> OpenFermionQubitMapping:
        return self._mapping_method(n_spin_orbitals, n_fermions, sz)

    def get_of_operator_mapper(
        self,
        n_spin_orbitals: Optional[int] = None,
        n_fermions: Optional[int] = None,
        sz: Optional[float] = None,
    ) -> OpenFermionQubitOperatorMapper:
        """Returns a function that maps an OpenFermion
        :class:`~openfermion.ops.FermionOperator`,
        :class:`~openfermion.ops.InteractionOperator` or
        :class:`~openfermion.ops.MajoranaOperator`
        to a :class:`openfermion.ops.QubitOperator`.

        Args:
            n_spin_orbitals:
                The number of spin orbitals to be mapped to qubits. Some mappings
                require this argument (e.g.
                :class:`~OpenFermionSymmetryConservingBravyiKitaev`) while others
                calculate it automatically when omitted.
            n_fermions:
                When specified, restrict the mapping to a subspace spanned by states
                containing the fixed number of Fermions. Some mappings require this
                argument (e.g. :class:`~OpenFermionSymmetryConservingBravyiKitaev`)
                while the others ignore it.
        """
        return self(n_spin_orbitals, n_fermions, sz).operator_mapper

    def get_state_mapper(
        self,
        n_spin_orbitals: int,
        n_fermions: Optional[int] = None,
        sz: Optional[float] = None,
    ) -> FermionQubitStateMapper:
        return self(n_spin_orbitals, n_fermions, sz).state_mapper

    def get_inv_state_mapper(
        self,
        n_spin_orbitals: int,
        n_fermions: Optional[int] = None,
        sz: Optional[float] = None,
    ) -> QubitFermionStateMapper:
        return self(n_spin_orbitals, n_fermions, sz).inv_state_mapper


class OpenFermionJordanWignerFactory(
    JordanWignerMapperFactory, OpenFermionQubitMapperFactory
):
    """Jordan-Wigner transformation using OpenFermion."""

    _mapping_method = OpenFermionJordanWigner


jordan_wigner = OpenFermionJordanWignerFactory()


class OpenFermionBravyiKitaevFactory(
    BravyiKitaevMapperFactory, OpenFermionQubitMapperFactory
):
    """Bravyi-Kitaev transformation using OpenFermion."""

    _mapping_method = OpenFermionBravyiKitaev


bravyi_kitaev = OpenFermionBravyiKitaevFactory()


class OpenFermionSymmetryConservingBravyiKitaevFactory(
    SymmetryConservingBravyiKitaevMapperFactory, OpenFermionQubitMapperFactory
):
    """Symmetry-conserving Bravyi-Kitaev transformation described in
    arXiv:1701.08213, using OpenFermion.

    Note that in this mapping the spin orbital indices are first
    reordered to all spin-up orbitals, then all spin-down orbitals.
    Bravyi-Kitaev transoformation is applied after the reordering and
    then two qubits are dropped using conservation of particle number
    and spin.

    Any operators which don't have particle number and spin symmetry are
    converted to `Operator()`, whose expectation value is zero for all
    states.
    """

    _mapping_method = OpenFermionSymmetryConservingBravyiKitaev


symmetry_conserving_bravyi_kitaev = OpenFermionSymmetryConservingBravyiKitaevFactory()


__all__ = ["OpenFermionQubitMapping", "OpenFermionQubitOperatorMapper"]
