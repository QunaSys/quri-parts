# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

from openfermion import FermionOperator, InteractionOperator, MajoranaOperator

from quri_parts.chem.mol import (
    ActiveSpaceMolecularOrbitals,
    MolecularOrbitals,
    SpinMOeIntSet,
)
from quri_parts.chem.transforms import FermionQubitStateMapper
from quri_parts.core.operator import Operator
from quri_parts.openfermion.transforms import (
    OpenFermionQubitMapping,
    OpenFermionQubitOperatorMapper,
    jordan_wigner,
)


def get_fermionic_hamiltonian(
    spin_mo_eint_set: SpinMOeIntSet,
) -> InteractionOperator:
    """Construct the molecular hamiltonian from the spin MO electron
    integrals."""
    nuc_energy = spin_mo_eint_set.const
    mo_1e_int = spin_mo_eint_set.mo_1e_int.array
    mo_2e_int = spin_mo_eint_set.mo_2e_int.array / 2
    return InteractionOperator(nuc_energy, mo_1e_int, mo_2e_int)


def convert_fermionic_hamiltonian_to_qubit_hamiltonian(
    fermionic_hamiltonian: Union[
        FermionOperator, InteractionOperator, MajoranaOperator
    ],
    mo: MolecularOrbitals,
    fermion_qubit_mapping: OpenFermionQubitMapping = jordan_wigner,
) -> tuple[Operator, OpenFermionQubitOperatorMapper, FermionQubitStateMapper]:
    """Converts the fermionic hamiltonian into qubit hamiltonian with a given
    mapping method, and returns the operator mapper along with the state
    mapper."""
    if isinstance(mo, ActiveSpaceMolecularOrbitals):
        n_spin_orbitals = 2 * mo.n_active_orb
        n_electrons = mo.n_active_ele
    else:
        n_spin_orbitals = 2 * mo.n_spatial_orb
        n_electrons = mo.n_electron
    operator_mapper = fermion_qubit_mapping.get_of_operator_mapper(
        n_spin_orbitals, n_electrons
    )
    state_mapper = fermion_qubit_mapping.get_state_mapper(n_spin_orbitals, n_electrons)
    return operator_mapper(fermionic_hamiltonian), operator_mapper, state_mapper


def get_qubit_mapped_hamiltonian_operator(
    mo: MolecularOrbitals,
    spin_mo_eint_set: SpinMOeIntSet,
    fermion_qubit_mapping: OpenFermionQubitMapping = jordan_wigner,
) -> tuple[Operator, OpenFermionQubitOperatorMapper, FermionQubitStateMapper]:
    """Computes the qubit hamiltonian and returns the operator mapper along
    with the state mapper."""
    fermionic_hamiltonian = get_fermionic_hamiltonian(spin_mo_eint_set)
    return convert_fermionic_hamiltonian_to_qubit_hamiltonian(
        fermionic_hamiltonian, mo, fermion_qubit_mapping
    )
