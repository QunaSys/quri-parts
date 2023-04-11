from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Union, cast

import pyscf
from pyscf import ao2mo, mcscf

from quri_parts.chem.mol import (
    ActiveSpace,
    ActiveSpaceMolecularOrbitals,
    AO1eIntArray,
    AO2eIntArray,
    AOeIntSet,
    MO1eIntArray,
    MO2eIntArray,
    MOeIntSet,
    check_active_space_consitency,
    get_active_space_integrals_from_mo,
    spatial_mo_eint_set_to_spin_mo_eint_set,
)

from .non_relativistic import PySCFAO1eInt, PySCFAO2eInt, ao1int, ao2int
from .pyscf_interface import PySCFMolecularOrbitals, get_nuc_energy


@dataclass
class PySCFAOeIntSet:
    mol: pyscf.gto.Mole
    constant: float
    ao_1e_int: PySCFAO1eInt
    ao_2e_int: PySCFAO2eInt

    def to_active_space_mo_int(
        self, active_space_mo: ActiveSpaceMolecularOrbitals
    ) -> MOeIntSet:
        return pyscf_get_active_space_integrals(active_space_mo, self)


def pyscf_get_active_space_integrals(
    active_space_mo: ActiveSpaceMolecularOrbitals, electron_ints: PySCFAOeIntSet
) -> MOeIntSet:
    n_active_orb, n_active_ele = (
        active_space_mo.n_active_orb,
        active_space_mo.n_active_ele,
    )
    cas_mf = mcscf.CASCI(electron_ints.mol, n_active_orb, n_active_ele)
    _, active_orb_list = active_space_mo.get_core_and_active_orb()
    if active_orb_list:
        mo = cas_mf.sort_mo(active_orb_list, mo_coeff=active_space_mo.mo_coeff, base=0)
    else:
        mo = active_space_mo.mo_coeff

    casci_mo_1e_int, casscf_nuc = cas_mf.get_h1eff(mo)
    casci_mo_2e_int = cas_mf.get_h2eff(mo)
    casci_mo_2e_int = ao2mo.restore(1, casci_mo_2e_int, cas_mf.ncas).transpose(
        0, 2, 3, 1
    )

    spatial_integrals = MOeIntSet(
        const=casscf_nuc,
        mo_1e_int=MO1eIntArray(casci_mo_1e_int),
        mo_2e_int=MO2eIntArray(casci_mo_2e_int),
    )

    spin_integrals = spatial_mo_eint_set_to_spin_mo_eint_set(spatial_integrals)

    return spin_integrals


def get_ao_eint_set(molecule: PySCFMolecularOrbitals) -> AOeIntSet:
    nuc_energy = get_nuc_energy(molecule)
    ao_1e_int = AO1eIntArray(ao1int(molecule).array)
    ao_2e_int = AO2eIntArray(ao2int(molecule).array)
    ao_e_int_set = AOeIntSet(
        constant=nuc_energy,
        ao_1e_int=ao_1e_int,
        ao_2e_int=ao_2e_int,
    )
    return ao_e_int_set


def pyscf_get_ao_eint_set(molecule: PySCFMolecularOrbitals) -> PySCFAOeIntSet:
    nuc_energy = get_nuc_energy(molecule)
    ao_1e_int = ao1int(molecule)
    ao_2e_int = ao2int(molecule)
    ao_e_int_set = PySCFAOeIntSet(
        mol=molecule.mol,
        constant=nuc_energy,
        ao_1e_int=ao_1e_int,
        ao_2e_int=ao_2e_int,
    )
    return ao_e_int_set


def get_mo_eint_set_from_ao_eint_set(
    molecule: PySCFMolecularOrbitals, ao_eint_set: Union[AOeIntSet, PySCFAOeIntSet]
) -> MOeIntSet:
    mo_1e_int = ao_eint_set.ao_1e_int.to_mo1int(molecule.mo_coeff)
    mo_2e_int = ao_eint_set.ao_2e_int.to_mo2int(molecule.mo_coeff)

    mo_eint_set = MOeIntSet(
        const=ao_eint_set.constant,
        mo_1e_int=mo_1e_int,
        mo_2e_int=mo_2e_int,
    )
    return mo_eint_set


def get_active_space_molecular_orbitals(
    molecule: pyscf.gto.Mole,
    n_active_ele: int,
    n_active_orb: int,
    active_orbs_indices: Optional[Sequence[int]] = None,
) -> ActiveSpaceMolecularOrbitals:
    active_space = ActiveSpace(n_active_ele, n_active_orb, active_orbs_indices)
    active_space_mo = ActiveSpaceMolecularOrbitals(
        mo=molecule, active_space=active_space
    )
    check_active_space_consitency(active_space_mo)
    return active_space_mo


class MolecularHamiltonianProtocol(Protocol):
    def get_active_space_molecular_integrals(
        self, n_active_ele: int, n_active_orb: int, active_orbs_indices: Sequence[int]
    ) -> MOeIntSet:
        ...

    def get_full_space_molecular_integrals(self) -> MOeIntSet:
        ...


class MolecularHamiltonian(MolecularHamiltonianProtocol):
    def __init__(self, molecule: PySCFMolecularOrbitals) -> None:
        self._mol = molecule
        self.ao_eint_set = get_ao_eint_set(molecule=molecule)
        self.mo_eint_set = get_mo_eint_set_from_ao_eint_set(
            molecule=molecule, ao_eint_set=self.ao_eint_set
        )

    def get_active_space_molecular_orbitals(
        self,
        n_active_ele: int,
        n_active_orb: int,
        active_orbs_indices: Optional[Sequence[int]] = None,
    ) -> ActiveSpaceMolecularOrbitals:
        # This might be redundant
        return get_active_space_molecular_orbitals(
            self._mol, n_active_ele, n_active_orb, active_orbs_indices
        )

    def get_active_space_molecular_integrals(
        self,
        n_active_ele: int,
        n_active_orb: int,
        active_orbs_indices: Optional[Sequence[int]] = None,
    ) -> MOeIntSet:
        active_space_mo = self.get_active_space_molecular_orbitals(
            n_active_ele,
            n_active_orb,
            active_orbs_indices,
        )
        active_space_integrals = get_active_space_integrals_from_mo(
            active_space_mo=active_space_mo, electron_mo_ints=self.mo_eint_set
        )
        return cast(MOeIntSet, active_space_integrals)

    def get_full_space_molecular_integrals(self) -> MOeIntSet:
        full_space_integrals = spatial_mo_eint_set_to_spin_mo_eint_set(self.mo_eint_set)
        return full_space_integrals


class PySCFMolecularHamiltonian(MolecularHamiltonianProtocol):
    def __init__(self, molecule: PySCFMolecularOrbitals) -> None:
        self._mol = molecule
        self.ao_eint_set = pyscf_get_ao_eint_set(molecule=molecule)
        self.mo_eint_set = get_mo_eint_set_from_ao_eint_set(
            molecule=molecule, ao_eint_set=self.ao_eint_set
        )

    def get_active_space_molecular_orbitals(
        self,
        n_active_ele: int,
        n_active_orb: int,
        active_orbs_indices: Optional[Sequence[int]] = None,
    ) -> ActiveSpaceMolecularOrbitals:
        # This might be redundant
        return get_active_space_molecular_orbitals(
            self._mol, n_active_ele, n_active_orb, active_orbs_indices
        )

    def get_active_space_molecular_integrals(
        self,
        n_active_ele: int,
        n_active_orb: int,
        active_orbs_indices: Optional[Sequence[int]] = None,
    ) -> MOeIntSet:
        active_space_mo = self.get_active_space_molecular_orbitals(
            n_active_ele,
            n_active_orb,
            active_orbs_indices,
        )
        active_space_integrals = pyscf_get_active_space_integrals(
            active_space_mo=active_space_mo,
            electron_ints=self.ao_eint_set,
        )
        return active_space_integrals

    def get_full_space_molecular_integrals(self) -> MOeIntSet:
        full_space_integrals = spatial_mo_eint_set_to_spin_mo_eint_set(self.mo_eint_set)
        return full_space_integrals


# from abc import ABC, abstractmethod
# from typing import Callable, NamedTuple, Optional, Protocol, Sequence, Union, cast
# class MolecularHamiltonianBase(ABC):
#     """Abstract base class for auto computation of molecular hamiltonians."""

#     def __init__(
#         self,
#         molecule: PySCFMolecularOrbitals,
#         ao1_int_computer: Callable[[PySCFMolecularOrbitals], AO1eInt],
#         ao2_int_computer: Callable[[PySCFMolecularOrbitals], AO2eInt],
#     ) -> None:
#         """
#         Args:
#             molecule:
#                 A PySCFMolecularOrbitals representing the molecule of interest.
#             ao1_int_computer:
#                 A function specifying how to compute the 1-electron integrals.
#             ao2_int_computer:
#                 A function specifying how to compute the 2-electron integrals.
#             ao_eint_set_type:
#                 A subclass of AOeIntSetBase for storing the ao integrals.
#         """
#         self.mol = molecule
#         self.ao1_int_computer, self.ao2_int_computer = (
#             ao1_int_computer,
#             ao2_int_computer,
#         )
#         self.ao_e_int_set, self.mo_e_int_set = self.get_ao_mo_e_int_set()

#     def get_ao_mo_e_int_set(self) -> tuple[AOeIntSet, MOeIntSet]:
#         """Computes core energy, ao and mo electron integrals upon
#         initialization of the class."""
#         nuc_energy = get_nuc_energy(self.mol)
#         ao_1e_int = self.ao1_int_computer(self.mol)
#         ao_2e_int = self.ao2_int_computer(self.mol)

#         mo_1e_int = ao_1e_int.to_mo1int(self.mol.mo_coeff)
#         mo_2e_int = ao_2e_int.to_mo2int(self.mol.mo_coeff)

#         ao_e_int_set = AOeIntSet(
#             constant=nuc_energy,
#             ao_1e_int=ao_1e_int,
#             ao_2e_int=ao_2e_int,
#         )

#         mo_e_int_set = MOeIntSet(
#             const=nuc_energy, mo_1e_int=mo_1e_int, mo_2e_int=mo_2e_int
#         )

#         return ao_e_int_set, mo_e_int_set

#     @abstractmethod
#     def get_active_space_molecular_integrals(
#         self,
#         n_active_ele: Optional[int] = None,
#         n_active_orb: Optional[int] = None,
#         active_orbs_indices: Optional[Sequence[int]] = None,
#     ) -> MOeIntSet:
#         """An abstract method determining how to compute the active space spin
#         orbital molecular integrals."""
#         ...

#     def get_molecular_hamiltonian(
#         self,
#         n_active_ele: Optional[int] = None,
#         n_active_orb: Optional[int] = None,
#         active_orbs_indices: Optional[Sequence[int]] = None,
#     ) -> MOeIntSet:
#         """Computes the spin orbital electron integrals for generating the
#         molecular hamiltonian."""
#         if n_active_ele and n_active_orb:
#             hamiltonian_component = self.get_active_space_molecular_integrals(
#                 n_active_ele, n_active_orb, active_orbs_indices
#             )
#             return cast(MOeIntSet, hamiltonian_component)

#         nuc_energy, mo_1e_int, mo_2e_int = (
#             self.mo_e_int_set.const,
#             self.mo_e_int_set.mo_1e_int,
#             self.mo_e_int_set.mo_2e_int,
#         )
#         n_qubit = mo_1e_int.array.shape[0] * 2
#         spin_1e_integrals, spin_2e_integrals = to_spin_orbital(
#             n_qubit, mo_1e_int.array, mo_2e_int.array
#         )

#         active_space_1e_int = MO1eIntArray(spin_1e_integrals)
#         active_space_2e_int = MO2eIntArray(spin_2e_integrals)

#         hamiltonian_component = MOeIntSet(
#             nuc_energy, active_space_1e_int, active_space_2e_int
#         )

#         return hamiltonian_component


# class MolecularHamiltonian(MolecularHamiltonianBase):
#     def __init__(self, molecule: PySCFMolecularOrbitals) -> None:
#         ao1_int_computer = ao1int
#         ao2_int_computer = ao2int
#         super().__init__(molecule, ao1_int_computer, ao2_int_computer)

#     def get_active_space_molecular_orbitals(
#         self,
#         n_active_ele: Optional[int] = None,
#         n_active_orb: Optional[int] = None,
#         active_orbs_indices: Optional[Sequence[int]] = None,
#     ) -> ActiveSpaceMolecularOrbitals:
#         """Construct the active space configuration."""
#         if n_active_ele is None or n_active_orb is None:
#             n_active_ele = self.mol.n_electron
#             n_active_orb = self.mol.mol.nao

#         active_space = ActiveSpace(
#             n_active_ele=n_active_ele,
#             n_active_orb=cast(int, n_active_orb),
#             active_orbs_indices=active_orbs_indices,
#         )
#         active_orbitals = ActiveSpaceMolecularOrbitals(self.mol, active_space)

#         self.check(active_orbitals)

#         return active_orbitals

#     @staticmethod
#     def check(active_orbitals: ActiveSpaceMolecularOrbitals) -> None:
#         """Consistency check of the active space configuration."""
#         assert active_orbitals.n_vir_orb >= 0, ValueError(
#             f"Number of virtual orbitals should be a positive integer or zero.\n"
#             f" n_vir = {active_orbitals.n_vir_orb}"
#         )
#         assert active_orbitals.n_core_ele >= 0, ValueError(
#             f"Number of core electrons should be a positive integer or zero.\n"
#             f" n_vir = {active_orbitals.n_core_ele}"
#         )
#         assert active_orbitals.n_ele_alpha >= 0, ValueError(
#             f"Number of spin up electrons should be a positive integer or zero.\n"
#             f" n_ele_alpha = {active_orbitals.n_ele_alpha}"
#         )
#         assert active_orbitals.n_ele_beta >= 0, ValueError(
#             f"Number of spin down electrons should be a positive integer or zero.\n"
#             f" n_ele_beta = {active_orbitals.n_ele_beta}"
#         )
#         assert active_orbitals.n_ele_alpha <= active_orbitals.n_active_orb, ValueError(  # noqa: E501
#             f"Number of spin up electrons should not exceed the number of active orbitals.\n"  # noqa: E501
#             f" n_ele_alpha = {active_orbitals.n_ele_alpha},\n"
#             f" n_active_orb = {active_orbitals.n_active_orb}"
#         )
#         assert active_orbitals.n_ele_beta <= active_orbitals.n_active_orb, ValueError(
#             f"Number of spin down electrons should not exceed the number of active orbitals.\n"  # noqa: E501
#             f" n_ele_beta = {active_orbitals.n_ele_beta},\n"
#             f" n_active_orb = {active_orbitals.n_active_orb}"
#         )

#     def get_active_space_molecular_integrals(
#         self,
#         n_active_ele: Optional[int] = None,
#         n_active_orb: Optional[int] = None,
#         active_orbs_indices: Optional[Sequence[int]] = None,
#     ) -> MOeIntSet:
#         """Computes the active space spin orbital electron integrals."""
#         active_orbitals = self.get_active_space_molecular_orbitals(
#             n_active_ele, n_active_orb, active_orbs_indices
#         )
#         hamiltonian_component = get_active_space_integrals_from_mo(
#             active_orbitals, self.mo_e_int_set
#         )
#         return hamiltonian_component


# class PySCFMolecularHamiltonian(MolecularHamiltonianBase):
#     def __init__(self, molecule: PySCFMolecularOrbitals) -> None:
#         ao1_int_computer = pyscf_ao1int
#         ao2_int_computer = pyscf_ao2int
#         super().__init__(molecule, ao1_int_computer, ao2_int_computer)

#     def get_active_space_molecular_integrals(
#         self,
#         n_active_ele: Optional[int] = None,
#         n_active_orb: Optional[int] = None,
#         active_orbs_indices: Optional[Sequence[int]] = None,
#     ) -> MOeIntSet:
#         """Computes the active space spin orbital electron integrals with
#         PySCF."""
#         cas_mf = mcscf.CASCI(self.mol.mol, n_active_orb, n_active_ele)
#         if active_orbs_indices:
#             mo = cas_mf.sort_mo(active_orbs_indices, mo_coeff=self.mol.mo_coeff, base=0)  # noqa: E501
#         else:
#             mo = self.mol.mo_coeff

#         casscf_mo_1e_int, casscf_nuc = cas_mf.get_h1eff(mo)
#         casscf_mo_2e_int = cas_mf.get_h2eff(mo)
#         casscf_mo_2e_int = ao2mo.restore(1, casscf_mo_2e_int, cas_mf.ncas).transpose(
#             0, 2, 3, 1
#         )

#         casscf_mo_1e_spin_int, casscf_mo_2e_spin_int = to_spin_orbital(
#             2 * casscf_mo_1e_int.shape[0], casscf_mo_1e_int, casscf_mo_2e_int
#         )

#         hamiltonian_component = MOeIntSet(
#             const=casscf_nuc,
#             mo_1e_int=MO1eIntArray(casscf_mo_1e_spin_int),
#             mo_2e_int=MO2eIntArray(casscf_mo_2e_spin_int),
#         )
#         return hamiltonian_component
