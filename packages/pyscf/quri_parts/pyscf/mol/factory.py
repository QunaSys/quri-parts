from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, cast

from pyscf import ao2mo, mcscf

from quri_parts.chem.mol import (
    ActiveSpace,
    ActiveSpaceMolecularOrbitals,
    AO1eIntProtocol,
    AO2eIntProtocol,
    AOeIntSet,
    MO1eIntArray,
    MO2eIntArray,
    MOeIntSet,
    get_active_space_integrals_from_mo,
    to_spin_orbital,
)

from .non_relativistic import ao1int, ao2int, pyscf_ao1int, pyscf_ao2int
from .pyscf_interface import PySCFMolecularOrbitals, get_nuc_energy


class MolecularHamiltonianBase(ABC):
    """Abstract base class for auto computation of molecular hamiltonians."""

    def __init__(
        self,
        molecule: PySCFMolecularOrbitals,
        ao1_int_computer: Callable[[PySCFMolecularOrbitals], AO1eIntProtocol],
        ao2_int_computer: Callable[[PySCFMolecularOrbitals], AO2eIntProtocol],
    ) -> None:
        """
        Args:
            molecule:
                A PySCFMolecularOrbitals representing the molecule of interest.
            ao1_int_computer:
                A function specifying how to compute the 1-electron integrals.
            ao2_int_computer:
                A function specifying how to compute the 2-electron integrals.
            ao_eint_set_type:
                A subclass of AOeIntSetBase for storing the ao integrals.
        """
        self.mol = molecule
        self.ao1_int_computer, self.ao2_int_computer = (
            ao1_int_computer,
            ao2_int_computer,
        )
        self.ao_e_int_set, self.mo_e_int_set = self.get_ao_mo_e_int_set()

    def get_ao_mo_e_int_set(self) -> tuple[AOeIntSet, MOeIntSet]:
        """Computes core energy, ao and mo electron integrals upon
        initialization of the class."""
        nuc_energy = get_nuc_energy(self.mol)
        ao_1e_int = self.ao1_int_computer(self.mol)
        ao_2e_int = self.ao2_int_computer(self.mol)

        mo_1e_int = ao_1e_int.to_mo1int(self.mol.mo_coeff)
        mo_2e_int = ao_2e_int.to_mo2int(self.mol.mo_coeff)

        ao_e_int_set = AOeIntSet(
            constant=nuc_energy,
            ao_1e_int=ao_1e_int,
            ao_2e_int=ao_2e_int,
        )

        mo_e_int_set = MOeIntSet(
            const=nuc_energy, mo_1e_int=mo_1e_int, mo_2e_int=mo_2e_int
        )

        return ao_e_int_set, mo_e_int_set

    @abstractmethod
    def get_active_space_molecular_integrals(
        self,
        n_active_ele: Optional[int] = None,
        n_active_orb: Optional[int] = None,
        active_orbs_indices: Optional[Sequence[int]] = None,
    ) -> MOeIntSet:
        """An abstract method determining how to compute the active space spin
        orbital molecular integrals."""
        ...

    def get_molecular_hamiltonian(
        self,
        n_active_ele: Optional[int] = None,
        n_active_orb: Optional[int] = None,
        active_orbs_indices: Optional[Sequence[int]] = None,
    ) -> MOeIntSet:
        """Computes the spin orbital electron integrals for generating the
        molecular hamiltonian."""
        if n_active_ele and n_active_orb:
            hamiltonian_component = self.get_active_space_molecular_integrals(
                n_active_ele, n_active_orb, active_orbs_indices
            )
            return cast(MOeIntSet, hamiltonian_component)

        nuc_energy, mo_1e_int, mo_2e_int = (
            self.mo_e_int_set.const,
            self.mo_e_int_set.mo_1e_int,
            self.mo_e_int_set.mo_2e_int,
        )
        n_qubit = mo_1e_int.array.shape[0] * 2
        spin_1e_integrals, spin_2e_integrals = to_spin_orbital(
            n_qubit, mo_1e_int.array, mo_2e_int.array
        )

        active_space_1e_int = MO1eIntArray(spin_1e_integrals)
        active_space_2e_int = MO2eIntArray(spin_2e_integrals)

        hamiltonian_component = MOeIntSet(
            nuc_energy, active_space_1e_int, active_space_2e_int
        )

        return hamiltonian_component


class MolecularHamiltonian(MolecularHamiltonianBase):
    def __init__(self, molecule: PySCFMolecularOrbitals) -> None:
        ao1_int_computer = ao1int
        ao2_int_computer = ao2int
        super().__init__(molecule, ao1_int_computer, ao2_int_computer)

    def get_active_space_molecular_orbitals(
        self,
        n_active_ele: Optional[int] = None,
        n_active_orb: Optional[int] = None,
        active_orbs_indices: Optional[Sequence[int]] = None,
    ) -> ActiveSpaceMolecularOrbitals:
        """Construct the active space configuration."""
        if n_active_ele is None or n_active_orb is None:
            n_active_ele = self.mol.n_electron
            n_active_orb = self.mol.mol.nao

        active_space = ActiveSpace(
            n_active_ele=n_active_ele,
            n_active_orb=cast(int, n_active_orb),
            active_orbs_indices=active_orbs_indices,
        )
        active_orbitals = ActiveSpaceMolecularOrbitals(self.mol, active_space)

        self.check(active_orbitals)

        return active_orbitals

    @staticmethod
    def check(active_orbitals: ActiveSpaceMolecularOrbitals) -> None:
        """Consistency check of the active space configuration."""
        assert active_orbitals.n_vir_orb >= 0, ValueError(
            f"Number of virtual orbitals should be a positive integer or zero.\n"
            f" n_vir = {active_orbitals.n_vir_orb}"
        )
        assert active_orbitals.n_core_ele >= 0, ValueError(
            f"Number of core electrons should be a positive integer or zero.\n"
            f" n_vir = {active_orbitals.n_core_ele}"
        )
        assert active_orbitals.n_ele_alpha >= 0, ValueError(
            f"Number of spin up electrons should be a positive integer or zero.\n"
            f" n_ele_alpha = {active_orbitals.n_ele_alpha}"
        )
        assert active_orbitals.n_ele_beta >= 0, ValueError(
            f"Number of spin down electrons should be a positive integer or zero.\n"
            f" n_ele_beta = {active_orbitals.n_ele_beta}"
        )
        assert active_orbitals.n_ele_alpha <= active_orbitals.n_active_orb, ValueError(
            f"Number of spin up electrons should not exceed the number of active orbitals.\n"  # noqa: E501
            f" n_ele_alpha = {active_orbitals.n_ele_alpha},\n"
            f" n_active_orb = {active_orbitals.n_active_orb}"
        )
        assert active_orbitals.n_ele_beta <= active_orbitals.n_active_orb, ValueError(
            f"Number of spin down electrons should not exceed the number of active orbitals.\n"  # noqa: E501
            f" n_ele_beta = {active_orbitals.n_ele_beta},\n"
            f" n_active_orb = {active_orbitals.n_active_orb}"
        )

    def get_active_space_molecular_integrals(
        self,
        n_active_ele: Optional[int] = None,
        n_active_orb: Optional[int] = None,
        active_orbs_indices: Optional[Sequence[int]] = None,
    ) -> MOeIntSet:
        """Computes the active space spin orbital electron integrals."""
        active_orbitals = self.get_active_space_molecular_orbitals(
            n_active_ele, n_active_orb, active_orbs_indices
        )
        hamiltonian_component = get_active_space_integrals_from_mo(
            active_orbitals, self.mo_e_int_set
        )
        return hamiltonian_component


class PySCFMolecularHamiltonian(MolecularHamiltonianBase):
    def __init__(self, molecule: PySCFMolecularOrbitals) -> None:
        ao1_int_computer = pyscf_ao1int
        ao2_int_computer = pyscf_ao2int
        super().__init__(molecule, ao1_int_computer, ao2_int_computer)

    def get_active_space_molecular_integrals(
        self,
        n_active_ele: Optional[int] = None,
        n_active_orb: Optional[int] = None,
        active_orbs_indices: Optional[Sequence[int]] = None,
    ) -> MOeIntSet:
        """Computes the active space spin orbital electron integrals with
        PySCF."""
        cas_mf = mcscf.CASCI(self.mol.mol, n_active_orb, n_active_ele)
        if active_orbs_indices:
            mo = cas_mf.sort_mo(active_orbs_indices, mo_coeff=self.mol.mo_coeff, base=0)
        else:
            mo = self.mol.mo_coeff

        casscf_mo_1e_int, casscf_nuc = cas_mf.get_h1eff(mo)
        casscf_mo_2e_int = cas_mf.get_h2eff(mo)
        casscf_mo_2e_int = ao2mo.restore(1, casscf_mo_2e_int, cas_mf.ncas).transpose(
            0, 2, 3, 1
        )

        casscf_mo_1e_spin_int, casscf_mo_2e_spin_int = to_spin_orbital(
            2 * casscf_mo_1e_int.shape[0], casscf_mo_1e_int, casscf_mo_2e_int
        )

        hamiltonian_component = MOeIntSet(
            const=casscf_nuc,
            mo_1e_int=MO1eIntArray(casscf_mo_1e_spin_int),
            mo_2e_int=MO2eIntArray(casscf_mo_2e_spin_int),
        )
        return hamiltonian_component
