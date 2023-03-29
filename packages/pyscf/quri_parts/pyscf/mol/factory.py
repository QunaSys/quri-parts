from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Type, Union, cast

import pyscf
from pyscf import ao2mo, scf

from quri_parts.chem.mol import (
    ActiveSpace,
    ActiveSpaceInfo,
    ActiveSpaceMolecularOrbitals,
    AO1eIntBase,
    AO2eIntBase,
    AOeIntSet,
    AOeIntSetBase,
    MO1eIntArray,
    MO2eIntArray,
    MOeIntSet,
    get_active_space_integrals,
    to_spin_orbital,
)

from .non_relativistic import ao1int, ao2int, pyscf_ao1int, pyscf_ao2int
from .pyscf_interface import PySCFMolecularOrbitals, get_nuc_energy

pyscf_constructor_dict = {
    "HF": scf.HF,
    "RHF": scf.RHF,
    "ROHF": scf.ROHF,
    "UHF": scf.UHF,
    "GHF": scf.GHF,
    "DHF": scf.DHF,
    "X2C": scf.X2C,
    "KS": scf.KS,
    "RKS": scf.RKS,
    "ROKS": scf.ROKS,
    "UKS": scf.UKS,
    "GKS": scf.GKS,
    "DKS": scf.DKS,
}


def PySCFMoleculeFactory(
    atom: str,
    spin: int = 0,
    charge: int = 0,
    basis: str = "sto-3g",
    symmetry: Union[bool, str] = False,
    verbose: int = 0,
    relativistic: bool = False,
    manual: Optional[bool] = False,
    construct_method: Optional[str] = None,
) -> PySCFMolecularOrbitals:
    """A factory for generating a PySCFMolecularOrbitals instance."""

    mol = pyscf.gto.Mole(
        atom=atom,
        basis=basis,
        charge=charge,
        spin=spin,
        symmetry=symmetry,
        verbose=verbose,
    )
    mol.build()

    if manual:
        assert (
            construct_method is not None
        ), "construct_method cannot be None when manually assigning how the molecule is built"  # noqa: E501
        mol_coeff = pyscf_constructor_dict[construct_method](mol).run().mo_coeff
        return PySCFMolecularOrbitals(mol=mol, mo_coeff=mol_coeff)

    if relativistic:
        mol_coeff = pyscf.scf.DHF(mol).run().mo_coeff
        return PySCFMolecularOrbitals(mol=mol, mo_coeff=mol_coeff)

    if spin:
        mol_coeff = pyscf.scf.ROHF(mol).run().mo_coeff
        return PySCFMolecularOrbitals(mol=mol, mo_coeff=mol_coeff)

    mol_coeff = pyscf.scf.RHF(mol).run().mo_coeff
    return PySCFMolecularOrbitals(mol=mol, mo_coeff=mol_coeff)


@dataclass(frozen=True)
class PySCFMoleculeFactoryInput:
    atom: list[list[Sequence[object]]]
    spin: int = 0
    charge: int = 0
    basis: str = "sto-3g"
    symmetry: Union[bool, str] = False
    verbose: int = 0
    relativistic: Optional[bool] = False
    manual: Optional[bool] = False
    construct_method: Optional[str] = None


class MolecularHamiltonianBase(ABC):
    def __init__(
        self,
        molecule: PySCFMoleculeFactoryInput,
        ao1_int_computer: Callable[[PySCFMolecularOrbitals], AO1eIntBase],
        ao2_int_computer: Callable[[PySCFMolecularOrbitals], AO2eIntBase],
        ao_eint_set_type: Type[AOeIntSetBase],
    ) -> None:
        self.mol = PySCFMoleculeFactory(**molecule.__dict__)
        self.ao1_int_computer, self.ao2_int_computer = (
            ao1_int_computer,
            ao2_int_computer,
        )
        self.ao_eint_set_type = ao_eint_set_type
        self.ao_e_int_set, self.mo_e_int_set = self.get_ao_mo_e_int_set()

    def get_ao_mo_e_int_set(self) -> tuple[AOeIntSetBase, MOeIntSet]:
        nuc_energy = get_nuc_energy(self.mol)
        ao_1e_int = self.ao1_int_computer(self.mol)
        ao_2e_int = self.ao2_int_computer(self.mol)

        mo_1e_int = ao_1e_int.to_mo1int(self.mol.mo_coeff)
        mo_2e_int = ao_2e_int.to_mo2int(self.mol.mo_coeff)

        ao_e_int_set = self.ao_eint_set_type(
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
        ...

    def get_molecular_hamiltonian(
        self,
        n_active_ele: Optional[int] = None,
        n_active_orb: Optional[int] = None,
        active_orbs_indices: Optional[Sequence[int]] = None,
    ) -> MOeIntSet:
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
        active_space_2e_int = MO2eIntArray(spin_2e_integrals / 2)

        hamiltonian_component = MOeIntSet(
            nuc_energy, active_space_1e_int, active_space_2e_int
        )

        return hamiltonian_component


class MolecularHamiltonian(MolecularHamiltonianBase):
    def __init__(self, molecule: PySCFMoleculeFactoryInput) -> None:
        ao1_int_computer = ao1int
        ao2_int_computer = ao2int
        ao_eint_set_type = AOeIntSet
        super().__init__(molecule, ao1_int_computer, ao2_int_computer, ao_eint_set_type)

    def get_active_space_molecular_orbitals(
        self,
        n_active_ele: Optional[int] = None,
        n_active_orb: Optional[int] = None,
        active_orbs_indices: Optional[Sequence[int]] = None,
    ) -> ActiveSpaceMolecularOrbitals:
        if n_active_ele is None or n_active_orb is None:
            n_active_ele = self.mol.n_electron
            n_active_orb = self.mol.mol.nao

        active_space = ActiveSpace(
            n_active_ele=n_active_ele,
            n_active_orb=cast(int, n_active_orb),
            active_orbs_indices=active_orbs_indices,
        )
        active_orbitals = ActiveSpaceMolecularOrbitals(self.mol, active_space)

        self.check(active_orbitals.info)

        return active_orbitals

    @staticmethod
    def check(active_orbitals: ActiveSpaceInfo) -> None:
        assert active_orbitals.n_vir_orb >= 0, ValueError(
            f"Number of virtual orbitals should be a positive integer.\n"
            f" n_vir = {active_orbitals.n_vir_orb}"
        )
        assert active_orbitals.n_ele_alpha >= 0, ValueError(
            f"Number of spin up electrons should be a positive integer.\n"
            f" n_ele_alpha = {active_orbitals.n_ele_alpha}"
        )
        assert active_orbitals.n_ele_beta >= 0, ValueError(
            f"Number of spin down electrons should be a positive integer.\n"
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
        active_orbitals = self.get_active_space_molecular_orbitals(
            n_active_ele, n_active_orb, active_orbs_indices
        )
        hamiltonian_component = get_active_space_integrals(
            active_orbitals, self.mo_e_int_set
        )
        return hamiltonian_component


class PySCFMolecularHamiltonian(MolecularHamiltonianBase):
    def __init__(self, molecule: PySCFMoleculeFactoryInput) -> None:
        ao1_int_computer = pyscf_ao1int
        ao2_int_computer = pyscf_ao2int
        ao_eint_set_type = AOeIntSet
        super().__init__(molecule, ao1_int_computer, ao2_int_computer, ao_eint_set_type)

    def get_active_space_molecular_integrals(
        self,
        n_active_ele: Optional[int] = None,
        n_active_orb: Optional[int] = None,
        active_orbs_indices: Optional[Sequence[int]] = None,
        fix_mo_coeff = True
    ) -> MOeIntSet:
        cas_mf = self.mol.mol.CASSCF(n_active_orb, n_active_ele)
        if fix_mo_coeff:
            cas_mf.frozen = self.mol.mol.nao
        cas_mf.kernel()

        casscf_mo_1e_int, casscf_nuc = cas_mf.get_h1eff()
        casscf_mo_2e_int = cas_mf.get_h2eff()
        casscf_mo_2e_int = ao2mo.restore(1, casscf_mo_2e_int, cas_mf.ncas).transpose(
            0, 2, 3, 1
        )

        casscf_mo_1e_spin_int, casscf_mo_2e_spin_int = to_spin_orbital(
            2 * casscf_mo_1e_int.shape[0], casscf_mo_1e_int, casscf_mo_2e_int
        )

        hamiltonian_component = MOeIntSet(
            const=casscf_nuc,
            mo_1e_int=MO1eIntArray(casscf_mo_1e_spin_int),
            mo_2e_int=MO1eIntArray(casscf_mo_2e_spin_int / 2),
        )
        return hamiltonian_component
    
    def get_molecular_hamiltonian(
            self, 
            n_active_ele: Optional[int] = None, 
            n_active_orb: Optional[int] = None, 
            active_orbs_indices: Optional[Sequence[int]] = None,
            fix_mo_coeff = True
        ) -> MOeIntSet:
        if not fix_mo_coeff and (n_active_ele and n_active_orb):
            hamiltonian_component = self.get_active_space_molecular_integrals(
                n_active_ele, n_active_orb, active_orbs_indices, fix_mo_coeff=False
            )
            return cast(MOeIntSet, hamiltonian_component)
        return super().get_molecular_hamiltonian(n_active_ele, n_active_orb, active_orbs_indices)
