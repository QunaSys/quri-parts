import unittest
from typing import cast

from numpy import allclose
from pyscf import gto, scf

from quri_parts.chem.mol import (
    ActiveSpaceMolecularOrbitals,
    AOeIntArraySet,
    SpatialMOeIntSet,
    SpinMOeIntSet,
    cas,
)
from quri_parts.pyscf.mol import PySCFAOeIntSet, PySCFMolecularOrbitals, get_ao_eint_set


def compute_integrals(
    molecule: PySCFMolecularOrbitals, active_space_mo: ActiveSpaceMolecularOrbitals
) -> tuple[
    AOeIntArraySet,
    PySCFAOeIntSet,
    SpinMOeIntSet,
    SpinMOeIntSet,
    SpatialMOeIntSet,
    SpatialMOeIntSet,
]:
    chem_ao_eint_set = get_ao_eint_set(molecule, store_array_on_memory=True)
    pyscf_ao_eint_set = get_ao_eint_set(molecule)
    pyscf_active_space_integrals = pyscf_ao_eint_set.to_active_space_mo_int(
        active_space_mo
    )
    active_space_integrals = chem_ao_eint_set.to_active_space_mo_int(active_space_mo)
    pyscf_active_space_spatial_integrals = (
        pyscf_ao_eint_set.to_active_space_spatial_mo_int(active_space_mo)
    )
    active_space_spatial_integrals = chem_ao_eint_set.to_active_space_spatial_mo_int(
        active_space_mo
    )
    return (
        cast(AOeIntArraySet, chem_ao_eint_set),
        cast(PySCFAOeIntSet, pyscf_ao_eint_set),
        pyscf_active_space_integrals,
        active_space_integrals,
        pyscf_active_space_spatial_integrals,
        active_space_spatial_integrals,
    )


class ElectronIntegrals:
    """Compare output of AOeIntArraySet and PySCFAOeIntSet."""

    molecule: PySCFMolecularOrbitals
    active_space_mo: ActiveSpaceMolecularOrbitals
    chem_ao_eint_set: AOeIntArraySet
    pyscf_ao_eint_set: PySCFAOeIntSet
    pyscf_active_space_integrals: SpinMOeIntSet
    active_space_integrals: SpinMOeIntSet
    pyscf_active_space_spatial_integrals: SpatialMOeIntSet
    active_space_spatial_integrals: SpatialMOeIntSet

    def test_compare_ao1eint(self) -> None:
        chem_ao1eint = self.chem_ao_eint_set.ao_1e_int
        pyscf_ao1eint = self.pyscf_ao_eint_set.ao_1e_int
        assert allclose(chem_ao1eint.array, pyscf_ao1eint.array)

    def test_compare_ao2eint(self) -> None:
        chem_ao2eint = self.chem_ao_eint_set.ao_2e_int
        pyscf_ao2eint = self.pyscf_ao_eint_set.ao_2e_int
        assert allclose(chem_ao2eint.array, pyscf_ao2eint.array)

    def test_compare_spin_mo1eint(self) -> None:
        chem_spin_mo1eint = self.chem_ao_eint_set.ao_1e_int.to_mo1int(
            self.molecule.mo_coeff
        )
        pyscf_spin_mo1eint = self.pyscf_ao_eint_set.ao_1e_int.to_mo1int(
            self.molecule.mo_coeff
        )
        assert allclose(chem_spin_mo1eint.array, pyscf_spin_mo1eint.array)

    def test_compare_spin_mo2eint(self) -> None:
        chem_spin_mo2eint = self.chem_ao_eint_set.ao_2e_int.to_mo2int(
            self.molecule.mo_coeff
        )
        pyscf_spin_mo2eint = self.pyscf_ao_eint_set.ao_2e_int.to_mo2int(
            self.molecule.mo_coeff
        )
        assert allclose(chem_spin_mo2eint.array, pyscf_spin_mo2eint.array)

    def test_compare_spatial_mo1eint(self) -> None:
        chem_spatial_mo1eint = self.chem_ao_eint_set.ao_1e_int.to_spatial_mo1int(
            self.molecule.mo_coeff
        )
        pyscf_spatial_mo1eint = self.pyscf_ao_eint_set.ao_1e_int.to_spatial_mo1int(
            self.molecule.mo_coeff
        )
        assert allclose(chem_spatial_mo1eint.array, pyscf_spatial_mo1eint.array)

    def test_compare_spatial_mo2eint(self) -> None:
        chem_spatial_mo2eint = self.chem_ao_eint_set.ao_2e_int.to_spatial_mo2int(
            self.molecule.mo_coeff
        )
        pyscf_spatial_mo2eint = self.pyscf_ao_eint_set.ao_2e_int.to_spatial_mo2int(
            self.molecule.mo_coeff
        )
        assert allclose(chem_spatial_mo2eint.array, pyscf_spatial_mo2eint.array)

    def test_compare_casci_const(self) -> None:
        assert allclose(
            self.pyscf_active_space_integrals.const, self.active_space_integrals.const
        )

    def test_compare_casci_spin_mo_1eint(self) -> None:
        assert allclose(
            self.pyscf_active_space_integrals.mo_1e_int.array,
            self.active_space_integrals.mo_1e_int.array,
        )

    def test_compare_casci_spin_mo_2eint(self) -> None:
        assert allclose(
            self.pyscf_active_space_integrals.mo_2e_int.array,
            self.active_space_integrals.mo_2e_int.array,
        )

    def test_compare_casci_const_from_spatial(self) -> None:
        assert allclose(
            self.pyscf_active_space_spatial_integrals.const,
            self.active_space_spatial_integrals.const,
        )

    def test_compare_casci_spatial_mo_1eint(self) -> None:
        assert allclose(
            self.pyscf_active_space_spatial_integrals.mo_1e_int.array,
            self.active_space_spatial_integrals.mo_1e_int.array,
        )

    def test_compare_casci_spatial_mo_2eint(self) -> None:
        assert allclose(
            self.pyscf_active_space_spatial_integrals.mo_2e_int.array,
            self.active_space_spatial_integrals.mo_2e_int.array,
        )


class TestH2O(ElectronIntegrals, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        atom_list = [["O", [0, 0, 0]], ["H", [0, 0, 1]], ["H", [2, 0, 0.5]]]
        mol = gto.M(atom=atom_list)
        mf = scf.RHF(mol)
        mf.kernel()

        cls.molecule = PySCFMolecularOrbitals(mol=mol, mo_coeff=mf.mo_coeff)
        cls.active_space_mo = ActiveSpaceMolecularOrbitals(
            cls.molecule, cas(n_active_ele=6, n_active_orb=4)
        )
        (
            cls.chem_ao_eint_set,
            cls.pyscf_ao_eint_set,
            cls.pyscf_active_space_integrals,
            cls.active_space_integrals,
            cls.pyscf_active_space_spatial_integrals,
            cls.active_space_spatial_integrals,
        ) = compute_integrals(
            molecule=cls.molecule, active_space_mo=cls.active_space_mo
        )


class TestSpinningH2O(ElectronIntegrals, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        atom_list = [["O", [0, 0, 0]], ["H", [0, 0, 1]], ["H", [2, 0, 0.5]]]
        mol = gto.M(atom=atom_list, spin=2)
        mf = scf.ROHF(mol)
        mf.kernel()

        cls.molecule = PySCFMolecularOrbitals(mol=mol, mo_coeff=mf.mo_coeff)
        cls.active_space_mo = ActiveSpaceMolecularOrbitals(
            cls.molecule, cas(n_active_ele=6, n_active_orb=4)
        )
        (
            cls.chem_ao_eint_set,
            cls.pyscf_ao_eint_set,
            cls.pyscf_active_space_integrals,
            cls.active_space_integrals,
            cls.pyscf_active_space_spatial_integrals,
            cls.active_space_spatial_integrals,
        ) = compute_integrals(
            molecule=cls.molecule, active_space_mo=cls.active_space_mo
        )


class TestCCPVDZH2O(ElectronIntegrals, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        atom_list = [["O", [0, 0, 0]], ["H", [0, 0, 1]], ["H", [2, 0, 0.5]]]
        mol = gto.M(atom=atom_list, basis="ccpvdz")
        mf = scf.RHF(mol)
        mf.kernel()

        cls.molecule = PySCFMolecularOrbitals(mol=mol, mo_coeff=mf.mo_coeff)
        cls.active_space_mo = ActiveSpaceMolecularOrbitals(
            cls.molecule, cas(n_active_ele=6, n_active_orb=4)
        )
        (
            cls.chem_ao_eint_set,
            cls.pyscf_ao_eint_set,
            cls.pyscf_active_space_integrals,
            cls.active_space_integrals,
            cls.pyscf_active_space_spatial_integrals,
            cls.active_space_spatial_integrals,
        ) = compute_integrals(
            molecule=cls.molecule, active_space_mo=cls.active_space_mo
        )


class TestChargedH2O(ElectronIntegrals, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        atom_list = [["O", [0, 0, 0]], ["H", [0, 0, 1]], ["H", [2, 0, 0.5]]]
        mol = gto.M(atom=atom_list, spin=1, charge=1)
        mf = scf.ROHF(mol)
        mf.kernel()

        cls.molecule = PySCFMolecularOrbitals(mol=mol, mo_coeff=mf.mo_coeff)
        cls.active_space_mo = ActiveSpaceMolecularOrbitals(
            cls.molecule, cas(n_active_ele=5, n_active_orb=4)
        )
        (
            cls.chem_ao_eint_set,
            cls.pyscf_ao_eint_set,
            cls.pyscf_active_space_integrals,
            cls.active_space_integrals,
            cls.pyscf_active_space_spatial_integrals,
            cls.active_space_spatial_integrals,
        ) = compute_integrals(
            molecule=cls.molecule, active_space_mo=cls.active_space_mo
        )
