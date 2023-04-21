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
    qp_chem_ao_eint_set = get_ao_eint_set(molecule, store_array_on_memory=True)
    qp_pyscf_ao_eint_set = get_ao_eint_set(molecule)
    pyscf_active_space_integrals = qp_pyscf_ao_eint_set.to_active_space_mo_int(
        active_space_mo
    )
    active_space_integrals = qp_chem_ao_eint_set.to_active_space_mo_int(active_space_mo)
    pyscf_active_space_spatial_integrals = (
        qp_pyscf_ao_eint_set.to_active_space_spatial_mo_int(active_space_mo)
    )
    active_space_spatial_integrals = qp_chem_ao_eint_set.to_active_space_spatial_mo_int(
        active_space_mo
    )
    return (
        cast(AOeIntArraySet, qp_chem_ao_eint_set),
        cast(PySCFAOeIntSet, qp_pyscf_ao_eint_set),
        pyscf_active_space_integrals,
        active_space_integrals,
        pyscf_active_space_spatial_integrals,
        active_space_spatial_integrals,
    )


class TestMolecule(unittest.TestCase):
    """Compare output of AOeIntArraySet and PySCFAOeIntSet."""

    molecule: PySCFMolecularOrbitals
    active_space_mo: ActiveSpaceMolecularOrbitals
    qp_chem_ao_eint_set: AOeIntArraySet
    qp_pyscf_ao_eint_set: PySCFAOeIntSet
    pyscf_active_space_integrals: SpinMOeIntSet
    active_space_integrals: SpinMOeIntSet
    pyscf_active_space_spatial_integrals: SpatialMOeIntSet
    active_space_spatial_integrals: SpatialMOeIntSet

    def compare_ao1eint(self) -> None:
        qp_chem_ao1eint = self.qp_chem_ao_eint_set.ao_1e_int
        qp_pyscf_ao1eint = self.qp_pyscf_ao_eint_set.ao_1e_int
        assert allclose(qp_chem_ao1eint.array, qp_pyscf_ao1eint.array)

    def compare_ao2eint(self) -> None:
        qp_chem_ao2eint = self.qp_chem_ao_eint_set.ao_2e_int
        qp_pyscf_ao2eint = self.qp_pyscf_ao_eint_set.ao_2e_int
        assert allclose(qp_chem_ao2eint.array, qp_pyscf_ao2eint.array)

    def compare_spin_mo1eint(self) -> None:
        qp_chem_spin_mo1eint = self.qp_chem_ao_eint_set.ao_1e_int.to_mo1int(
            self.molecule.mo_coeff
        )
        qp_pyscf_spin_mo1eint = self.qp_pyscf_ao_eint_set.ao_1e_int.to_mo1int(
            self.molecule.mo_coeff
        )
        assert allclose(qp_chem_spin_mo1eint.array, qp_pyscf_spin_mo1eint.array)

    def compare_spin_mo2eint(self) -> None:
        qp_chem_spin_mo2eint = self.qp_chem_ao_eint_set.ao_2e_int.to_mo2int(
            self.molecule.mo_coeff
        )
        qp_pyscf_spin_mo2eint = self.qp_pyscf_ao_eint_set.ao_2e_int.to_mo2int(
            self.molecule.mo_coeff
        )
        assert allclose(qp_chem_spin_mo2eint.array, qp_pyscf_spin_mo2eint.array)

    def compare_spatial_mo1eint(self) -> None:
        qp_chem_spatial_mo1eint = self.qp_chem_ao_eint_set.ao_1e_int.to_spatial_mo1int(
            self.molecule.mo_coeff
        )
        qp_pyscf_spatial_mo1eint = (
            self.qp_pyscf_ao_eint_set.ao_1e_int.to_spatial_mo1int(
                self.molecule.mo_coeff
            )
        )
        assert allclose(qp_chem_spatial_mo1eint.array, qp_pyscf_spatial_mo1eint.array)

    def compare_spatial_mo2eint(self) -> None:
        qp_chem_spatial_mo2eint = self.qp_chem_ao_eint_set.ao_2e_int.to_spatial_mo2int(
            self.molecule.mo_coeff
        )
        qp_pyscf_spatial_mo2eint = (
            self.qp_pyscf_ao_eint_set.ao_2e_int.to_spatial_mo2int(
                self.molecule.mo_coeff
            )
        )
        assert allclose(qp_chem_spatial_mo2eint.array, qp_pyscf_spatial_mo2eint.array)

    def compare_casci_const(self) -> None:
        assert allclose(
            self.pyscf_active_space_integrals.const, self.active_space_integrals.const
        )

    def compare_casci_spin_mo_1eint(self) -> None:
        assert allclose(
            self.pyscf_active_space_integrals.mo_1e_int.array,
            self.active_space_integrals.mo_1e_int.array,
        )

    def compare_casci_spin_mo_2eint(self) -> None:
        assert allclose(
            self.pyscf_active_space_integrals.mo_2e_int.array,
            self.active_space_integrals.mo_2e_int.array,
        )

    def compare_casci_const_from_spatial(self) -> None:
        assert allclose(
            self.pyscf_active_space_spatial_integrals.const,
            self.active_space_spatial_integrals.const,
        )

    def compare_casci_spatial_mo_1eint(self) -> None:
        assert allclose(
            self.pyscf_active_space_spatial_integrals.mo_1e_int.array,
            self.active_space_spatial_integrals.mo_1e_int.array,
        )

    def compare_casci_spatial_mo_2eint(self) -> None:
        assert allclose(
            self.pyscf_active_space_spatial_integrals.mo_2e_int.array,
            self.active_space_spatial_integrals.mo_2e_int.array,
        )

    def run_test(self) -> None:
        self.compare_ao1eint()
        self.compare_ao2eint()
        self.compare_spin_mo1eint()
        self.compare_spin_mo2eint()
        self.compare_spatial_mo1eint()
        self.compare_spatial_mo2eint()
        self.compare_casci_const()
        self.compare_casci_spin_mo_1eint()
        self.compare_casci_spin_mo_2eint()
        self.compare_casci_const_from_spatial()
        self.compare_casci_spatial_mo_1eint()
        self.compare_casci_spatial_mo_2eint()


class TestH2O(TestMolecule):
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
            cls.qp_chem_ao_eint_set,
            cls.qp_pyscf_ao_eint_set,
            cls.pyscf_active_space_integrals,
            cls.active_space_integrals,
            cls.pyscf_active_space_spatial_integrals,
            cls.active_space_spatial_integrals,
        ) = compute_integrals(
            molecule=cls.molecule, active_space_mo=cls.active_space_mo
        )

    def test_integrals(self) -> None:
        self.run_test()


class TestSpinningH2O(TestMolecule):
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
            cls.qp_chem_ao_eint_set,
            cls.qp_pyscf_ao_eint_set,
            cls.pyscf_active_space_integrals,
            cls.active_space_integrals,
            cls.pyscf_active_space_spatial_integrals,
            cls.active_space_spatial_integrals,
        ) = compute_integrals(
            molecule=cls.molecule, active_space_mo=cls.active_space_mo
        )

    def test_integrals(self) -> None:
        self.run_test()


class TestCCPVDZH2O(TestMolecule):
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
            cls.qp_chem_ao_eint_set,
            cls.qp_pyscf_ao_eint_set,
            cls.pyscf_active_space_integrals,
            cls.active_space_integrals,
            cls.pyscf_active_space_spatial_integrals,
            cls.active_space_spatial_integrals,
        ) = compute_integrals(
            molecule=cls.molecule, active_space_mo=cls.active_space_mo
        )

    def test_integrals(self) -> None:
        self.run_test()


class TestChargedH2O(TestMolecule):
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
            cls.qp_chem_ao_eint_set,
            cls.qp_pyscf_ao_eint_set,
            cls.pyscf_active_space_integrals,
            cls.active_space_integrals,
            cls.pyscf_active_space_spatial_integrals,
            cls.active_space_spatial_integrals,
        ) = compute_integrals(
            molecule=cls.molecule, active_space_mo=cls.active_space_mo
        )

    def test_integrals(self) -> None:
        self.run_test()
