import unittest

from numpy import allclose
from pyscf import gto, scf

from quri_parts.chem.mol import ActiveSpaceMolecularOrbitals, cas
from quri_parts.pyscf.mol import PySCFMolecularOrbitals, get_ao_eint_set


class TestH2O(unittest.TestCase):
    """Compare output of AOeIntArraySet and PySCFAOeIntSet."""

    atom_list = [["O", [0, 0, 0]], ["H", [0, 0, 1]], ["H", [2, 0, 0.5]]]
    mol = gto.M(atom=atom_list)
    mf = scf.RHF(mol)
    mf.kernel()
    molecule = PySCFMolecularOrbitals(mol=mol, mo_coeff=mf.mo_coeff)

    def setUp(self) -> None:
        self.qp_chem_ao_eint_set = get_ao_eint_set(
            self.molecule, store_array_on_memory=True
        )
        self.qp_pyscf_ao_eint_set = get_ao_eint_set(self.molecule)

        active_space_mo = ActiveSpaceMolecularOrbitals(
            self.molecule, cas(n_active_ele=6, n_active_orb=4)
        )

        self.pyscf_active_space_integrals = (
            self.qp_pyscf_ao_eint_set.to_active_space_mo_int(active_space_mo)
        )
        self.active_space_integrals = self.qp_chem_ao_eint_set.to_active_space_mo_int(
            active_space_mo
        )

        self.pyscf_active_space_spatial_integrals = (
            self.qp_pyscf_ao_eint_set.to_active_space_spatial_mo_int(active_space_mo)
        )
        self.active_space_spatial_integrals = (
            self.qp_chem_ao_eint_set.to_active_space_spatial_mo_int(active_space_mo)
        )

    def test_ao1eint(self) -> None:
        qp_chem_ao1eint = self.qp_chem_ao_eint_set.ao_1e_int
        qp_pyscf_ao1eint = self.qp_pyscf_ao_eint_set.ao_1e_int
        assert allclose(qp_chem_ao1eint.array, qp_pyscf_ao1eint.array)

    def test_ao2eint(self) -> None:
        qp_chem_ao2eint = self.qp_chem_ao_eint_set.ao_2e_int
        qp_pyscf_ao2eint = self.qp_pyscf_ao_eint_set.ao_2e_int
        assert allclose(qp_chem_ao2eint.array, qp_pyscf_ao2eint.array)

    def test_spin_mo1eint(self) -> None:
        qp_chem_spin_mo1eint = self.qp_chem_ao_eint_set.ao_1e_int.to_mo1int(
            self.molecule.mo_coeff
        )
        qp_pyscf_spin_mo1eint = self.qp_pyscf_ao_eint_set.ao_1e_int.to_mo1int(
            self.molecule.mo_coeff
        )
        assert allclose(qp_chem_spin_mo1eint.array, qp_pyscf_spin_mo1eint.array)

    def test_spin_mo2eint(self) -> None:
        qp_chem_spin_mo2eint = self.qp_chem_ao_eint_set.ao_2e_int.to_mo2int(
            self.molecule.mo_coeff
        )
        qp_pyscf_spin_mo2eint = self.qp_pyscf_ao_eint_set.ao_2e_int.to_mo2int(
            self.molecule.mo_coeff
        )
        assert allclose(qp_chem_spin_mo2eint.array, qp_pyscf_spin_mo2eint.array)

    def test_spatial_mo1eint(self) -> None:
        qp_chem_spatial_mo1eint = self.qp_chem_ao_eint_set.ao_1e_int.to_spatial_mo1int(
            self.molecule.mo_coeff
        )
        qp_pyscf_spatial_mo1eint = (
            self.qp_pyscf_ao_eint_set.ao_1e_int.to_spatial_mo1int(
                self.molecule.mo_coeff
            )
        )
        assert allclose(qp_chem_spatial_mo1eint.array, qp_pyscf_spatial_mo1eint.array)

    def test_spatial_mo2eint(self) -> None:
        qp_chem_spatial_mo2eint = self.qp_chem_ao_eint_set.ao_2e_int.to_spatial_mo2int(
            self.molecule.mo_coeff
        )
        qp_pyscf_spatial_mo2eint = (
            self.qp_pyscf_ao_eint_set.ao_2e_int.to_spatial_mo2int(
                self.molecule.mo_coeff
            )
        )
        assert allclose(qp_chem_spatial_mo2eint.array, qp_pyscf_spatial_mo2eint.array)

    def test_casci_const(self) -> None:
        assert allclose(
            self.pyscf_active_space_integrals.const, self.active_space_integrals.const
        )

    def test_casci_spin_mo_1eint(self) -> None:
        assert allclose(
            self.pyscf_active_space_integrals.mo_1e_int.array,
            self.active_space_integrals.mo_1e_int.array,
        )

    def test_casci_spin_mo_2eint(self) -> None:
        assert allclose(
            self.pyscf_active_space_integrals.mo_2e_int.array,
            self.active_space_integrals.mo_2e_int.array,
        )

    def test_casci_const_from_spatial(self) -> None:
        assert allclose(
            self.pyscf_active_space_spatial_integrals.const,
            self.active_space_spatial_integrals.const,
        )

    def test_casci_spatial_mo_1eint(self) -> None:
        assert allclose(
            self.pyscf_active_space_spatial_integrals.mo_1e_int.array,
            self.active_space_spatial_integrals.mo_1e_int.array,
        )

    def test_casci_spatial_mo_2eint(self) -> None:
        assert allclose(
            self.pyscf_active_space_spatial_integrals.mo_2e_int.array,
            self.active_space_spatial_integrals.mo_2e_int.array,
        )


class TestSpinningH2O(TestH2O):
    atom_list = [["O", [0, 0, 0]], ["H", [0, 0, 1]], ["H", [2, 0, 0.5]]]
    mol = gto.M(atom=atom_list)
    mf = scf.ROHF(mol)
    mf.kernel()
    molecule = PySCFMolecularOrbitals(mol=mol, mo_coeff=mf.mo_coeff)


class TestSpinningHCl(TestH2O):
    atom_list = [["H", [0, 0, 1]], ["Cl", [0, 0, 0]]]
    mol = gto.M(atom=atom_list)
    mf = scf.RHF(mol)
    mf.kernel()
    molecule = PySCFMolecularOrbitals(mol=mol, mo_coeff=mf.mo_coeff)
