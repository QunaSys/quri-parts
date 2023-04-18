from numpy import isclose
from pyscf import gto, scf

from quri_parts.pyscf.mol import (
    PySCFAO1eInt,
    PySCFAO2eInt,
    PySCFMolecularOrbitals,
    ao1int,
    ao2int,
    get_ao_eint_set,
    pyscf_get_ao_eint_set,
)

h2o_atom_list = [["O", [0, 0, 0]], ["H", [0, 0, 1]], ["H", [2, 0, 0.5]]]
h2o_mol = gto.M(atom=h2o_atom_list)
h2o_mf = scf.RHF(h2o_mol)
h2o_mf.kernel()
h2o = PySCFMolecularOrbitals(mol=h2o_mol, mo_coeff=h2o_mf.mo_coeff)

h2o_spin_mol = gto.M(atom=h2o_atom_list, spin=2)
h2o_spin_mf = scf.ROHF(h2o_spin_mol)
h2o_spin_mf.kernel()
h2o_spin = PySCFMolecularOrbitals(mol=h2o_spin_mol, mo_coeff=h2o_spin_mf.mo_coeff)


hcl_atom_list = [["H", [0, 0, 1]], ["Cl", [0, 0, 0]]]
hcl_mol = gto.M(atom=hcl_atom_list)
hcl_mf = scf.RHF(hcl_mol)
hcl_mf.kernel()
hcl = PySCFMolecularOrbitals(mol=hcl_mol, mo_coeff=hcl_mf.mo_coeff)


def qp_chem_qp_pyscf_comparison(molecule: PySCFMolecularOrbitals) -> None:
    """ Compare output of AOeIntArraySet and PySCFAOeIntSet
    """
    qp_chem_ao_eint_set = get_ao_eint_set(molecule)
    qp_chem_ao1eint = qp_chem_ao_eint_set.ao_1e_int
    qp_chem_spin_mo1eint = qp_chem_ao1eint.to_mo1int(molecule.mo_coeff)
    qp_chem_spatial_mo1eint = qp_chem_ao1eint.to_spatial_mo1int(molecule.mo_coeff)
    qp_chem_ao2eint = qp_chem_ao_eint_set.ao_2e_int
    qp_chem_spin_mo2eint = qp_chem_ao2eint.to_mo2int(molecule.mo_coeff)
    qp_chem_spatial_mo2eint = qp_chem_ao2eint.to_spatial_mo2int(molecule.mo_coeff)

    qp_pyscf_ao_eint_set = pyscf_get_ao_eint_set(molecule)
    qp_pyscf_ao1eint = qp_pyscf_ao_eint_set.ao_1e_int
    qp_pyscf_spin_mo1eint = qp_pyscf_ao1eint.to_mo1int(molecule.mo_coeff)
    qp_pyscf_spatial_mo1eint = qp_pyscf_ao1eint.to_spatial_mo1int(molecule.mo_coeff)
    qp_pyscf_ao2eint = qp_pyscf_ao_eint_set.ao_2e_int
    qp_pyscf_spin_mo2eint = qp_pyscf_ao2eint.to_mo2int(molecule.mo_coeff)
    qp_pyscf_spatial_mo2eint = qp_pyscf_ao2eint.to_spatial_mo2int(molecule.mo_coeff)

    assert isclose(qp_chem_ao1eint.array - qp_pyscf_ao1eint.array, 0).all()
    assert isclose(qp_chem_ao2eint.array - qp_pyscf_ao2eint.array, 0).all()
    assert isclose(qp_chem_spin_mo1eint.array - qp_pyscf_spin_mo1eint.array, 0).all()
    assert isclose(qp_chem_spin_mo2eint.array - qp_pyscf_spin_mo2eint.array, 0).all()
    assert isclose(qp_chem_spatial_mo1eint.array - qp_pyscf_spatial_mo1eint.array, 0).all()
    assert isclose(qp_chem_spatial_mo2eint.array - qp_pyscf_spatial_mo2eint.array, 0).all()

def test_h2o() -> None:
    qp_chem_qp_pyscf_comparison(h2o)


def test_h2o_spin() -> None:
    qp_chem_qp_pyscf_comparison(h2o_spin)


def test_hcl() -> None:
    qp_chem_qp_pyscf_comparison(hcl)
