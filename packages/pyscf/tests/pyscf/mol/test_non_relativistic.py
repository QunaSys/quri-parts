from numpy import isclose

from quri_parts.pyscf.mol import (
    PySCFAO1eInt,
    PySCFAO2eInt,
    PySCFMoleculeFactory,
    PySCFMoleculeFactoryInput,
    ao1int,
    ao2int,
)

h2o_atom_list = [["O", [0, 0, 0]], ["H", [0, 0, 1]], ["H", [2, 0, 0.5]]]
hcl_atom_list = [["H", [0, 0, 1]], ["Cl", [0, 0, 0]]]
h2o = PySCFMoleculeFactoryInput(atom=h2o_atom_list)
h2o_spin = PySCFMoleculeFactoryInput(atom=h2o_atom_list, spin=2)
hcl = PySCFMoleculeFactoryInput(atom=hcl_atom_list)


def qp_chem_qp_pyscf_comparison(mol: PySCFMoleculeFactoryInput) -> None:
    molecule = PySCFMoleculeFactory(**mol.__dict__)

    qp_chem_ao1eint = ao1int(molecule)
    qp_chem_mo1eint = qp_chem_ao1eint.to_mo1int(molecule.mo_coeff)
    qp_chem_ao2eint = ao2int(molecule)
    qp_chem_mo2eint = qp_chem_ao2eint.to_mo2int(molecule.mo_coeff)

    qp_pyscf_ao1eint = PySCFAO1eInt(molecule.mol)
    qp_pyscf_mo1eint = qp_pyscf_ao1eint.to_mo1int(molecule.mo_coeff)
    qp_pyscf_ao2eint = PySCFAO2eInt(molecule.mol)
    qp_pyscf_mo2eint = qp_pyscf_ao2eint.to_mo2int(molecule.mo_coeff)

    assert isclose(qp_chem_ao1eint.array - qp_pyscf_ao1eint.array, 0).all()
    assert isclose(qp_chem_ao2eint.array - qp_pyscf_ao2eint.array, 0).all()
    assert isclose(qp_chem_mo1eint.array - qp_pyscf_mo1eint.array, 0).all()
    assert isclose(qp_chem_mo2eint.array - qp_pyscf_mo2eint.array, 0).all()


def test_h2o() -> None:
    qp_chem_qp_pyscf_comparison(h2o)


def test_h2o_spin() -> None:
    qp_chem_qp_pyscf_comparison(h2o_spin)


def test_hcl() -> None:
    qp_chem_qp_pyscf_comparison(hcl)
