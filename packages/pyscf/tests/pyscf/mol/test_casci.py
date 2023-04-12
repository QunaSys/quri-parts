from quri_parts.chem.mol import ActiveSpaceMolecularOrbitals, ActiveSpace
from quri_parts.pyscf.mol import PySCFMolecularOrbitals, get_ao_eint_set, pyscf_get_ao_eint_set
from numpy import allclose
import pyscf

h2o_atom_list = [['H', [0, 0, 0]], ['O', [2, 0, 1]], ['H',  [0, 0, 2]]]
h2o_mol = pyscf.gto.Mole(
    atom=h2o_atom_list, 
    charge = 0,
    spin = 0,
    basis = 'sto-3g',
    verbose = 0
)
h2o_mol.build()
h2o_mf = pyscf.scf.ROHF(h2o_mol)
h2o_mf.kernel()

h2o_mo = PySCFMolecularOrbitals(mol=h2o_mol, mo_coeff=h2o_mf.mo_coeff)

pyscf_ao_eint_set = pyscf_get_ao_eint_set(h2o_mo)
ao_eint_set = get_ao_eint_set(h2o_mo)

h2o_active_space_mo = ActiveSpaceMolecularOrbitals(h2o_mo, ActiveSpace(n_active_ele=6, n_active_orb=4))

pyscf_active_space_integrals = pyscf_ao_eint_set.to_active_space_mo_int(h2o_active_space_mo)
active_space_integrals = ao_eint_set.to_active_space_mo_int(h2o_active_space_mo)


def test_casci_result() -> None:
    assert allclose(pyscf_active_space_integrals.const, active_space_integrals.const)
    assert allclose(pyscf_active_space_integrals.mo_1e_int.array, active_space_integrals.mo_1e_int.array)
    assert allclose(pyscf_active_space_integrals.mo_2e_int.array, active_space_integrals.mo_2e_int.array)
    