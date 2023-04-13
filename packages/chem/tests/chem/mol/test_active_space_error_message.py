import pyscf
import pytest

from quri_parts.chem.mol import ActiveSpace, ActiveSpaceMolecularOrbitals
from quri_parts.pyscf.mol import PySCFMolecularOrbitals

h2o_atom_list = [["H", [0, 0, 0]], ["O", [2, 0, 1]], ["H", [0, 0, 2]]]

h2o_mol = pyscf.gto.Mole(
    atom=h2o_atom_list, charge=0, spin=0, basis="sto-3g", verbose=0
)
h2o_mol.build()
h2o_mf = pyscf.scf.RHF(h2o_mol)
h2o_mf.kernel()
h2o = PySCFMolecularOrbitals(h2o_mol, h2o_mf.mo_coeff)


charged_h2o_mol = pyscf.gto.Mole(
    atom=h2o_atom_list, charge=1, spin=1, basis="sto-3g", verbose=0
)
charged_h2o_mol.build()
charged_h2o_mf = pyscf.scf.RHF(charged_h2o_mol)
charged_h2o_mf.kernel()
charged_h2o = PySCFMolecularOrbitals(charged_h2o_mol, charged_h2o_mf.mo_coeff)


spinning_h2o_mol = pyscf.gto.Mole(
    atom=h2o_atom_list, charge=0, spin=6, basis="ccpvdz", verbose=0
)
spinning_h2o_mol.build()
spinning_h2o_mf = pyscf.scf.ROHF(spinning_h2o_mol)
spinning_h2o_mf.kernel()
spinning_h2o = PySCFMolecularOrbitals(spinning_h2o_mol, spinning_h2o_mf.mo_coeff)


neg_spinning_h2o_mol = pyscf.gto.Mole(
    atom=h2o_atom_list, charge=0, spin=-6, basis="ccpvdz", verbose=0
)
neg_spinning_h2o_mol.build()
neg_spinning_h2o_mf = pyscf.scf.ROHF(neg_spinning_h2o_mol)
neg_spinning_h2o_mf.kernel()
neg_spinning_h2o = PySCFMolecularOrbitals(
    neg_spinning_h2o_mol, neg_spinning_h2o_mf.mo_coeff
)


def test_odd_core_ele() -> None:
    with pytest.raises(
        AssertionError,
        match=(
            "The number of electrons in core must be even."
            " Please set the active electron to a even number"
        ),
    ):
        ActiveSpaceMolecularOrbitals(h2o, ActiveSpace(n_active_ele=3, n_active_orb=2))


def test_neg_core_ele() -> None:
    n_active_ele = 100
    n_active_orb = 2
    n_electron = h2o.n_electron
    n_core_ele = n_electron - n_active_ele
    with pytest.raises(
        AssertionError,
        match=(
            f"Number of core electrons should be a positive integer or zero.\n"
            f" n_core_ele = {n_core_ele}.\n"
            f" Possible fix: n_active_ele should be less than"
            f" total number of electrons: {n_electron}."
        ),
    ):
        ActiveSpaceMolecularOrbitals(
            h2o, ActiveSpace(n_active_ele=n_active_ele, n_active_orb=n_active_orb)
        )


def test_neg_vir_orb() -> None:
    n_active_ele = 6
    n_active_orb = 6
    n_electron = h2o.n_electron
    n_spatial_orb = h2o.n_spatial_orb
    n_core_ele = n_electron - n_active_ele
    n_core_orb = n_core_ele // 2
    n_vir_orb = n_spatial_orb - n_active_orb - n_core_orb
    import re

    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Number of virtual orbitals should be a positive integer or zero.\n"
            f" n_vir = {n_vir_orb}.\n"
            f" Possible fix: (n_active_orb - n_active_ele//2) should not"
            f" exceed {n_spatial_orb - n_electron//2}."
        ),
    ):
        ActiveSpaceMolecularOrbitals(
            h2o, ActiveSpace(n_active_ele=n_active_ele, n_active_orb=n_active_orb)
        )


def test_neg_vir_orb_2() -> None:
    n_active_ele = 7
    n_active_orb = 7
    n_electron = charged_h2o.n_electron
    n_spatial_orb = charged_h2o.n_spatial_orb
    n_core_ele = n_electron - n_active_ele
    n_core_orb = n_core_ele // 2
    n_vir_orb = n_spatial_orb - n_active_orb - n_core_orb

    import re

    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Number of virtual orbitals should be a positive integer or zero.\n"
            f" n_vir = {n_vir_orb}.\n"
            f" Possible fix: (n_active_orb - n_active_ele//2) should not"
            f" exceed {n_spatial_orb - n_electron//2}."
        ),
    ):
        ActiveSpaceMolecularOrbitals(
            charged_h2o,
            ActiveSpace(n_active_ele=n_active_ele, n_active_orb=n_active_orb),
        )


def test_neg_alpha_elec() -> None:
    n_active_ele = 4
    n_active_orb = 4
    spin = neg_spinning_h2o.spin
    n_ele_beta = (n_active_ele - spin) // 2
    n_ele_alpha = n_active_ele - n_ele_beta

    import re

    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Number of spin up electrons should be a positive integer or zero.\n"
            f" n_ele_alpha = {n_ele_alpha}"
            f" Possible_fix: - n_active_ele should not"
            f" be less then the value of spin: {spin}."
        ),
    ):
        ActiveSpaceMolecularOrbitals(
            neg_spinning_h2o,
            ActiveSpace(n_active_ele=n_active_ele, n_active_orb=n_active_orb),
        )


def test_neg_beta_elec() -> None:
    n_active_ele = 4
    n_active_orb = 4
    spin = spinning_h2o.spin
    n_ele_beta = (n_active_ele - spin) // 2

    import re

    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Number of spin down electrons should be a positive integer or zero.\n"
            f" n_ele_beta = {n_ele_beta}.\n"
            f" Possible_fix: n_active_ele should not"
            f" exceed the value of spin: {spin}."
        ),
    ):
        ActiveSpaceMolecularOrbitals(
            spinning_h2o,
            ActiveSpace(n_active_ele=n_active_ele, n_active_orb=n_active_orb),
        )


def test_large_alpha_ele() -> None:
    n_active_ele = 8
    n_active_orb = 2
    spin = spinning_h2o.spin
    n_ele_beta = (n_active_ele - spin) // 2
    n_ele_alpha = n_active_ele - n_ele_beta

    import re

    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Number of spin up electrons should not exceed the number of active orbitals.\n"  # noqa: E501
            f" n_ele_alpha = {n_ele_alpha},\n"
            f" n_active_orb = {n_active_orb}.\n"
            f" Possible fix: [(n_active_ele + spin)//2] should be"
            f" less than n_active_orb: {n_active_orb}"
        ),
    ):
        ActiveSpaceMolecularOrbitals(
            spinning_h2o,
            ActiveSpace(n_active_ele=n_active_ele, n_active_orb=n_active_orb),
        )


def test_large_beta_ele() -> None:
    n_active_ele = 8
    n_active_orb = 2
    spin = neg_spinning_h2o.spin
    n_ele_beta = (n_active_ele - spin) // 2

    import re

    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Number of spin down electrons should not exceed the number of active orbitals.\n"  # noqa: E501
            f" n_ele_beta = {n_ele_beta},\n"
            f" n_active_orb = {n_active_orb}"
            f" Possible fix: [(n_active_ele - spin)//2] should be"
            f" less than n_active_orb: {n_active_orb}"
        ),
    ):
        ActiveSpaceMolecularOrbitals(
            neg_spinning_h2o,
            ActiveSpace(n_active_ele=n_active_ele, n_active_orb=n_active_orb),
        )
