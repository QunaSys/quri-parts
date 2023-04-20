from typing import Any, Optional, Sequence, Union, cast

import pyscf
from numpy import isclose
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf  # type: ignore

from quri_parts.chem.mol import ActiveSpace, ActiveSpaceMolecularOrbitals, SpinMOeIntSet
from quri_parts.pyscf.mol import PySCFMolecularOrbitals, get_ao_eint_set


def get_pyscf_mo(
    atom: list[Any],
    spin: int = 0,
    charge: int = 0,
    basis: str = "sto-3g",
    symmetry: Union[bool, str] = False,
    verbose: int = 0,
) -> PySCFMolecularOrbitals:
    """construct PySCFMolecularOrbitals."""
    mol = pyscf.gto.Mole(
        atom=atom,
        basis=basis,
        charge=charge,
        spin=spin,
        symmetry=symmetry,
        verbose=verbose,
    )
    mol.build()

    if spin:
        mol_coeff = pyscf.scf.ROHF(mol).run().mo_coeff
    else:
        mol_coeff = pyscf.scf.RHF(mol).run().mo_coeff

    mo = PySCFMolecularOrbitals(mol=mol, mo_coeff=mol_coeff)
    return mo


def get_active_space_mo(
    mol: PySCFMolecularOrbitals,
    n_active_ele: int,
    n_active_orb: int,
    active_orbs_indices: Optional[Sequence[int]] = None,
) -> ActiveSpaceMolecularOrbitals:
    """Construct instance of active ActiveSpaceMolecularOrbitals automatically
    by assigning:

    - n_active_ele
    - n_active_orb
    - active_orbs_indices
    """
    active_space = ActiveSpace(
        n_active_ele,
        n_active_orb,
        active_orbs_indices,
    )
    return ActiveSpaceMolecularOrbitals(mol, active_space)


def get_active_space_integrals(
    mol: PySCFMolecularOrbitals,
    n_active_ele: int,
    n_active_orb: int,
    active_orbs_indices: Optional[Sequence[int]] = None,
) -> SpinMOeIntSet:
    """Get active space integral by assigning active space information."""
    ao_eint_set = get_ao_eint_set(molecule=mol)
    active_space_mo = get_active_space_mo(
        mol, n_active_ele, n_active_orb, active_orbs_indices
    )
    active_space_mo_eint_set = ao_eint_set.to_active_space_mo_int(active_space_mo)
    return cast(SpinMOeIntSet, active_space_mo_eint_set)


def qp_of_comparison(
    atom: list[Any],
    spin: int,
    basis: str,
    n_active_ele: Optional[int] = None,
    n_active_orb: Optional[int] = None,
    active_orbs_indices: Optional[Sequence[int]] = None,
) -> None:
    """Generate the quri-parts computed electron integrals and compare them
    with openfermion results."""
    qp_molecule = get_pyscf_mo(atom=atom, spin=spin, basis=basis)

    # Get occupied_indices and active_indices indices.
    if n_active_ele is None and n_active_orb is None:
        n_active_ele = qp_molecule.n_electron
        n_active_orb = qp_molecule.n_spatial_orb
        occupied_indices, active_indices = cast(Sequence[int], None), cast(
            Sequence[int], None
        )

    else:
        (
            occupied_indices,
            active_indices,
        ) = get_active_space_mo(
            mol=qp_molecule,
            n_active_ele=cast(int, n_active_ele),
            n_active_orb=cast(int, n_active_orb),
            active_orbs_indices=active_orbs_indices,
        ).get_core_and_active_orb()

    # qp computation
    hamiltonian_component = get_active_space_integrals(
        mol=qp_molecule,
        n_active_ele=cast(int, n_active_ele),
        n_active_orb=cast(int, n_active_orb),
        active_orbs_indices=active_orbs_indices,
    )

    core_energy, spin_1e_int, spin_2e_int = (
        hamiltonian_component.const,
        hamiltonian_component.mo_1e_int.array,
        hamiltonian_component.mo_2e_int.array,
    )

    # of computation
    of_molecule = run_pyscf(
        MolecularData(
            geometry=atom,
            multiplicity=spin + 1,
            basis=basis,
        )
    )
    (
        of_core_energy,
        of_spin_1e_int,
        of_spin_2e_int,
    ) = of_molecule.get_molecular_hamiltonian(
        occupied_indices=occupied_indices, active_indices=active_indices
    ).n_body_tensors.values()

    # Assertions
    assert isclose(of_core_energy - core_energy, 0)
    assert isclose(of_spin_1e_int - spin_1e_int, 0).all()
    assert isclose(of_spin_2e_int - spin_2e_int / 2, 0).all()


def test_h2o() -> None:
    atom = [["O", [0, 0, 0]], ["H", [0, 0, 1]], ["H", [2, 0, 0.5]]]
    spin = 0
    basis = "sto-3g"
    qp_of_comparison(atom=atom, spin=spin, basis=basis, n_active_ele=6, n_active_orb=4)


def test_h2o_spin() -> None:
    atom = [["O", [0, 0, 0]], ["H", [0, 0, 1]], ["H", [2, 0, 0.5]]]
    spin = 2
    basis = "sto-3g"
    qp_of_comparison(atom=atom, spin=spin, basis=basis, n_active_ele=6, n_active_orb=4)


def test_hcl() -> None:
    atom = [["H", [0, 0, 1]], ["Cl", [0, 0, 0]]]
    spin = 0
    basis = "sto-3g"
    qp_of_comparison(atom=atom, spin=spin, basis=basis, n_active_ele=6, n_active_orb=4)


def test_ch3no() -> None:
    atom = [
        ["C", (0.000000, 0.418626, 0.000000)],
        ["H", (-0.460595, 1.426053, 0.000000)],
        ["O", (1.196516, 0.242075, 0.000000)],
        ["N", (-0.936579, -0.568753, 0.000000)],
        ["H", (-0.634414, -1.530889, 0.000000)],
        ["H", (-1.921071, -0.362247, 0.000000)],
    ]
    spin = 0
    basis = "ccpvdz"
    qp_of_comparison(atom=atom, spin=spin, basis=basis, n_active_ele=6, n_active_orb=4)


def test_h2o_all_active() -> None:
    atom = [["O", [0, 0, 0]], ["H", [0, 0, 1]], ["H", [2, 0, 0.5]]]
    spin = 0
    basis = "sto-3g"
    qp_of_comparison(atom=atom, spin=spin, basis=basis)


def test_h2o_spin_all_active() -> None:
    atom = [["O", [0, 0, 0]], ["H", [0, 0, 1]], ["H", [2, 0, 0.5]]]
    spin = 2
    basis = "sto-3g"
    qp_of_comparison(atom=atom, spin=spin, basis=basis)
