from typing import Any, Optional, Sequence, Union, cast

import pyscf
from numpy import complex128, isclose
from numpy.typing import NDArray
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf  # type: ignore

from quri_parts.chem.mol import (
    ActiveSpace,
    ActiveSpaceMolecularOrbitals,
    AOeIntSet,
    MOeIntSet,
    spatial_mo_eint_set_to_spin_mo_eint_set,
)
from quri_parts.pyscf.mol import PySCFMolecularOrbitals, get_ao_eint_set


def get_pyscf_mo(
    atom: list[Any],
    spin: int = 0,
    charge: int = 0,
    basis: str = "sto-3g",
    symmetry: Union[bool, str] = False,
    verbose: int = 0,
) -> PySCFMolecularOrbitals:
    # construct PySCFMolecularOrbitals
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
    active_space = ActiveSpace(
        n_active_ele,
        n_active_orb,
        active_orbs_indices,
    )
    return ActiveSpaceMolecularOrbitals(mol, active_space)


def get_mo_eint_set_from_ao_eint_set(
    molecule: PySCFMolecularOrbitals, ao_eint_set: AOeIntSet
) -> MOeIntSet:
    mo_1e_int = ao_eint_set.ao_1e_int.to_mo1int(molecule.mo_coeff)
    mo_2e_int = ao_eint_set.ao_2e_int.to_mo2int(molecule.mo_coeff)

    mo_eint_set = MOeIntSet(
        const=ao_eint_set.constant,
        mo_1e_int=mo_1e_int,
        mo_2e_int=mo_2e_int,
    )
    return mo_eint_set


def get_full_space_integrals(mol: PySCFMolecularOrbitals) -> MOeIntSet:
    ao_eint_set = get_ao_eint_set(molecule=mol)
    mo_eint_set = get_mo_eint_set_from_ao_eint_set(
        molecule=mol, ao_eint_set=ao_eint_set
    )
    return spatial_mo_eint_set_to_spin_mo_eint_set(mo_eint_set)


def get_active_space_integrals(
    mol: PySCFMolecularOrbitals,
    n_active_ele: int,
    n_active_orb: int,
    active_orbs_indices: Optional[Sequence[int]] = None,
) -> MOeIntSet:
    ao_eint_set = get_ao_eint_set(molecule=mol)
    active_space_mo = get_active_space_mo(
        mol, n_active_ele, n_active_orb, active_orbs_indices
    )
    active_space_mo_eint_set = ao_eint_set.to_active_space_mo_int(active_space_mo)
    return active_space_mo_eint_set


class OpenFermionMolecularHamiltonian:
    def __init__(self, atom: list[Any], spin: int = 0, basis: str = "sto-3g") -> None:
        mol = MolecularData(
            geometry=atom,
            multiplicity=spin + 1,
            basis=basis,
        )
        self.mol = run_pyscf(mol)

    def get_molecular_hamiltonian(
        self, occupied_indices: Sequence[int], active_indices: Sequence[int]
    ) -> tuple[float, NDArray[complex128], NDArray[complex128]]:
        hamiltonian_component = self.mol.get_molecular_hamiltonian(
            occupied_indices=occupied_indices, active_indices=active_indices
        ).n_body_tensors
        return (
            hamiltonian_component[tuple()],
            hamiltonian_component[(1, 0)],
            hamiltonian_component[(1, 1, 0, 0)],
        )


h2o_atom_list = [["O", [0, 0, 0]], ["H", [0, 0, 1]], ["H", [2, 0, 0.5]]]
hcl_atom_list = [["H", [0, 0, 1]], ["Cl", [0, 0, 0]]]
ch3no_atom_list = [
    ["C", (0.000000, 0.418626, 0.000000)],
    ["H", (-0.460595, 1.426053, 0.000000)],
    ["O", (1.196516, 0.242075, 0.000000)],
    ["N", (-0.936579, -0.568753, 0.000000)],
    ["H", (-0.634414, -1.530889, 0.000000)],
    ["H", (-1.921071, -0.362247, 0.000000)],
]


qp_h2o = get_pyscf_mo(atom=h2o_atom_list)
qp_h2o_spin = get_pyscf_mo(atom=h2o_atom_list, spin=2)
qp_hcl = get_pyscf_mo(atom=hcl_atom_list)
qp_ch3no = get_pyscf_mo(atom=ch3no_atom_list, basis="cc-pvdz")


of_h2o = OpenFermionMolecularHamiltonian(atom=h2o_atom_list)
of_h2o_spin = OpenFermionMolecularHamiltonian(atom=h2o_atom_list, spin=2)
of_hcl = OpenFermionMolecularHamiltonian(atom=hcl_atom_list)
of_ch3no = OpenFermionMolecularHamiltonian(atom=ch3no_atom_list, basis="cc-pvdz")


def qp_of_comparison(
    qp_molecule: PySCFMolecularOrbitals,
    of_molecule: OpenFermionMolecularHamiltonian,
    n_active_ele: Optional[int] = None,
    n_active_orb: Optional[int] = None,
    active_orbs_indices: Optional[Sequence[int]] = None,
) -> None:
    # qp computation
    if n_active_ele and n_active_orb:
        hamiltonian_component = get_active_space_integrals(
            mol=qp_molecule,
            n_active_ele=n_active_ele,
            n_active_orb=n_active_orb,
            active_orbs_indices=active_orbs_indices,
        )
    else:
        hamiltonian_component = get_full_space_integrals(qp_molecule)

    core_energy, spin_1e_int, spin_2e_int = (
        hamiltonian_component.const,
        hamiltonian_component.mo_1e_int.array,
        hamiltonian_component.mo_2e_int.array,
    )

    if n_active_ele is not None and n_active_orb is not None:
        (
            occupied_indices,
            active_indices,
        ) = get_active_space_mo(
            mol=qp_molecule,
            n_active_ele=n_active_ele,
            n_active_orb=n_active_orb,
            active_orbs_indices=active_orbs_indices,
        ).get_core_and_active_orb()
    else:
        occupied_indices, active_indices = cast(Sequence[int], None), cast(
            Sequence[int], None
        )

    # of computation
    (
        of_core_energy,
        of_spin_1e_int,
        of_spin_2e_int,
    ) = of_molecule.get_molecular_hamiltonian(occupied_indices, active_indices)

    # Assertions
    assert isclose(of_core_energy - core_energy, 0)
    assert isclose(of_spin_1e_int - spin_1e_int, 0).all()
    assert isclose(of_spin_2e_int - spin_2e_int / 2, 0).all()


def test_h2o() -> None:
    print("\n")
    qp_of_comparison(qp_h2o, of_h2o, n_active_ele=6, n_active_orb=4)


def test_h2o_spin() -> None:
    print("\n")
    qp_of_comparison(qp_h2o_spin, of_h2o_spin, n_active_ele=6, n_active_orb=4)


def test_hcl() -> None:
    print("\n")
    qp_of_comparison(qp_hcl, of_hcl, n_active_ele=6, n_active_orb=4)


def test_ch3no() -> None:
    print("\n")
    qp_of_comparison(qp_ch3no, of_ch3no, n_active_ele=6, n_active_orb=4)


def test_h2o_all_active() -> None:
    print("\n")
    qp_of_comparison(qp_h2o, of_h2o)


def test_h2o_spin_all_active() -> None:
    print("\n")
    qp_of_comparison(qp_h2o_spin, of_h2o_spin)
