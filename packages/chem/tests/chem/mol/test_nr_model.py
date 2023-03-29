from typing import Optional, Sequence, cast

from numpy import complex128, isclose
from numpy.typing import NDArray
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf  # type: ignore

from quri_parts.pyscf.mol.factory import MolecularHamiltonian, PySCFMoleculeFactoryInput


class OpenFermionMolecularHamiltonian:
    def __init__(self, molecule: PySCFMoleculeFactoryInput) -> None:
        mol = MolecularData(
            geometry=molecule.atom,
            multiplicity=molecule.spin + 1,
            basis=molecule.basis,
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

h2o = PySCFMoleculeFactoryInput(atom=h2o_atom_list)
h2o_spin = PySCFMoleculeFactoryInput(atom=h2o_atom_list, spin=2)
hcl = PySCFMoleculeFactoryInput(atom=hcl_atom_list)
ch3no = PySCFMoleculeFactoryInput(atom=ch3no_atom_list, basis="cc-pvdz")

qp_h2o = MolecularHamiltonian(h2o)
qp_h2o_spin = MolecularHamiltonian(h2o_spin)
qp_hcl = MolecularHamiltonian(hcl)
qp_ch3no = MolecularHamiltonian(ch3no)

of_h2o = OpenFermionMolecularHamiltonian(h2o)
of_h2o_spin = OpenFermionMolecularHamiltonian(h2o_spin)
of_hcl = OpenFermionMolecularHamiltonian(hcl)
of_ch3no = OpenFermionMolecularHamiltonian(ch3no)


def qp_of_comparison(
    qp_molecule: MolecularHamiltonian,
    of_molecule: OpenFermionMolecularHamiltonian,
    n_active_ele: Optional[int] = None,
    n_active_orb: Optional[int] = None,
    active_orbs_indices: Optional[Sequence[int]] = None,
) -> None:
    # qp computation
    hamiltonian_component = qp_molecule.get_molecular_hamiltonian(
        n_active_ele=n_active_ele,
        n_active_orb=n_active_orb,
        active_orbs_indices=active_orbs_indices,
    )
    core_energy, spin_1e_int, spin_2e_int = (
        hamiltonian_component.const,
        hamiltonian_component.mo_1e_int.array,
        hamiltonian_component.mo_2e_int.array,
    )

    if n_active_ele is not None and n_active_orb is not None:
        (
            occupied_indices,
            active_indices,
        ) = qp_molecule.get_active_space_molecular_orbitals(
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
    assert isclose(of_spin_2e_int - spin_2e_int, 0).all()


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


# def test_hcl_all_active() -> None:
#     """Github pytest error to be figured out.

#     Local PyTest passes without any problem
#     """
#     print("\n")
#     qp_of_comparison(qp_hcl, of_hcl)


# def test_ch3no_all_active() -> None:
#     print("\n")
#     # takes too long to execute
#     qp_of_comparison(qp_ch3no, of_ch3no)
