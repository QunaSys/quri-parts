from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
from pyscf import ao2mo, gto, mcscf, scf

from quri_parts.chem.mol import (
    ActiveSpaceMolecularOrbitals,
    AO1eInt,
    AO1eIntArray,
    AO2eInt,
    AO2eIntArray,
    AOeIntArraySet,
    AOeIntSet,
    MO1eInt,
    MO1eIntArray,
    MO2eInt,
    MO2eIntArray,
    MOeIntSet,
    spatial_mo_eint_set_to_spin_mo_eint_set,
)

from .model import PySCFMolecularOrbitals, get_nuc_energy


class PySCFAO1eInt(AO1eInt):
    def __init__(self, mol: gto.Mole) -> None:
        self._mol = mol

    @property
    def array(self) -> "npt.NDArray[np.complex128]":
        """Computes the ao 1-electron integrals with pyscf."""
        ao_int1e = scf.hf.get_hcore(self._mol)
        return cast(npt.NDArray[np.complex128], ao_int1e)

    def to_mo1int(self, mo_coeff: "npt.NDArray[np.complex128]") -> MO1eInt:
        return PySCFMO1eInt(mol=self._mol, mo_coeff=mo_coeff)


class PySCFAO2eInt(AO2eInt):
    def __init__(self, mol: gto.Mole) -> None:
        self._mol = mol

    @property
    def array(self) -> "npt.NDArray[np.complex128]":
        """Computes the ao 2-electron integrals in a memory effecient way, and
        returns the array in physicist's notation."""
        ao_int2e = ao2mo.get_ao_eri(mol=self._mol)
        ao_int2e = ao2mo.restore(1, ao_int2e, self._mol.nao).transpose(0, 2, 3, 1)
        return cast(npt.NDArray[np.complex128], ao_int2e)

    def to_mo2int(self, mo_coeff: "npt.NDArray[np.complex128]") -> MO2eInt:
        return PySCFMO2eInt(mol=self._mol, mo_coeff=mo_coeff)


@dataclass
class PySCFAOeIntSet(AOeIntSet):
    mol: gto.Mole
    constant: float
    ao_1e_int: PySCFAO1eInt
    ao_2e_int: PySCFAO2eInt

    def to_active_space_mo_int(
        self, active_space_mo: ActiveSpaceMolecularOrbitals
    ) -> MOeIntSet:
        return pyscf_get_active_space_integrals(active_space_mo, self)


class PySCFMO1eInt(MO1eInt):
    def __init__(self, mol: gto.Mole, mo_coeff: npt.NDArray[np.complex128]) -> None:
        self._mol = mol
        self._mo_coeff = mo_coeff

    @property
    def array(self) -> "npt.NDArray[np.complex128]":
        """Computes the mo 1-electron integrals and returns the array in
        physicist's notation."""
        h1_mo = (
            self._mo_coeff.conjugate().T @ scf.hf.get_hcore(self._mol) @ self._mo_coeff
        )
        return cast(npt.NDArray[np.complex128], h1_mo)


class PySCFMO2eInt(MO2eInt):
    def __init__(self, mol: gto.Mole, mo_coeff: npt.NDArray[np.complex128]) -> None:
        self._mol = mol
        self._mo_coeff = mo_coeff

    @property
    def array(self) -> "npt.NDArray[np.complex128]":
        """Computes the mo 2-electron integrals in a memory effecient way, and
        returns the array in physicist's notation."""
        # Pass in Mole object to the eri_or_mol kwarg for memory efficiency
        # The ao2mo.get_mo_eri function uses ao2mo.outcore to
        # evaluate the matrix elements
        mo_int_2e = ao2mo.get_mo_eri(eri_or_mol=self._mol, mo_coeffs=self._mo_coeff)
        mo_int_2e = ao2mo.restore(1, mo_int_2e, self._mol.nao)
        mo_int_2e = mo_int_2e.transpose(0, 2, 3, 1)
        return cast(npt.NDArray[np.complex128], mo_int_2e)


def ao1int(mo: PySCFMolecularOrbitals) -> PySCFAO1eInt:
    """Calculate the atomic orbital one-electron integral in a memory effcient
    way."""
    return PySCFAO1eInt(mol=mo.mol)


def ao2int(mo: PySCFMolecularOrbitals) -> PySCFAO2eInt:
    """Calculate the atomic orbital two-electron integral in a memory effcient
    way."""
    return PySCFAO2eInt(mol=mo.mol)


def get_ao_eint_set(molecule: PySCFMolecularOrbitals) -> AOeIntSet:
    nuc_energy = get_nuc_energy(molecule)
    ao_1e_int = AO1eIntArray(ao1int(molecule).array)
    ao_2e_int = AO2eIntArray(ao2int(molecule).array)
    ao_e_int_set = AOeIntArraySet(
        constant=nuc_energy,
        ao_1e_int=ao_1e_int,
        ao_2e_int=ao_2e_int,
    )
    return ao_e_int_set


def pyscf_get_ao_eint_set(molecule: PySCFMolecularOrbitals) -> PySCFAOeIntSet:
    nuc_energy = get_nuc_energy(molecule)
    ao_1e_int = ao1int(molecule)
    ao_2e_int = ao2int(molecule)
    ao_e_int_set = PySCFAOeIntSet(
        mol=molecule.mol,
        constant=nuc_energy,
        ao_1e_int=ao_1e_int,
        ao_2e_int=ao_2e_int,
    )
    return ao_e_int_set


def pyscf_get_active_space_integrals(
    active_space_mo: ActiveSpaceMolecularOrbitals, electron_ints: PySCFAOeIntSet
) -> MOeIntSet:
    n_active_orb, n_active_ele = (
        active_space_mo.n_active_orb,
        active_space_mo.n_active_ele,
    )
    cas_mf = mcscf.CASCI(electron_ints.mol, n_active_orb, n_active_ele)
    _, active_orb_list = active_space_mo.get_core_and_active_orb()
    if active_orb_list:
        mo = cas_mf.sort_mo(active_orb_list, mo_coeff=active_space_mo.mo_coeff, base=0)
    else:
        mo = active_space_mo.mo_coeff

    casci_mo_1e_int, casscf_nuc = cas_mf.get_h1eff(mo)
    casci_mo_2e_int = cas_mf.get_h2eff(mo)
    casci_mo_2e_int = ao2mo.restore(1, casci_mo_2e_int, cas_mf.ncas).transpose(
        0, 2, 3, 1
    )

    spatial_integrals = MOeIntSet(
        const=casscf_nuc,
        mo_1e_int=MO1eIntArray(casci_mo_1e_int),
        mo_2e_int=MO2eIntArray(casci_mo_2e_int),
    )

    spin_integrals = spatial_mo_eint_set_to_spin_mo_eint_set(spatial_integrals)

    return spin_integrals
