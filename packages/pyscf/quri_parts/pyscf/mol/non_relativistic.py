from typing import cast

import numpy as np
import numpy.typing as npt
from pyscf import ao2mo, gto, scf

from quri_parts.chem.mol import (
    AO1eInt,
    AO1eIntArray,
    AO2eInt,
    AO2eIntArray,
    MO1eInt,
    MO2eInt,
)

from .pyscf_interface import PySCFMolecularOrbitals


def ao1int(mo: PySCFMolecularOrbitals) -> AO1eIntArray:
    """Calculate the atomic orbital one-electron integral."""
    h1e = scf.hf.get_hcore(mo.mol)
    return AO1eIntArray(ao1eint_array=h1e)


def ao2int(mo: PySCFMolecularOrbitals) -> AO2eIntArray:
    """Calculate the atomic orbital two-electron integral."""
    a2e_int = mo.mol.intor("int2e").transpose(0, 2, 3, 1)
    return AO2eIntArray(ao2eint_array=a2e_int)


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


def pyscf_ao1int(mo: PySCFMolecularOrbitals) -> PySCFAO1eInt:
    """Calculate the atomic orbital one-electron integral in a memory effcient
    way."""
    return PySCFAO1eInt(mol=mo.mol)


def pyscf_ao2int(mo: PySCFMolecularOrbitals) -> PySCFAO2eInt:
    """Calculate the atomic orbital two-electron integral in a memory effcient
    way."""
    return PySCFAO2eInt(mol=mo.mol)
