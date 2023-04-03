from typing import cast

import numpy as np
import numpy.typing as npt
from pyscf import ao2mo, gto, scf

from quri_parts.chem.mol import (
    AO1eInt,
    AO1eIntBase,
    AO2eInt,
    AO2eIntBase,
    MO1eInt,
    MO1eIntArray,
    MO2eInt,
    MO2eIntArray,
)

from .pyscf_interface import PySCFMolecularOrbitals


def ao1int(mo: PySCFMolecularOrbitals) -> AO1eInt:
    """Calculate the atomic orbital one-electron integral."""
    h1e = scf.hf.get_hcore(mo.mol)
    return AO1eInt(ao1eint_array=h1e)


def ao2int(mo: PySCFMolecularOrbitals) -> AO2eInt:
    """Calculate the atomic orbital two-electron integral."""
    a2e_int = mo.mol.intor("int2e")
    return AO2eInt(ao2eint_array=a2e_int)


class PySCFAO1eInt(AO1eIntBase):
    def __init__(self, mol: gto.Mole) -> None:
        self._mol = mol

    @property
    def array(self) -> "npt.NDArray[np.complex128]":
        """Computes the ao 1-electron integrals in a memory effecient way,
        and returns the array in physicist's notation.
        """
        ao_int1e = scf.hf.get_hcore(self._mol)
        return cast(npt.NDArray[np.complex128], ao_int1e)

    def to_mo1int(self, mo_coeff: "npt.NDArray[np.complex128]") -> MO1eInt:
        """Computes the mo 1-electron integrals and returns the array in physicist's notation.
        """
        h1_mo = mo_coeff.conjugate().T @ self.array @ mo_coeff
        return MO1eIntArray(h1_mo)


class PySCFAO2eInt(AO2eIntBase):
    def __init__(self, mol: gto.Mole) -> None:
        self._mol = mol

    @property
    def array(self) -> "npt.NDArray[np.complex128]":
        """Computes the ao 2-electron integrals in a memory effecient way,
        and returns the array in physicist's notation.
        """
        ao_int2e = ao2mo.get_ao_eri(mol=self._mol)
        ao_int2e = ao2mo.restore(1, ao_int2e, self._mol.nao).transpose(0, 2, 3, 1)
        return cast(npt.NDArray[np.complex128], ao_int2e)

    def to_mo2int(self, mo_coeff: "npt.NDArray[np.complex128]") -> MO2eInt:
        """Computes the mo 2-electron integrals in a memory effecient way,
        and returns the array in physicist's notation.
        """
        # Pass in Mole object to the eri_or_mol kwarg for memory efficiency
        # The ao2mo.get_mo_eri function uses ao2mo.outcore to
        # evaluate the matrix elements
        mo_int_2e = ao2mo.get_mo_eri(eri_or_mol=self._mol, mo_coeffs=mo_coeff)
        mo_int_2e = ao2mo.restore(1, mo_int_2e, self._mol.nao)
        mo_int_2e = mo_int_2e.transpose(0, 2, 3, 1)
        return MO2eIntArray(mo_int_2e)


def pyscf_ao1int(mo: PySCFMolecularOrbitals) -> PySCFAO1eInt:
    """Calculate the atomic orbital one-electron integral in a memory effcient way.
    """
    return PySCFAO1eInt(mol=mo.mol)


def pyscf_ao2int(mo: PySCFMolecularOrbitals) -> PySCFAO2eInt:
    """Calculate the atomic orbital two-electron integral in a memory effcient way.
    """
    return PySCFAO2eInt(mol=mo.mol)
