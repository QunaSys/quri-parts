from dataclasses import dataclass
from itertools import product
from typing import Any, Sequence, Union, cast

import numpy as np
import numpy.typing as npt
from numpy import arange, tensordot, trace, zeros

from .models import (
    ActiveSpaceMolecularOrbitals,
    AO1eInt,
    AO2eInt,
    AOeIntSet,
    MolecularOrbitals,
    SpatialMO1eIntArray,
    SpatialMO2eIntArray,
    SpatialMOeIntSet,
    SpinMO1eInt,
    SpinMO2eInt,
    SpinMOeIntSet,
)


class AO1eIntArray(AO1eInt):
    def __init__(self, ao1eint_array: "npt.NDArray[np.complex128]") -> None:
        self._ao1eint_array = ao1eint_array

    @property
    def array(self) -> "npt.NDArray[np.complex128]":
        """Returns the ao 1-electron integrals in physicist's notation."""
        return self._ao1eint_array

    def to_mo1int(self, mo_coeff: "npt.NDArray[np.complex128]") -> SpatialMO1eIntArray:
        """Returns the spatial mo 1-electron integrals in physicist's
        notation."""
        h1_mo = mo_coeff.conjugate().T @ self._ao1eint_array @ mo_coeff
        return SpatialMO1eIntArray(h1_mo)


class AO2eIntArray(AO2eInt):
    def __init__(self, ao2eint_array: npt.NDArray[np.complex128]) -> None:
        # The input should be in physicist's convention
        self._ao2eint_array = ao2eint_array

    @property
    def array(self) -> "npt.NDArray[np.complex128]":
        """Returns the ao 2-electron integrals in physicist's convention."""
        return self._ao2eint_array

    def to_mo2int(self, mo_coeff: "npt.NDArray[np.complex128]") -> SpatialMO2eIntArray:
        """Returns the spatial mo 2-electron integrals in physicist's
        convention."""
        # Transpose back to chemist convention for computation.
        ao2eint_chem_array = self._ao2eint_array.transpose(0, 3, 1, 2)
        tensor = tensordot(mo_coeff, ao2eint_chem_array, axes=([0], [0]))
        tensor = tensordot(mo_coeff.conjugate(), tensor, axes=([0], [1]))
        tensor = tensordot(mo_coeff, tensor, axes=([0], [2]))
        tensor = tensordot(mo_coeff.conjugate(), tensor, axes=([0], [3]))
        tensor = tensor.transpose(0, 2, 3, 1)
        return SpatialMO2eIntArray(tensor)


@dataclass
class AOeIntArraySet(AOeIntSet):
    constant: float
    ao_1e_int: AO1eIntArray
    ao_2e_int: AO2eIntArray

    def to_full_space_mo_int(
        self, mo: MolecularOrbitals, return_spin_integrals: bool = True
    ) -> Union[SpinMOeIntSet, SpatialMOeIntSet]:
        """Computes the full space spin or spatial mo integrals."""
        spatial_mo_eint_set = SpatialMOeIntSet(
            const=self.constant,
            mo_1e_int=self.ao_1e_int.to_mo1int(mo.mo_coeff),
            mo_2e_int=self.ao_2e_int.to_mo2int(mo.mo_coeff),
        )
        if return_spin_integrals:
            spin_mo_eint_set = spatial_mo_eint_set_to_spin_mo_eint_set(
                spatial_mo_eint_set
            )
            return spin_mo_eint_set
        return spatial_mo_eint_set

    def to_active_space_mo_int(
        self,
        active_space_mo: ActiveSpaceMolecularOrbitals,
        return_spin_integrals: bool = True,
    ) -> Union[SpinMOeIntSet, SpatialMOeIntSet]:
        """Computes the active space mo integrals.

        Note:
        Does not provide speed advantage compared to PySCFAOeIntSet as it
        does not store the mo electron integrals on memory and call the
        get_active_space_integrals_from_mo function.
        """
        return get_active_space_integrals(active_space_mo, self, return_spin_integrals)


def get_effective_active_space_core_energy(
    original_core_energy: float,
    mo_1e_int: npt.NDArray[np.complex128],
    mo_2e_int: npt.NDArray[np.complex128],
    core_spatial_orb_idx: Sequence[int],
) -> float:
    """Compute the effective active space core energy for a given active space
    configuration. (for spatial orbital.)

    Args:
        original_core_energy:
            The orginal space core erengy.
        mo_1e_int:
            The orginal space spatial mo 1-electron integral.
        mo_2e_int:
            The orginal space spatial mo 2-electron integral.
        core_spatial_orb_idx:
            The core spatial orbital indices.
    """
    get_core_array_from_1e = np.ix_(core_spatial_orb_idx, core_spatial_orb_idx)
    get_core_array_from_2e = np.ix_(
        core_spatial_orb_idx,
        core_spatial_orb_idx,
        core_spatial_orb_idx,
        core_spatial_orb_idx,
    )
    mo_2e_int_core_subset = mo_2e_int[get_core_array_from_2e]

    delta_E = 0
    delta_E += 2 * trace(mo_1e_int[get_core_array_from_1e])
    delta_E += 2 * trace(trace(mo_2e_int_core_subset, axis1=0, axis2=3))
    delta_E -= 1 * trace(trace(mo_2e_int_core_subset, axis1=0, axis2=2))
    return original_core_energy + delta_E


def get_effective_active_space_1e_integrals(
    original_1e_integrals: npt.NDArray[np.complex128],
    mo_1e_int: npt.NDArray[np.complex128],
    mo_2e_int: npt.NDArray[np.complex128],
    core_spatial_orb_idx: Sequence[int],
) -> npt.NDArray[np.complex128]:
    """Compute the effective active space 1-electron integral for a given
    active space configuration. (for spatial orbital.)

    Args:
        original_core_energy:
            The orginal space core erengy.
        mo_1e_int:
            The orginal space spatial mo 1-electron integral.
        mo_2e_int:
            The orginal space spatial mo 2-electron integral.
        core_spatial_orb_idx:
            The core spatial orbital indices.
    """
    full_idx = arange(mo_1e_int.shape[0])
    get_core_array_from_2e_1 = np.ix_(
        cast(Any, core_spatial_orb_idx),
        full_idx,
        full_idx,
        cast(Any, core_spatial_orb_idx),
    )
    get_core_array_from_2e_2 = np.ix_(
        cast(Any, core_spatial_orb_idx),
        full_idx,
        cast(Any, core_spatial_orb_idx),
        full_idx,
    )

    original_1e_integrals_new = original_1e_integrals.copy()
    original_1e_integrals_new += 2 * trace(
        mo_2e_int[get_core_array_from_2e_1], axis1=0, axis2=3
    )
    original_1e_integrals_new -= 1 * trace(
        mo_2e_int[get_core_array_from_2e_2], axis1=0, axis2=2
    )

    return cast(npt.NDArray[np.complex128], original_1e_integrals_new)


def to_spin_orbital_integrals(
    n_spin_orb: int,
    spatial_1e_integrals: npt.NDArray[np.complex128],
    spatial_2e_integrals: npt.NDArray[np.complex128],
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Convert spatial orbital electron integrals to spin orbital electron
    integrals."""
    spin_1e_integrals = cast(
        npt.NDArray[np.complex128], zeros((n_spin_orb, n_spin_orb))
    )
    spin_2e_integrals = cast(
        npt.NDArray[np.complex128],
        zeros((n_spin_orb, n_spin_orb, n_spin_orb, n_spin_orb)),
    )

    for p, q in product(range(n_spin_orb // 2), repeat=2):
        p_a, q_a = 2 * p, 2 * q
        p_b, q_b = 2 * p + 1, 2 * q + 1

        spin_1e_integrals[p_a, q_a] = spatial_1e_integrals[p, q]
        spin_1e_integrals[p_b, q_b] = spatial_1e_integrals[p, q]

        for r, s in product(range(n_spin_orb // 2), repeat=2):
            r_a, s_a = 2 * r, 2 * s
            r_b, s_b = 2 * r + 1, 2 * s + 1
            # Mixed spin
            spin_2e_integrals[p_a, q_b, r_b, s_a] = spatial_2e_integrals[p, q, r, s]
            spin_2e_integrals[p_b, q_a, r_a, s_b] = spatial_2e_integrals[p, q, r, s]

            # Same spin
            spin_2e_integrals[p_a, q_a, r_a, s_a] = spatial_2e_integrals[p, q, r, s]
            spin_2e_integrals[p_b, q_b, r_b, s_b] = spatial_2e_integrals[p, q, r, s]

    return spin_1e_integrals, spin_2e_integrals


def spatial_mo_eint_set_to_spin_mo_eint_set(
    spatial_mo_eint_set: SpatialMOeIntSet,
) -> SpinMOeIntSet:
    """Compute the spin space mo_eint_set from spatial mo_eint_set."""
    nuc_energy = spatial_mo_eint_set.const
    spatial_mo_1e_int_array = spatial_mo_eint_set.mo_1e_int.array
    spatial_mo_2e_int_array = spatial_mo_eint_set.mo_2e_int.array
    n_spin_orb = 2 * spatial_mo_1e_int_array.shape[0]

    spin_mo_1e_int_array, spin_mo_2e_int_array = to_spin_orbital_integrals(
        n_spin_orb, spatial_mo_1e_int_array, spatial_mo_2e_int_array
    )

    return SpinMOeIntSet(
        const=nuc_energy,
        mo_1e_int=SpinMO1eInt(array=spin_mo_1e_int_array),
        mo_2e_int=SpinMO2eInt(array=spin_mo_2e_int_array),
    )


def get_active_space_integrals_from_mo(
    active_space_mo: ActiveSpaceMolecularOrbitals,
    electron_mo_ints: SpatialMOeIntSet,
    return_spin_integrals: bool = True,
) -> Union[SpinMOeIntSet, SpatialMOeIntSet]:
    """Compute the active space effective core energy and all the spin space
    electron integrals in the physicist's convention.

    Note:
    This function yields the active space electron integral faster than
    pyscf_get_active_space_integrals function in the quri_parts.pyscf package,
    as it stores the mo electron integrals on memory fot computations.
    """
    mo_1e_int = electron_mo_ints.mo_1e_int.array
    mo_2e_int = electron_mo_ints.mo_2e_int.array

    (
        core_spatial_orb_idx,
        active_spatial_orb_idx,
    ) = active_space_mo.get_core_and_active_orb()

    effective_core_energy = get_effective_active_space_core_energy(
        electron_mo_ints.const,
        mo_1e_int=mo_1e_int,
        mo_2e_int=mo_2e_int,
        core_spatial_orb_idx=core_spatial_orb_idx,
    )
    effective_mo_1e_int = get_effective_active_space_1e_integrals(
        mo_1e_int,
        mo_1e_int=mo_1e_int,
        mo_2e_int=mo_2e_int,
        core_spatial_orb_idx=core_spatial_orb_idx,
    )

    spatial_1e_integrals_subset = effective_mo_1e_int[
        np.ix_(active_spatial_orb_idx, active_spatial_orb_idx)
    ]
    spatial_2e_integrals_subset = mo_2e_int[
        np.ix_(
            active_spatial_orb_idx,
            active_spatial_orb_idx,
            active_spatial_orb_idx,
            active_spatial_orb_idx,
        )
    ]

    spatial_mo_eint_set = SpatialMOeIntSet(
        const=effective_core_energy,
        mo_1e_int=SpatialMO1eIntArray(array=spatial_1e_integrals_subset),
        mo_2e_int=SpatialMO2eIntArray(array=spatial_2e_integrals_subset),
    )

    if return_spin_integrals:
        return spatial_mo_eint_set_to_spin_mo_eint_set(spatial_mo_eint_set)

    return spatial_mo_eint_set


def get_active_space_integrals(
    active_space_mo: ActiveSpaceMolecularOrbitals,
    electron_ao_ints: AOeIntSet,
    return_spin_integrals: bool = True,
) -> Union[SpinMOeIntSet, SpatialMOeIntSet]:
    """Compute the active space electron integrals from ao electron
    integrals."""
    mo_coeff = active_space_mo.mo_coeff
    core_energy = electron_ao_ints.constant
    mo_1e_int = electron_ao_ints.ao_1e_int.to_mo1int(mo_coeff)
    mo_2e_int = electron_ao_ints.ao_2e_int.to_mo2int(mo_coeff)
    electron_mo_ints = SpatialMOeIntSet(
        const=core_energy, mo_1e_int=mo_1e_int, mo_2e_int=mo_2e_int
    )
    active_space_integrals = get_active_space_integrals_from_mo(
        active_space_mo=active_space_mo,
        electron_mo_ints=electron_mo_ints,
        return_spin_integrals=return_spin_integrals,
    )

    return active_space_integrals
