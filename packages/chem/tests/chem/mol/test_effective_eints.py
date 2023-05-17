from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy import array
from openfermion.chem.molecular_data import spinorb_from_spatial

from quri_parts.chem.mol import (
    ActiveSpaceMolecularOrbitals,
    AO1eIntArray,
    AO2eIntArray,
    AOeIntArraySet,
    MolecularOrbitals,
    SpatialMO1eIntArray,
    SpatialMO2eIntArray,
    cas,
    get_effective_active_space_1e_integrals,
    get_effective_active_space_2e_integrals,
    get_effective_active_space_core_energy,
    to_spin_orbital_integrals,
)


@dataclass
class MolecularOrbitalsInfo(MolecularOrbitals):
    _n_electron: int
    _n_spatial_orb: int
    _spin: int = 0
    _mo_coeff: npt.NDArray[np.complex128] = np.zeros((1,), dtype=np.complex128)

    @property
    def n_electron(self) -> int:
        return self._n_electron

    @property
    def spin(self) -> int:
        return self._spin

    @property
    def n_spatial_orb(self) -> int:
        return self._n_spatial_orb

    @property
    def mo_coeff(self) -> "npt.NDArray[np.complex128]":
        return self._mo_coeff


# Hard coded full space spatial electron integrals of a spin up H3 chain

mo_coeff = array(
    [
        [0.36752998, 0.75344304, -0.81446919],
        [0.5465483, 0.0, 1.22005563],
        [0.36752998, -0.75344304, -0.81446919],
    ]
)

core_energy = 1.3229430273

ao_1e_int = AO1eIntArray(
    array(
        [
            [-1.24398564, -0.86080698, -0.22631877],
            [-0.86080698, -1.4924109, -0.86080698],
            [-0.22631877, -0.86080698, -1.24398564],
        ]
    )
)

ao_2e_int = AO2eIntArray(
    array(
        [
            [
                [
                    [0.77460594, 0.30930897, 0.05584872],
                    [0.30930897, 0.15786578, 0.03152398],
                    [0.05584872, 0.03152398, 0.00732843],
                ],
                [
                    [0.30930897, 0.15786578, 0.03152398],
                    [0.47804137, 0.30930897, 0.07065588],
                    [0.17348403, 0.11521591, 0.03152398],
                ],
                [
                    [0.05584872, 0.03152398, 0.00732843],
                    [0.17348403, 0.11521591, 0.03152398],
                    [0.26369505, 0.17348403, 0.05584872],
                ],
            ],
            [
                [
                    [0.30930897, 0.47804137, 0.17348403],
                    [0.15786578, 0.30930897, 0.11521591],
                    [0.03152398, 0.07065588, 0.03152398],
                ],
                [
                    [0.15786578, 0.30930897, 0.11521591],
                    [0.30930897, 0.77460594, 0.30930897],
                    [0.11521591, 0.30930897, 0.15786578],
                ],
                [
                    [0.03152398, 0.07065588, 0.03152398],
                    [0.11521591, 0.30930897, 0.15786578],
                    [0.17348403, 0.47804137, 0.30930897],
                ],
            ],
            [
                [
                    [0.05584872, 0.17348403, 0.26369505],
                    [0.03152398, 0.11521591, 0.17348403],
                    [0.00732843, 0.03152398, 0.05584872],
                ],
                [
                    [0.03152398, 0.11521591, 0.17348403],
                    [0.07065588, 0.30930897, 0.47804137],
                    [0.03152398, 0.15786578, 0.30930897],
                ],
                [
                    [0.00732843, 0.03152398, 0.05584872],
                    [0.03152398, 0.15786578, 0.30930897],
                    [0.05584872, 0.30930897, 0.77460594],
                ],
            ],
        ]
    )
)

mo_1e_int = SpatialMO1eIntArray(
    array(
        [
            [-1.53466912, -0.0, -0.12053411],
            [0.0, -1.15541096, -0.0],
            [-0.12053411, -0.0, -0.75066168],
        ]
    )
)

mo_2e_int = SpatialMO2eIntArray(
    array(
        [
            [
                [
                    [0.55936827, -0.0, 0.08803043],
                    [-0.0, 0.15477155, 0.0],
                    [0.08803043, 0.0, 0.13563321],
                ],
                [
                    [-0.0, 0.15477155, 0.0],
                    [0.48660662, -0.0, -0.0379191],
                    [0.0, -0.14084555, -0.0],
                ],
                [
                    [0.08803043, 0.0, 0.13563321],
                    [0.0, -0.14084555, -0.0],
                    [0.57575727, -0.0, 0.08987634],
                ],
            ],
            [
                [
                    [-0.0, 0.48660662, 0.0],
                    [0.15477155, -0.0, -0.14084555],
                    [0.0, -0.0379191, -0.0],
                ],
                [
                    [0.15477155, -0.0, -0.14084555],
                    [-0.0, 0.53466412, 0.0],
                    [-0.14084555, -0.0, 0.15039551],
                ],
                [
                    [0.0, -0.0379191, -0.0],
                    [-0.14084555, -0.0, 0.15039551],
                    [-0.0, 0.5123315, 0.0],
                ],
            ],
            [
                [
                    [0.08803043, 0.0, 0.57575727],
                    [0.0, -0.14084555, -0.0],
                    [0.13563321, -0.0, 0.08987634],
                ],
                [
                    [0.0, -0.14084555, -0.0],
                    [-0.0379191, 0.0, 0.5123315],
                    [-0.0, 0.15039551, 0.0],
                ],
                [
                    [0.13563321, -0.0, 0.08987634],
                    [-0.0, 0.15039551, 0.0],
                    [0.08987634, 0.0, 0.62546613],
                ],
            ],
        ]
    )
)


ao_eint_set = AOeIntArraySet(
    constant=core_energy, ao_1e_int=ao_1e_int, ao_2e_int=ao_2e_int
)

active_space_mo = ActiveSpaceMolecularOrbitals(
    MolecularOrbitalsInfo(_n_electron=3, _n_spatial_orb=3, _spin=1, _mo_coeff=mo_coeff),
    cas(1, 1),
)


def test_spatial_to_spin_conversion() -> None:
    of_full_space_spin_integrals = spinorb_from_spatial(
        ao_1e_int.array, ao_2e_int.array
    )

    full_space_spin_integrals = to_spin_orbital_integrals(
        2 * ao_1e_int.array.shape[0], ao_1e_int.array, ao_2e_int.array
    )

    assert np.allclose(of_full_space_spin_integrals[0], full_space_spin_integrals[0])
    assert np.allclose(of_full_space_spin_integrals[1], full_space_spin_integrals[1])


def test_test_eff_nuc_energy() -> None:
    # openfermion effective core energy for spin up H3.
    of_effective_nuc_energy = -1.1870269447600394

    effective_nuc_energy = get_effective_active_space_core_energy(
        core_energy=core_energy,
        mo_1e_int=mo_1e_int.array,
        mo_2e_int=mo_2e_int.array,
        core_spatial_orb_idx=[0],
    )

    assert np.isclose(of_effective_nuc_energy, effective_nuc_energy)


def test_test_eff_1e_integrals() -> None:
    # openfermion effective 1e electron integral for spin up H3.
    of_effective_1e_integrals = array([[-0.33696926]])

    effective_1e_integrals = get_effective_active_space_1e_integrals(
        mo_1e_int=mo_1e_int.array,
        mo_2e_int=mo_2e_int.array,
        core_spatial_orb_idx=[0],
        active_spatial_orb_idx=[1],
    )

    assert np.allclose(of_effective_1e_integrals, effective_1e_integrals)


def test_test_eff_2e_integrals() -> None:
    # openfermion effective 2e electron integral for spin up H3.
    of_effective_2e_integrals = array([[[[0.53466412]]]])

    effective_2e_integrals = get_effective_active_space_2e_integrals(
        mo_2e_int=mo_2e_int.array,
        active_spatial_orb_idx=[1],
    )

    assert np.allclose(of_effective_2e_integrals, effective_2e_integrals)
