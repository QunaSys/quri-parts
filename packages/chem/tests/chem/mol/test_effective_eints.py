import unittest
from dataclasses import dataclass
from typing import cast

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
    SpinMOeIntSet,
    cas,
    get_active_space_integrals_from_ao_eint,
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

mo_coeff = array(
    [
        [0.36752998, 0.75344304, -0.81446919],
        [0.5465483, 0.0, 1.22005563],
        [0.36752998, -0.75344304, -0.81446919],
    ]
)

full_space_spin_integrals = to_spin_orbital_integrals(
    2 * ao_1e_int.array.shape[0], ao_1e_int.array, ao_2e_int.array
)

ao_eint_set = AOeIntArraySet(
    constant=core_energy, ao_1e_int=ao_1e_int, ao_2e_int=ao_2e_int
)


active_space_mo = ActiveSpaceMolecularOrbitals(
    MolecularOrbitalsInfo(_n_electron=3, _n_spatial_orb=3, _spin=1, _mo_coeff=mo_coeff),
    cas(1, 1),
)


class TestActiveSpaceIntegrals(unittest.TestCase):
    _nuc_energy: float
    _spin_mo_1e_int: npt.NDArray[np.complex128]
    _spin_mo_2e_int: npt.NDArray[np.complex128]
    _full_space_spin_integrals: npt.NDArray[np.complex128]
    qp_chem_active_space_integrals: SpinMOeIntSet

    @classmethod
    def setUpClass(cls) -> None:
        cls.qp_chem_active_space_integrals = cast(
            SpinMOeIntSet,
            get_active_space_integrals_from_ao_eint(
                active_space_mo=active_space_mo, electron_ao_ints=ao_eint_set
            ),
        )

        # openfermion active space result.
        # _spin_mo_2e_int is multiplied by a factor of 2 to be consistent with
        # qp convention
        cls._nuc_energy = -1.1870269447600394
        cls._spin_mo_1e_int = array([[-0.33696926, 0.0], [0.0, -0.33696926]])
        cls._spin_mo_2e_int = array(
            [
                [[[0.53466412, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.53466412, 0.0]]],
                [[[0.0, 0.53466412], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.53466412]]],
            ]
        )
        cls._full_space_spin_integrals = spinorb_from_spatial(
            ao_1e_int.array, ao_2e_int.array
        )

    def test_spatial_to_spin_conversion(self) -> None:
        assert np.allclose(
            self._full_space_spin_integrals[0], full_space_spin_integrals[0]
        )
        assert np.allclose(
            self._full_space_spin_integrals[1], full_space_spin_integrals[1]
        )

    def test_eff_nuc_energy(self) -> None:
        assert np.allclose(self.qp_chem_active_space_integrals.const, self._nuc_energy)

    def test_eff_h1(self) -> None:
        assert np.allclose(
            self.qp_chem_active_space_integrals.mo_1e_int.array, self._spin_mo_1e_int
        )

    def test_eff_h2(self) -> None:
        assert np.allclose(
            self.qp_chem_active_space_integrals.mo_2e_int.array, self._spin_mo_2e_int
        )
