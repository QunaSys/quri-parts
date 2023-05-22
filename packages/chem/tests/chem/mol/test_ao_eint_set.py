# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

spatial_mo_1e_int = SpatialMO1eIntArray(
    array(
        [
            [-1.53466912, -0.0, -0.12053411],
            [0.0, -1.15541096, -0.0],
            [-0.12053411, -0.0, -0.75066168],
        ]
    )
)

spatial_mo_2e_int = SpatialMO2eIntArray(
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

mo = MolecularOrbitalsInfo(_n_electron=3, _n_spatial_orb=3, _spin=1, _mo_coeff=mo_coeff)
active_space_mo = ActiveSpaceMolecularOrbitals(
    mo,
    cas(1, 1),
)


def test_aoeint_set_to_full_space_spatial_mo_int() -> None:
    spatial_mo_int = ao_eint_set.to_full_space_spatial_mo_int(mo)
    computed_nuc_energy = spatial_mo_int.const
    computed_mo_1e_int = spatial_mo_int.mo_1e_int.array
    computed_mo_2e_int = spatial_mo_int.mo_2e_int.array

    assert np.isclose(computed_nuc_energy, core_energy)
    assert np.allclose(computed_mo_1e_int, spatial_mo_1e_int.array)
    assert np.allclose(computed_mo_2e_int, spatial_mo_2e_int.array)


def test_aoeint_set_to_full_space_spin_mo_int() -> None:
    spin_mo_int = ao_eint_set.to_full_space_mo_int(mo)
    computed_nuc_energy = spin_mo_int.const
    computed_mo_1e_int = spin_mo_int.mo_1e_int.array
    computed_mo_2e_int = spin_mo_int.mo_2e_int.array

    (expected_spin_mo_1eint, expected_spin_mo_2eint) = spinorb_from_spatial(
        spatial_mo_1e_int.array, spatial_mo_2e_int.array
    )

    assert np.isclose(computed_nuc_energy, core_energy)
    assert np.allclose(computed_mo_1e_int, expected_spin_mo_1eint)
    assert np.allclose(computed_mo_2e_int, expected_spin_mo_2eint)


def test_aoeint_set_to_active_space_spatial_mo_int() -> None:
    spatial_mo_int = ao_eint_set.to_active_space_spatial_mo_int(active_space_mo)
    computed_nuc_energy = spatial_mo_int.const
    computed_mo_1e_int = spatial_mo_int.mo_1e_int.array
    computed_mo_2e_int = spatial_mo_int.mo_2e_int.array

    expected_as_nuc_energy = -1.1870269447600394
    expected_as_1e_integrals = array([[-0.33696926]])
    expected_as_2e_integrals = array([[[[0.53466412]]]])

    assert np.isclose(computed_nuc_energy, expected_as_nuc_energy)
    assert np.allclose(computed_mo_1e_int, expected_as_1e_integrals)
    assert np.allclose(computed_mo_2e_int, expected_as_2e_integrals)


def test_aoeint_set_to_active_space_spin_mo_int() -> None:
    spatial_mo_int = ao_eint_set.to_active_space_mo_int(active_space_mo)
    computed_nuc_energy = spatial_mo_int.const
    computed_mo_1e_int = spatial_mo_int.mo_1e_int.array
    computed_mo_2e_int = spatial_mo_int.mo_2e_int.array

    expected_as_nuc_energy = -1.1870269447600394
    expected_as_1e_integrals = array([[-0.33696926, 0.0], [0.0, -0.33696926]])

    # This array is generated by openfermionpyscf's
    # PyscfMolecularData.get_molecular_hamiltonian().n_body_tensors,
    # It divides the spin electron integral by a factor of 2,
    # so I multiply it back to make the convention consistent with
    # quri-parts'.
    expected_as_2e_integrals = 2 * array(
        [
            [[[0.26733206, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.26733206, 0.0]]],
            [[[0.0, 0.26733206], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.26733206]]],
        ]
    )

    assert np.isclose(computed_nuc_energy, expected_as_nuc_energy)
    assert np.allclose(computed_mo_1e_int, expected_as_1e_integrals)
    assert np.allclose(computed_mo_2e_int, expected_as_2e_integrals)
