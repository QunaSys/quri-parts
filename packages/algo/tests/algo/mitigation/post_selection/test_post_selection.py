# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from quri_parts.algo.mitigation.post_selection.post_selection import (
    _inv_bk_trans_mat,
    create_bk_electron_number_post_selection_sampler,
    create_general_post_selection_sampler,
    create_jw_electron_number_post_selection_sampler,
    post_selection,
)
from quri_parts.circuit import NonParametricQuantumCircuit, QuantumCircuit
from quri_parts.core.sampling import MeasurementCounts


def test_post_selection() -> None:
    def _filter_fn(bits: int) -> bool:
        if bits >> 2 & 1:
            return True
        return False

    meas_counts = {
        0b000: 1,
        0b001: 2,
        0b010: 3,
        0b011: 4,
        0b100: 5,
        0b101: 6,
        0b110: 7,
        0b111: 8,
    }

    assert post_selection(_filter_fn, meas_counts) == {
        0b100: 5,
        0b101: 6,
        0b110: 7,
        0b111: 8,
    }


def _mock_sampler(_: NonParametricQuantumCircuit, __: int) -> MeasurementCounts:
    return {0b01: 1, 0b10: 10, 0b111: 20, 0b0101: 5, 0b1110: 100}


def test_create_general_post_selection_sampler() -> None:
    circuit = QuantumCircuit(2)

    def _filter_fn(bits: int) -> bool:
        if bits < 3:
            return True
        return False

    ps_sampler = create_general_post_selection_sampler(_mock_sampler, _filter_fn)
    assert ps_sampler(circuit, 100) == {0b01: 1, 0b10: 10}


def test_create_jw_electron_number_post_selection_sampler() -> None:
    qubit_count = 5
    n_electrons = 3
    circuit = QuantumCircuit(qubit_count)
    ps_sampler = create_jw_electron_number_post_selection_sampler(
        _mock_sampler, n_electrons
    )
    assert ps_sampler(circuit, 100) == {0b111: 20, 0b1110: 100}

    ps_sampler_sz05 = create_jw_electron_number_post_selection_sampler(
        _mock_sampler, n_electrons, sz=0.5
    )
    assert ps_sampler_sz05(circuit, 100) == {0b111: 20}


def test_create_bk_electron_number_post_selection_sampler() -> None:
    qubit_count = 10
    n_electrons = 2
    circuit = QuantumCircuit(qubit_count)
    ps_sampler = create_bk_electron_number_post_selection_sampler(
        _mock_sampler, qubit_count, n_electrons
    )
    assert ps_sampler(circuit, 100) == {0b01: 1, 0b10: 10, 0b111: 20}

    ps_sampler_sz1 = create_bk_electron_number_post_selection_sampler(
        _mock_sampler, qubit_count, n_electrons, sz=1.0
    )
    assert ps_sampler_sz1(circuit, 100) == {0b111: 20}


def test_inv_bk_trans_mat() -> None:
    assert np.array_equal(_inv_bk_trans_mat(0), np.array([1]))
    assert np.array_equal(_inv_bk_trans_mat(1), np.array([[1, 0], [1, 1]]))
    assert np.array_equal(
        _inv_bk_trans_mat(2),
        np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 1, 1, 1]]),
    )
    assert np.array_equal(
        _inv_bk_trans_mat(3),
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 1, 1, 1],
            ]
        ),
    )
    assert np.array_equal(
        _inv_bk_trans_mat(4),
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
            ]
        ),
    )
