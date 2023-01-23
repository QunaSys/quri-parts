# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence

import numpy as np
import pytest

from quri_parts.core.utils.differentiation import (
    create_backward_difference_formula_gradient,
    create_backward_difference_formula_hessian,
    create_central_difference_formula_gradient,
    create_central_difference_formula_hessian,
    create_forward_difference_formula_gradient,
    create_forward_difference_formula_hessian,
    gradient,
    hessian,
)


def f(params: Sequence[float]) -> float:
    return (
        params[0] * params[0] * params[0] * 0.1
        - params[1] * params[1] * params[1] * 0.2
        + params[2] * params[2] * params[2] * 0.3
        + params[0] * params[0] * params[1] * 0.4
        - params[0] * params[1] * params[1] * 0.5
        + params[1] * params[1] * params[2] * 0.6
        - params[0] * params[2] * params[2] * 0.7
    )


def test_create_forward_differene_formula_gradient() -> None:
    params = [1, 2, 3]
    diff_formula = create_forward_difference_formula_gradient()
    expected = [-6.4, 3.2, 6.3]
    assert np.allclose(diff_formula(f, params), expected)


def test_create_backward_differene_formula_gradient() -> None:
    params = [1, 2, 3]
    diff_formula = create_backward_difference_formula_gradient()
    expected = [-6.4, 3.2, 6.3]
    assert np.allclose(diff_formula(f, params), expected)


def test_create_central_differene_formula_gradient() -> None:
    params = [1, 2, 3]
    diff_formula = create_central_difference_formula_gradient()
    expected = [-6.4, 3.2, 6.3]
    assert np.allclose(diff_formula(f, params), expected)


def test_create_forward_differene_formula_hessian() -> None:
    params = [1, 2, 3]
    diff_formula = create_forward_difference_formula_hessian()
    expected = [[2.2, -1.2, -4.2], [-1.2, 0.2, 2.4], [-4.2, 2.4, 4.0]]
    assert np.allclose(diff_formula(f, params), expected, rtol=1e-3)


def test_create_backward_differene_formula_hessian() -> None:
    params = [1, 2, 3]
    diff_formula = create_backward_difference_formula_hessian()
    expected = [[2.2, -1.2, -4.2], [-1.2, 0.2, 2.4], [-4.2, 2.4, 4.0]]
    assert np.allclose(diff_formula(f, params), expected, rtol=1e-3)


def test_create_2nd_order_central_differene_formula() -> None:
    params = [1, 2, 3]
    diff_formula = create_central_difference_formula_hessian()
    expected = [[2.2, -1.2, -4.2], [-1.2, 0.2, 2.4], [-4.2, 2.4, 4.0]]
    assert np.allclose(diff_formula(f, params), expected, rtol=1e-3)


def test_gradient() -> None:
    params = [1, 2, 3]
    step = 1e-5
    expected = [-6.4, 3.2, 6.3]
    assert np.allclose(gradient(f, params, step=step), expected)

    fdf = create_forward_difference_formula_gradient(step)
    bdf = create_backward_difference_formula_gradient(step)
    cdf = create_central_difference_formula_gradient(step)

    fdf_res = gradient(f, params, step=step, difference_formula=fdf)
    bdf_res = gradient(f, params, step=step, difference_formula=bdf)
    cdf_res = gradient(f, params, step=step, difference_formula=cdf)

    assert fdf_res != bdf_res
    assert np.allclose(fdf_res, bdf_res)
    assert fdf_res != cdf_res
    assert np.allclose(fdf_res, bdf_res)
    assert bdf_res != cdf_res


def test_gradient_invalid_formulas() -> None:
    params = [1, 2, 3]

    fdf_hessian = create_forward_difference_formula_hessian()
    bdf_hessian = create_backward_difference_formula_hessian()
    cdf_hessian = create_central_difference_formula_hessian()

    with pytest.raises(ValueError):
        gradient(f, params, difference_formula=fdf_hessian)  # type: ignore

    with pytest.raises(ValueError):
        gradient(f, params, difference_formula=bdf_hessian)  # type: ignore

    with pytest.raises(ValueError):
        gradient(f, params, difference_formula=cdf_hessian)  # type: ignore


def test_hessian() -> None:
    params = [1, 2, 3]
    step = 1e-5
    expected = [[2.2, -1.2, -4.2], [-1.2, 0.2, 2.4], [-4.2, 2.4, 4.0]]
    assert np.allclose(hessian(f, params, step=1e-3), expected)

    fdf = create_forward_difference_formula_hessian(step)
    bdf = create_backward_difference_formula_hessian(step)
    cdf = create_central_difference_formula_hessian(step)

    fdf_res = hessian(f, params, step=step, difference_formula=fdf)
    bdf_res = hessian(f, params, step=step, difference_formula=bdf)
    cdf_res = hessian(f, params, step=step, difference_formula=cdf)

    assert fdf_res != bdf_res
    assert np.allclose(fdf_res, bdf_res, rtol=1e-3)
    assert fdf_res != cdf_res
    assert np.allclose(fdf_res, bdf_res, rtol=1e-3)
    assert bdf_res != cdf_res


def test_hessian_invalid_formulas() -> None:
    params = [1, 2, 3]

    fdf_gradient = create_forward_difference_formula_gradient()
    bdf_gradient = create_backward_difference_formula_gradient()
    cdf_gradient = create_central_difference_formula_gradient()

    with pytest.raises(ValueError):
        hessian(f, params, difference_formula=fdf_gradient)  # type: ignore

    with pytest.raises(ValueError):
        hessian(f, params, difference_formula=bdf_gradient)  # type: ignore

    with pytest.raises(ValueError):
        hessian(f, params, difference_formula=cdf_gradient)  # type: ignore
