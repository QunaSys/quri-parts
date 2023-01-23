# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Sequence

from typing_extensions import TypeAlias

FirstOrderDifferenceFormula: TypeAlias = Callable[
    [Callable[[Sequence[float]], float], Sequence[float]],
    Sequence[float],
]
SecondOrderDifferenceFormula: TypeAlias = Callable[
    [Callable[[Sequence[float]], float], Sequence[float]],
    Sequence[Sequence[float]],
]


def create_forward_difference_formula_gradient(
    step: float = 1e-5,
) -> FirstOrderDifferenceFormula:
    """Create the function that returns the gradient of a passed function from
    `params` with two-point forward-difference formula."""

    def _diff_formula(
        f: Callable[[Sequence[float]], float],
        params: Sequence[float],
    ) -> Sequence[float]:
        dim = len(params)
        forward_points = [
            [params[k] + step * (k == i) for k in range(dim)] for i in range(dim)
        ]
        f_vals_forward = [f(forward) for forward in forward_points]
        f_val_orig = f(params)

        return [(f_val_fw - f_val_orig) / step for f_val_fw in f_vals_forward]

    return _diff_formula


def create_backward_difference_formula_gradient(
    step: float = 1e-5,
) -> FirstOrderDifferenceFormula:
    """Create the function that returns the gradient of a passed function from
    `params` with two-point backward-difference formula."""

    def _diff_formula(
        f: Callable[[Sequence[float]], float],
        params: Sequence[float],
    ) -> Sequence[float]:
        dim = len(params)
        backward_points = [
            [params[k] - step * (k == i) for k in range(dim)] for i in range(dim)
        ]
        f_vals_backward = [f(backward) for backward in backward_points]
        f_val_orig = f(params)

        return [(f_val_orig - f_val_bw) / step for f_val_bw in f_vals_backward]

    return _diff_formula


def create_central_difference_formula_gradient(
    step: float = 1e-5,
) -> FirstOrderDifferenceFormula:
    """Create the function that returns the gradient of a passed function from
    `params` with central-difference formula."""

    def _diff_formula(
        f: Callable[[Sequence[float]], float],
        params: Sequence[float],
    ) -> Sequence[float]:
        dim = len(params)
        forward_points = [
            [params[k] + step * (k == i) for k in range(dim)] for i in range(dim)
        ]
        backward_points = [
            [params[k] - step * (k == i) for k in range(dim)] for i in range(dim)
        ]
        f_vals_forward = [f(forward) for forward in forward_points]
        f_vals_backward = [f(backward) for backward in backward_points]

        return [
            (f_val_fw - f_val_bw) / (2 * step)
            for f_val_fw, f_val_bw in zip(f_vals_forward, f_vals_backward)
        ]

    return _diff_formula


def create_forward_difference_formula_hessian(
    step: float = 1e-5,
) -> SecondOrderDifferenceFormula:
    """Create the function that returns the hessian of a passed function from
    `params` with forward-difference formula."""

    def _diff_formula(
        f: Callable[[Sequence[float]], float],
        params: Sequence[float],
    ) -> Sequence[Sequence[float]]:
        dim = len(params)
        points_pp = [
            [
                [params[k] + step * (k == i) + step * (k == j) for k in range(dim)]
                for j in range(dim)
            ]
            for i in range(dim)
        ]
        points_p = [
            [params[k] + step * (k == i) for k in range(dim)] for i in range(dim)
        ]
        f_vals_pp = [[f(point_ij) for point_ij in points_i] for points_i in points_pp]
        f_vals_p = [f(point_i) for point_i in points_p]
        f_val_orig = f(params)

        ret = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            for j in range(dim):
                ret[i][j] = (
                    f_vals_pp[i][j] - f_vals_p[i] - f_vals_p[j] + f_val_orig
                ) / (step * step)

        return ret

    return _diff_formula


def create_backward_difference_formula_hessian(
    step: float = 1e-5,
) -> SecondOrderDifferenceFormula:
    """Create the function that returns the hessian of a passed function from
    `params` with backward-difference formula."""

    def _diff_formula(
        f: Callable[[Sequence[float]], float],
        params: Sequence[float],
    ) -> Sequence[Sequence[float]]:
        dim = len(params)
        points_mm = [
            [
                [params[k] - step * (k == i) - step * (k == j) for k in range(dim)]
                for j in range(dim)
            ]
            for i in range(dim)
        ]
        points_m = [
            [params[k] - step * (k == i) for k in range(dim)] for i in range(dim)
        ]
        f_vals_mm = [[f(point_ij) for point_ij in points_i] for points_i in points_mm]
        f_vals_m = [f(point_i) for point_i in points_m]
        f_val_orig = f(params)

        ret = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            for j in range(dim):
                ret[i][j] = (
                    f_val_orig - f_vals_m[i] - f_vals_m[j] + f_vals_mm[i][j]
                ) / (step * step)

        return ret

    return _diff_formula


def create_central_difference_formula_hessian(
    step: float = 1e-5,
) -> SecondOrderDifferenceFormula:
    """Create the function that returns the hessian of a passed function from
    `params` with central-difference formula."""

    def _diff_formula(
        f: Callable[[Sequence[float]], float],
        params: Sequence[float],
    ) -> Sequence[Sequence[float]]:
        dim = len(params)
        points_pp = [
            [
                [params[k] + step * (k == i) + step * (k == j) for k in range(dim)]
                for j in range(dim)
            ]
            for i in range(dim)
        ]
        points_pm = [
            [
                [params[k] + step * (k == i) - step * (k == j) for k in range(dim)]
                for j in range(dim)
            ]
            for i in range(dim)
        ]
        points_mp = [
            [
                [params[k] - step * (k == i) + step * (k == j) for k in range(dim)]
                for j in range(dim)
            ]
            for i in range(dim)
        ]
        points_mm = [
            [
                [params[k] - step * (k == i) - step * (k == j) for k in range(dim)]
                for j in range(dim)
            ]
            for i in range(dim)
        ]

        f_vals_pp = [[f(point_ij) for point_ij in points_i] for points_i in points_pp]
        f_vals_pm = [[f(point_ij) for point_ij in points_i] for points_i in points_pm]
        f_vals_mp = [[f(point_ij) for point_ij in points_i] for points_i in points_mp]
        f_vals_mm = [[f(point_ij) for point_ij in points_i] for points_i in points_mm]

        ret = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            for j in range(dim):
                ret[i][j] = (
                    f_vals_pp[i][j]
                    - f_vals_pm[i][j]
                    - f_vals_mp[i][j]
                    + f_vals_mm[i][j]
                ) / (4 * step * step)

        return ret

    return _diff_formula


def gradient(
    f: Callable[[Sequence[float]], float],
    params: Sequence[float],
    step: float = 1e-3,
    *,
    difference_formula: Optional[FirstOrderDifferenceFormula] = None,
) -> Sequence[float]:
    """Returns the gradient of a passed function from given `params`.

    Args:
        f: Function to find the gradient
        params: Parameters from which the gradient is computed
        step: Step size for numerical differentiation
        difference_formula: Difference formula (keyward-only)
    """

    if not difference_formula:
        difference_formula = create_central_difference_formula_gradient(step)
    result = difference_formula(f, params)
    if isinstance(result[0], list):
        raise ValueError(
            "Used incorrect formula for differentiation. Try using"
            "the formula designed for gradient."
        )
    return result


def hessian(
    f: Callable[[Sequence[float]], float],
    params: Sequence[float],
    step: float = 1e-3,
    *,
    difference_formula: Optional[SecondOrderDifferenceFormula] = None,
) -> Sequence[Sequence[float]]:
    """Returns the hessian of a passed function from given `params`.

    Args:
        f: Function to find the hessian
        params: Parameters from which the hessian is computed
        step: Step size for numerical differentiation
        difference_formula: Difference formula (keyward-only)
    """

    if not difference_formula:
        difference_formula = create_central_difference_formula_hessian(step)
    result = difference_formula(f, params)
    if isinstance(result[0], float):
        raise ValueError(
            "Used incorrect formula for differentiation. Try using"
            "the formula designed for hessian."
        )
    return result
