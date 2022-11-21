# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, TypeVar, Union

from quri_parts.core.estimator import (
    ConcurrentParametricQuantumEstimator,
    Estimatable,
    Estimates,
    GradientEstimator,
)
from quri_parts.core.state import (
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
)

_ParametricStateT = TypeVar(
    "_ParametricStateT",
    bound=Union[ParametricCircuitQuantumState, ParametricQuantumStateVector],
)


@dataclass
class _Estimates:
    values: Sequence[complex]
    error_matrix: Optional[Sequence[Sequence[float]]]


def numerical_gradient_estimates(
    op: Estimatable,
    state: _ParametricStateT,
    params: Sequence[float],
    estimator: ConcurrentParametricQuantumEstimator[_ParametricStateT],
    delta: float,
) -> Estimates[complex]:
    """Estimate a gradient of an expectation value of a given operator for a
    parametric state with respect to the state parameter by a numerical
    differentiation.

    The gradient estimates are configured with arguments as follows.

    Args:
        op: An operator of which expectation value is estimated.
        state: A parametric quantum state on which the operator
            expectation is evaluated.
        params: Parameter values for which the gradient is estimated.
        estimator: An estimator that estimates expectation values
            of the operator for the parametric states.
        delta: Step size for numerical differentiation.

    Returns:
        The estimated values (can be accessed with :attr:`.values`) with errors
        of estimation (can be accessed with :attr:`.error_matrix`). Currently,
        :attr:`.error_matrix` returns `None`.
    """

    v = []
    for i in range(len(params)):
        a = list(params)
        a[i] = params[i] + (delta * 0.5)
        v.append(a)
        a = list(params)
        a[i] = params[i] - (delta * 0.5)
        v.append(a)

    estimates = list(estimator(op, state, v))

    grad = []
    for i in range(len(params)):
        d = estimates[2 * i].value - estimates[2 * i + 1].value
        grad.append(d / delta)

    return _Estimates(grad, None)


def create_numerical_gradient_estimator(
    parametric_estimator: ConcurrentParametricQuantumEstimator[_ParametricStateT],
    delta: float,
) -> GradientEstimator[_ParametricStateT]:
    """Create a :class:`GradientEstimator` that estimates gradient values.

    The gradient estimates are configured with arguments as follows.

    Args:
        parametric_estimator: An estimator that estimates expectation values
            of the operator for the parametric states.
        delta: Step size for numerical differentiation.
    """

    def estimator(
        operator: Estimatable, state: _ParametricStateT, params: Sequence[float]
    ) -> Estimates[complex]:
        return numerical_gradient_estimates(
            operator,
            state,
            params,
            parametric_estimator,
            delta,
        )

    return estimator
