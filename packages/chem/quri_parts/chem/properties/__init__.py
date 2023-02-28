# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Sequence, TypeVar, Union

from typing_extensions import TypeAlias

from quri_parts.core.estimator import ConcurrentParametricQuantumEstimator
from quri_parts.core.operator import Operator
from quri_parts.core.state import (
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
)
from quri_parts.core.utils.differentiation import (
    NumericalOperatorGradientCalculator,
    create_numerical_operator_gradient_calculator,
)

#: A type variable represents *any* parametric quantum state classes.
#: This is different from :class:`quri_parts.core.state.ParametricQuantumStateT`;
#: ``ParametricQuantumStateT`` represents *either one of* the classes,
#: while ``_ParametricStateT`` also covers *a union of* multiple state classes.
_ParametricStateT = TypeVar(
    "_ParametricStateT",
    bound=Union[ParametricCircuitQuantumState, ParametricQuantumStateVector],
)

#: EnergyGradientEstimator represents a function that estimates the energy gradient
#: of a given parametric state w.r.t. the Hamiltonian parameters at given circuit
#: parameters.
#: This is a generic type and you need to specify what kind of state classes
#: it is applicable to.
EnergyGradientEstimator: TypeAlias = Callable[
    [_ParametricStateT, Sequence[float]], Sequence[float]
]


def create_energy_gradient_estimator(
    estimator: ConcurrentParametricQuantumEstimator[_ParametricStateT],
    h_params: Sequence[float],
    h_generator: Callable[[Sequence[float]], Operator],
    h_gradient_calculator: Optional[NumericalOperatorGradientCalculator] = None,
) -> EnergyGradientEstimator[_ParametricStateT]:
    """Create a :class:`EnergyGradientEstimator` that calculates the energy
    gradients with respect to the hamiltonian parameters at the given circuit
    parameters."""
    if h_gradient_calculator is None:
        h_gradient_calculator = create_numerical_operator_gradient_calculator(
            h_generator
        )
    h_grad = h_gradient_calculator(h_params)

    # EnergyGradientEstimator
    def energy_gradient_estimator(
        parametric_state: _ParametricStateT, params: Sequence[float]
    ) -> Sequence[float]:
        return [
            list(estimator(op, parametric_state, [params]))[0].value.real
            for op in h_grad
        ]

    return energy_gradient_estimator
