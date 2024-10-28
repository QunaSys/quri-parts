# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union

from quri_parts.core.estimator import Estimate
from quri_parts.core.sampling import Sampler, StateSampler

from quri_algo.circuit.hadamard_test import HadamardTestCircuitFactory
from quri_algo.circuit.time_evolution.interface import (
    ControlledTimeEvolutionCircuitFactory,
)
from quri_algo.core.estimator import (
    ExpectationValueEstimator,
    OperatorPowerEstimatorBase,
    StateT,
)
from quri_algo.core.estimator.hadamard_test import HadamardTest
from quri_algo.problem import HamiltonianT


@dataclass
class TimeEvolutionExpectationValueEstimator(
    ExpectationValueEstimator[HamiltonianT, StateT], ABC
):
    r"""Estimator that computes :math:`\langle e^{-iHt} \rangle`"""

    @abstractmethod
    def __call__(
        self, state: StateT, evolution_time: float, n_shots: Optional[int] = None
    ) -> Estimate[complex]:
        ...


@dataclass
class TimeEvolutionHadamardTest(
    TimeEvolutionExpectationValueEstimator[HamiltonianT, StateT],
):
    r"""Computes :math:`\langle e^{-iHt} \rangle` with Hadamard test.

    This object can be instantiated or inherited for performing the
    estimation with Hadamard test for arbitrary implementations of the
    time evolution operator.
    """

    controlled_time_evolution_factory: ControlledTimeEvolutionCircuitFactory[
        HamiltonianT
    ]
    sampler: Union[Sampler, StateSampler[StateT]]

    def __post_init__(self) -> None:
        self._hadamard_test = HadamardTest(
            self.encoded_problem,
            self.controlled_time_evolution_factory,
            self.sampler,
            transpiler=self.transpiler,
        )

    @property
    def real_circuit_factory(self) -> HadamardTestCircuitFactory[HamiltonianT]:
        return self._hadamard_test.real_circuit_factory

    @property
    def imag_circuit_factory(self) -> HadamardTestCircuitFactory[HamiltonianT]:
        return self._hadamard_test.imag_circuit_factory

    def __call__(
        self, state: StateT, evolution_time: float, n_shots: Optional[int] = None
    ) -> Estimate[complex]:
        assert n_shots is not None
        return self._hadamard_test(state, n_shots, evolution_time)


@dataclass
class TimeEvolutionPowerEstimator(OperatorPowerEstimatorBase[HamiltonianT, StateT]):
    r"""Turns a :class:`TimeEvolutionExpectationValueEstimator` for evaluating
    :math:`e^{-iH k \tau}` into a  :class:`OperatorPowerEstimatorBase` so that
    :math:`e^{-iH\tau}` is treated as the unitary operator and :math:`k` as the power.
    """

    encoded_problem: HamiltonianT = field(init=False)
    time_evo_estimator: TimeEvolutionExpectationValueEstimator[HamiltonianT, StateT]
    tau: float

    def __post_init__(self) -> None:
        self.encoded_problem = self.time_evo_estimator.encoded_problem

    def __call__(
        self, state: StateT, operator_power: int | float, n_shots: Optional[int] = None
    ) -> Estimate[complex]:
        return self.time_evo_estimator(state, operator_power * self.tau, n_shots)
