# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from quri_algo.circuit.time_evolution.exact_unitary import (
    ExactUnitaryControlledTimeEvolutionCircuitFactory,
)
from quri_algo.core.estimator import State
from quri_algo.problem import QubitHamiltonianInput

from .interface import TimeEvolutionHadamardTest


@dataclass
class ExactUnitaryTimeEvolutionHadamardTest(
    TimeEvolutionHadamardTest[QubitHamiltonianInput, State]
):
    r"""Performs :math:`\langle e^{-iHt} \rangle` base on exact unitary matrix
    implementation of the time evolution operator :math:`e^{-iHt}`."""

    controlled_time_evolution_factory: (
        ExactUnitaryControlledTimeEvolutionCircuitFactory
    ) = field(init=False)

    def __post_init__(self) -> None:
        self.controlled_time_evolution_factory = (
            ExactUnitaryControlledTimeEvolutionCircuitFactory(
                self.encoded_problem, transpiler=self.transpiler
            )
        )
        super().__post_init__()
