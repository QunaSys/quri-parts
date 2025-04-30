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
from typing import Any, Optional, Sequence

from quri_parts.algo.optimizer import OptimizerState
from quri_parts.circuit import NonParametricQuantumCircuit

from quri_algo.algo.interface import (
    Algorithm,
    AlgorithmResult,
    QuantumAlgorithm,
    VariationalAlgorithmResultMixin,
)
from quri_algo.algo.utils.timer import timer
from quri_algo.algo.utils.variational_solvers import VariationalSolver
from quri_algo.core.cost_functions.base_classes import CostFunction


class CompilationResult(AlgorithmResult, VariationalAlgorithmResultMixin):
    def __init__(
        self,
        algorithm: Algorithm,
        optimized_circuit: NonParametricQuantumCircuit,
        optimizer_history: Sequence[OptimizerState],
        elapsed_time: Optional[float] = None,
    ):
        AlgorithmResult.__init__(self, algorithm, elapsed_time)
        VariationalAlgorithmResultMixin.__init__(self, optimizer_history)
        self.optimized_circuit = optimized_circuit


class QuantumCompiler(QuantumAlgorithm, ABC):
    @property
    @abstractmethod
    def cost_function(self) -> CostFunction:
        ...

    @property
    @abstractmethod
    def solver(self) -> VariationalSolver:
        ...

    @timer
    @abstractmethod
    def run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> CompilationResult:
        pass
