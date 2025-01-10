# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional

from quri_parts.algo.optimizer import Optimizer, OptimizerState, Params
from quri_parts.circuit import (
    LinearMappedUnboundParametricQuantumCircuit,
    NonParametricQuantumCircuit,
)

from quri_algo.algo.compiler.base_classes import QuantumCompilerGeneric
from quri_algo.circuit.interface import CircuitFactory
from quri_algo.core.cost_functions.base_classes import CostFunction


class QAQC(QuantumCompilerGeneric):
    """Quantum-Assisted Quantum Compilation (QAQC)

    This class provides an implementation of QAQC https://quantum-
    journal.org/papers/q-2019-05-13-140/. The optimize method returns a
    compiled circuit along with the converged state of the optimizer. It
    is also possible to use the __call__ method to return the compiled
    circuit only.
    """

    def __init__(self, cost_fn: CostFunction, optimizer: Optimizer):
        super().__init__(cost_fn, optimizer)

    @property
    def cost_function(self) -> CostFunction:
        return self._cost_fn

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    def optimize(
        self,
        circuit_factory: CircuitFactory,
        ansatz: LinearMappedUnboundParametricQuantumCircuit,
        init_params: Optional[Params] = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[NonParametricQuantumCircuit, OptimizerState]:
        """This function initiates the variational optimization.

        Arguments:
        circuit_factory - Circuit factory which generates the target circuit for the compilation
        ansatz - Parametric ansatz circuit that is used to approximate the target circuit
        init_params - initial variational parameters for the optimization

        variable arguments if any are passed to the circuit_factory's __call__ method

        Return value:
        Tuple of compiled circuit and the state of the optimizer
        """
        return self._optimize(circuit_factory, ansatz, init_params, *args, **kwargs)
