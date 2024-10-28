# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import Mock

import numpy as np
from quri_parts.algo.optimizer import OptimizerStatus

from quri_algo.algo.compiler.qaqc import QAQC
from quri_algo.circuit.interface import CircuitFactory
from quri_algo.core.cost_functions.base_classes import LocalCostFunction


def test_qaqc() -> None:
    cost_fn = Mock(spec=LocalCostFunction)
    optimizer = Mock()
    ansatz = Mock()
    circuit_factory = Mock(spec=CircuitFactory)
    circuit_factory.qubit_count = 4

    evolution_time = 0.5
    opt_params = np.array([0.5, 0.1, 0.3])
    ansatz.parameter_count = 16
    optimizer_state_success = Mock()
    optimizer_state_success.status = OptimizerStatus.SUCCESS
    optimizer_state_converged = Mock()
    optimizer_state_converged.status = OptimizerStatus.CONVERGED
    optimizer_state_converged.params = opt_params
    optimizer.get_init_state = Mock(return_value=optimizer_state_success)
    optimizer.step = Mock(return_value=optimizer_state_converged)

    qaqc = QAQC(cost_fn, optimizer)
    qaqc(circuit_factory, ansatz, evolution_time=evolution_time)

    assert optimizer.get_init_state.call_count == 1
    assert optimizer.step.call_count == 1
    assert ansatz.bind_parameters.called_with(opt_params)
