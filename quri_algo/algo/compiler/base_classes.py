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
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.random import default_rng
from quri_parts.algo.optimizer import Optimizer, OptimizerState, OptimizerStatus, Params
from quri_parts.circuit import (
    LinearMappedUnboundParametricQuantumCircuit,
    NonParametricQuantumCircuit,
)
from quri_parts.circuit.parameter_shift import ShiftedParameters

from quri_algo.circuit.interface import CircuitFactory
from quri_algo.core.cost_functions.base_classes import CostFunction


@dataclass
class QuantumCompiler(CircuitFactory, ABC):
    cost_fn: CostFunction
    optimizer: Optimizer

    @abstractmethod
    def optimize(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[NonParametricQuantumCircuit, OptimizerState]:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> NonParametricQuantumCircuit:
        pass


@dataclass
class QuantumCompilerGeneric(QuantumCompiler, ABC):
    def _optimize(
        self,
        circuit_factory: CircuitFactory,
        ansatz: LinearMappedUnboundParametricQuantumCircuit,
        init_params: Optional[Params] = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[NonParametricQuantumCircuit, OptimizerState]:
        if not isinstance(circuit_factory, CircuitFactory):
            raise TypeError(
                "Target circuit factory should be an instance of CircuitFactory."
            )
        target_circuit = circuit_factory(*args, **kwargs)

        def cost(params: Params) -> float:
            trial_circuit = ansatz.bind_parameters(list(params.tolist()))
            return self.cost_fn(target_circuit, trial_circuit).value.real

        def gradient(params: Params) -> Params:
            """Parameter shift gradient function.

            This calculates the gradient based on the parameter shift rule, as in (https://quri-parts.qunasys.com/docs/tutorials/advanced/parametric/gradient_estimators/#parameter-shift-gradient-estimator)
            """

            param_mapping = ansatz.param_mapping
            raw_circuit = ansatz.primitive_circuit()
            parameter_shift = ShiftedParameters(param_mapping)
            derivatives = parameter_shift.get_derivatives()
            shifted_parameters_and_coefs = [
                d.get_shifted_parameters_and_coef(list(params)) for d in derivatives
            ]

            gate_params = set()
            for params_and_coefs in shifted_parameters_and_coefs:
                for p, _ in params_and_coefs:
                    gate_params.add(p)
            gate_params_list = list(gate_params)

            estimates = [
                self.cost_fn(target_circuit, raw_circuit.bind_parameters(list(p)))
                for p in gate_params_list
            ]
            estimates_dict = dict(zip(gate_params_list, estimates))

            gradient = []
            for params_and_coefs in shifted_parameters_and_coefs:
                g = 0.0
                for p, c in params_and_coefs:
                    g += estimates_dict[p].value.real * c.real
                gradient.append(g)

            return np.array(gradient)

        if init_params is None:
            rng = default_rng()
            init_params = rng.random(ansatz.parameter_count) * 2 * np.pi
        optimizer_state = self.optimizer.get_init_state(init_params)
        while optimizer_state.status == OptimizerStatus.SUCCESS:
            optimizer_state = self.optimizer.step(
                optimizer_state, cost, grad_function=gradient
            )

        opt_params = optimizer_state.params
        opt_circuit = ansatz.bind_parameters(list(opt_params.tolist()))

        return opt_circuit, optimizer_state

    def __call__(
        self,
        circuit_factory: CircuitFactory,
        ansatz: LinearMappedUnboundParametricQuantumCircuit,
        init_params: Optional[Params] = None,
        *args: Any,
        **kwargs: Any,
    ) -> NonParametricQuantumCircuit:
        """Perform QAQC compilation and return circuit.

        Arguments:
        circuit_factory - Circuit factory which generates the target circuit for the compilation
        ansatz - Parametric ansatz circuit that is used to approximate the target circuit
        init_params - initial variational parameters for the optimization
        variable arguments if any are passed to the circuit_factory's __call__ method

        Return value:
        Compiled circuit
        """
        self.compiled_circuit, _ = self.optimize(
            circuit_factory, ansatz, init_params, *args, **kwargs
        )
        return self.compiled_circuit
