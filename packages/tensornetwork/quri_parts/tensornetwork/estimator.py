# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import NamedTuple

import tensornetwork as tn

from quri_parts.core.estimator import Estimate, QuantumEstimator
from quri_parts.core.operator import Operator, PauliLabel
from quri_parts.core.state import GeneralCircuitQuantumState
from quri_parts.tensornetwork.circuit import (
    TensorNetworkLayer,
    TensorNetworkOperator,
    TensorNetworkState,
    convert_state,
)
from quri_parts.tensornetwork.operator import operator_to_tensor, tensor_to_mpo


class _Estimate(NamedTuple):
    value: complex
    error: float = 0.0


def tensor_network_estimate(
    operator: TensorNetworkOperator, state: TensorNetworkState
) -> Estimate:
    copy_state = state.copy()
    conj_state = state.conjugate()
    copy_operator = operator.copy()

    for i, (e, f, g, h) in enumerate(
        zip(
            copy_state.edges,
            copy_operator.input_edges,
            copy_operator.output_edges,
            conj_state.edges,
        )
    ):
        if i in operator.index_list:
            e ^ f
            g ^ h
        else:
            e ^ h

    contracted_node = tn.contractors.optimal(
        copy_state._container.union(
            conj_state._container.union(copy_operator._container)
        )
    )

    return _Estimate(
        value=contracted_node.tensor.item(),
    )  # Can we estimate the error based on the MPO truncation error?


def create_tensornetwork_estimator(
    backend="numpy", max_bond_dimension=None, max_truncation_error=None
) -> QuantumEstimator:
    """This creates an estimator using the tensornetwork backend.

    Args:
        backend - the computational backend to use. Currently supports numpy.
        max_bond_dimension - the bond dimension of the MPO
        max_truncation_error - the maximum allowed truncation error when performing the SVD on the MPO
    """

    def estimate(
        operator: Operator | PauliLabel, state: GeneralCircuitQuantumState
    ) -> Estimate:
        tn_operator = operator_to_tensor(operator)
        mpo = tensor_to_mpo(tn_operator, max_bond_dimension, max_truncation_error)
        tn_circuit = convert_state(state)
        return tensor_network_estimate(mpo, tn_circuit)

    return estimate
