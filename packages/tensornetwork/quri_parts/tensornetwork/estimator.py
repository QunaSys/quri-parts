# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import NamedTuple, Optional, Union

import tensornetwork as tn

from quri_parts.core.estimator import Estimate, QuantumEstimator
from quri_parts.core.operator import Operator, PauliLabel
from quri_parts.core.state import CircuitQuantumState
from quri_parts.tensornetwork.operator import TensorNetworkOperator, operator_to_tensor
from quri_parts.tensornetwork.state import TensorNetworkState, convert_state


class _Estimate(NamedTuple):
    value: complex
    error: float = 0.0


def tensor_network_estimate(
    operator: TensorNetworkOperator, state: TensorNetworkState
) -> Estimate[complex]:
    copy_state = state.copy()
    conj_state = copy_state.conjugate()
    copy_operator = operator.copy()
    operator_ordered_indices = sorted(list(operator.index_list))

    for i, (e, h) in enumerate(
        zip(
            copy_state.edges,
            conj_state.edges,
        )
    ):
        if i in operator.index_list:
            indx = operator_ordered_indices.index(i)
            f = copy_operator.input_edges[indx]
            g = copy_operator.output_edges[indx]
            e ^ f
            g ^ h
        else:
            e ^ h

    contracted_node = tn.contractors.greedy(
        copy_state._container.union(
            conj_state._container.union(copy_operator._container)
        )
    )

    return _Estimate(
        value=contracted_node.tensor.item(),
    )  # Can we estimate the error based on the MPO truncation error?


def create_tensornetwork_estimator(
    backend: str = "numpy",
    matrix_product_operator: bool = True,
    max_bond_dimension: Optional[int] = None,
    max_truncation_err: Optional[float] = None,
) -> QuantumEstimator[CircuitQuantumState]:
    """This creates an estimator using the tensornetwork backend.

    Args:
        backend - the computational backend to use. Currently supports numpy.
        max_bond_dimension - the bond dimension of the MPO
        max_truncation_error - the maximum allowed truncation error when
            performing the SVD on the MPO
    """

    def estimate(
        operator: Union[Operator, PauliLabel], state: CircuitQuantumState
    ) -> Estimate[complex]:
        tn_operator = operator_to_tensor(
            operator,
            convert_to_mpo=matrix_product_operator,
            max_bond_dimension=max_bond_dimension,
            max_truncation_err=max_truncation_err,
            backend=backend,
        )
        tn_state = convert_state(state,
            backend=backend,)
        return tensor_network_estimate(tn_operator, tn_state)

    return estimate

from tqdm import tqdm
from quri_parts.core.operator import pauli_label
from quri_parts.core.state import GeneralCircuitQuantumState
from quri_parts.circuit import QuantumCircuit

def main():
    op = Operator({
        pauli_label("X0 X1"): 1.0,
        pauli_label("X1 X2"): 1.0,
        pauli_label("X2 X3"): 1.0,
        pauli_label("X3 X4"): 1.0,
        pauli_label("X4 X5"): 1.0,
        pauli_label("X5 X6"): 1.0,
        pauli_label("X6 X7"): 1.0,
        pauli_label("X7 X8"): 1.0,
        pauli_label("X8 X9"): 1.0,
        pauli_label("X9 X0"): 1.0,
    })
    circuit = QuantumCircuit(10)
    circuit.add_H_gate(0)
    circuit.add_H_gate(1)
    circuit.add_H_gate(2)
    circuit.add_H_gate(3)
    circuit.add_H_gate(4)
    circuit.add_H_gate(5)
    circuit.add_H_gate(6)
    circuit.add_H_gate(7)
    circuit.add_H_gate(8)
    circuit.add_H_gate(9)
    state = GeneralCircuitQuantumState(10, circuit)
    estimator = create_tensornetwork_estimator(backend = "pytorch")

    for _ in tqdm(range(100)):
        e = estimator(op,state)
    print(e)


if __name__ == "__main__":
    main()