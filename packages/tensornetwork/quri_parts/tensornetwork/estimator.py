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
from quri_parts.tensornetwork.state import (
    TensorNetworkState,
    convert_state,
    convert_state_mps,
)


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
        tn_state = convert_state(
            state,
            backend=backend,
        )
        return tensor_network_estimate(tn_operator, tn_state)

    return estimate


def create_tensornetwork_mps_estimator(
    backend: str = "numpy",
    matrix_product_operator: bool = True,
    max_bond_dimension: Optional[int] = None,
    max_truncation_err: Optional[float] = None,
) -> QuantumEstimator[CircuitQuantumState]:
    """This creates an estimator using the tensornetwork backend. Quantum
    states are represented using matrix product states with time-evolving block
    decimation.

    Args:
        backend - the computational backend to use. Currently supports numpy.
        max_bond_dimension - the bond dimension of the MPO and MPS
        max_truncation_error - the maximum allowed truncation error when
            performing the SVD on the MPO and MPS
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
        tn_state = convert_state_mps(
            state,
            backend=backend,
        )
        tn_state = tn_state.contract()
        return tensor_network_estimate(tn_operator, tn_state)

    return estimate


from quri_parts.core.operator import pauli_label
from quri_parts.algo.ansatz import HardwareEfficient
from quri_parts.tensornetwork.estimator import create_tensornetwork_mps_estimator
from quri_parts.algo.optimizer import LBFGS
from quri_parts.algo.ansatz.two_local import (
    EntanglementPatternType,
    build_entangler_map,
)
from quri_parts.core.state import GeneralCircuitQuantumState


def main():
    REPS = 4
    d = 4
    j = 1.0
    s = 1 / 2
    heisenberg = Operator(
        {
            pauli_label("X0 X1"): j * s**2,
            pauli_label("X1 X2"): j * s**2,
            pauli_label("X2 X3"): j * s**2,
            pauli_label("X3 X0"): j * s**2,
            pauli_label("Y0 Y1"): j * s**2,
            pauli_label("Y1 Y2"): j * s**2,
            pauli_label("Y2 Y3"): j * s**2,
            pauli_label("Y3 Y0"): j * s**2,
            pauli_label("Z0 Z1"): j * s**2,
            pauli_label("Z1 Z2"): j * s**2,
            pauli_label("Z2 Z3"): j * s**2,
            pauli_label("Z3 Z0"): j * s**2,
        }
    )
    # heisenberg_input = QubitHamiltonianInput(4, heisenberg)
    # circuit_factory = TrotterTimeEvolutionCircuitFactory(heisenberg_input, 1)
    # estimator = create_qulacs_vector_estimator()
    mps_estimator = create_tensornetwork_mps_estimator(
        matrix_product_operator=True, max_bond_dimension=d
    )
    # local_cost_fn = LocalHilbertSchmidtTestRestrictedPBC(
    #     mps_estimator, restriction_size=2
    # )
    # optimizer = LBFGS()

    # lvqc = LVQC(local_cost_fn, optimizer)
    entangler_map = build_entangler_map(4, [EntanglementPatternType.LINEAR] * REPS)
    ansatz = HardwareEfficient(4, REPS, entangler_map_seq=entangler_map)

    angles = [
        0.0,
    ] * ansatz.parameter_count
    circuit = ansatz.bind_parameters(angles)
    state = GeneralCircuitQuantumState(4, circuit)
    result = mps_estimator(heisenberg, state)
    print(result.value)


if __name__ == "__main__":
    main()
