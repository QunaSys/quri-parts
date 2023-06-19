# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, TypeVar, Union, cast

from openfermion.ops import FermionOperator

from quri_parts.chem.utils.excitations import DoubleExcitation, SingleExcitation
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit, Parameter
from quri_parts.core.operator import Operator

from ..transforms import OpenFermionQubitOperatorMapper

Excitation = TypeVar("Excitation", SingleExcitation, DoubleExcitation)


def add_exp_excitation_gates_trotter_decomposition(
    circuit: LinearMappedUnboundParametricQuantumCircuit,
    excitation_indices: Sequence[Excitation],
    params: Sequence[Parameter],
    operator_mapper: OpenFermionQubitOperatorMapper,
    coef: float,
) -> LinearMappedUnboundParametricQuantumCircuit:
    """Add parametric Pauli rotation gates as a product of the exponentials of
    the excitations to the given :attr:`circuit`."""
    for i, sorb_indices in enumerate(excitation_indices):
        op = _create_operator(sorb_indices, operator_mapper)
        for pauli, op_coef in op.items():
            pauli_index_list, pauli_id_list = zip(*pauli)
            op_coef = op_coef.imag
            circuit.add_ParametricPauliRotation_gate(
                pauli_index_list,
                pauli_id_list,
                {params[i]: -2.0 * op_coef * coef},
            )
    return circuit


def add_exp_pauli_gates_from_linear_mapped_function(
    circuit: LinearMappedUnboundParametricQuantumCircuit,
    param_fn: dict[Parameter, float],
    qp_operator: Operator,
    coeff: float = 1,
) -> None:
    """Add exponential pauli rotation gate to a
    :class:`~LinearMappedUnboundParametricQuantumCircuit` in place
    according to the equation:

    .. math::
        \\exp (
            i \\text{c} * f(\\theta_1, \\theta_2, \\cdots) * \\text{qp_operator}
        )

    Arg:
        circuit: The circuit to add a pauli exponential rotation gate to.
        param_fn: A dict representing parametric function in front of the qp_operator.
        qp_operator: A Hermitian quri-parts operator.
        coeff: An overall real coeffcient in the exponent.
    """
    for pauli, op_coeff in qp_operator.items():
        if str(pauli) == "I":
            # This corresponds to a global phase.
            continue
        pauli_index_list, pauli_id_list = zip(*pauli)
        new_param_mapping = {
            param: -2 * op_coeff.real * old_coeff * coeff
            for param, old_coeff in param_fn.items()
        }
        circuit.add_ParametricPauliRotation_gate(
            pauli_index_list, pauli_id_list, new_param_mapping
        )


def _create_operator(
    excitation_indices: Union[SingleExcitation, DoubleExcitation],
    operator_mapper: OpenFermionQubitOperatorMapper,
) -> Operator:
    op = FermionOperator()
    if len(excitation_indices) == 2:
        op += FermionOperator(
            ((excitation_indices[1], 1), (excitation_indices[0], 0)), 1.0
        )
        op += FermionOperator(
            ((excitation_indices[0], 1), (excitation_indices[1], 0)), -1.0
        )
    elif len(excitation_indices) == 4:
        excitation_indices = cast(DoubleExcitation, excitation_indices)
        op += FermionOperator(
            (
                (excitation_indices[3], 1),
                (excitation_indices[2], 1),
                (excitation_indices[1], 0),
                (excitation_indices[0], 0),
            ),
            1.0,
        )
        op += FermionOperator(
            (
                (excitation_indices[0], 1),
                (excitation_indices[1], 1),
                (excitation_indices[2], 0),
                (excitation_indices[3], 0),
            ),
            -1.0,
        )
    return operator_mapper(op)
