# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Sequence

from quri_parts.circuit import (
    ImmutableBoundParametricQuantumCircuit,
    UnboundParametricQuantumCircuit,
)
from quri_parts.core.estimator import (
    ConcurrentParametricQuantumEstimator,
    Estimatable,
    GeneralQuantumEstimators,
    ParametricQuantumEstimator,
    create_concurrent_estimator_from_estimator,
    create_concurrent_parametric_estimator_from_concurrent_estimator,
    create_estimator_from_concurrent_estimator,
    create_general_estimators_from_concurrent_estimator,
    create_general_estimators_from_estimator,
    create_parametric_estimator_from_concurrent_estimator,
)
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label
from quri_parts.core.state import (
    CircuitQuantumState,
    ComputationalBasisState,
    GeneralCircuitQuantumState,
    ParametricCircuitQuantumState,
)


@dataclass
class _Estimate:
    value: complex
    error: float = 0.0


def fake_estimator(op: Estimatable, state: CircuitQuantumState) -> _Estimate:
    op_contribution = len(op) if isinstance(op, Operator) else 1

    if isinstance(state, ComputationalBasisState):
        return _Estimate(value=state.bits * op_contribution + 0j)
    elif isinstance(state, GeneralCircuitQuantumState) and isinstance(
        state.circuit, ImmutableBoundParametricQuantumCircuit
    ):
        e_list = [v * op_contribution for v in state.circuit.parameter_map.values()]
        return _Estimate(sum(e_list))
    else:
        assert False


def fake_concurrent_estimator(
    operators: Sequence[Estimatable], states: Sequence[CircuitQuantumState]
) -> Sequence[_Estimate]:
    num_ops = len(operators)
    num_states = len(states)

    if num_states == 1:
        states = [next(iter(states))] * num_ops

    if num_ops == 1:
        operators = [next(iter(operators))] * num_states

    return [fake_estimator(op, state) for (op, state) in zip(operators, states)]


def test_create_concurrent_estimator_from_estimator() -> None:
    concurrent_estimator = create_concurrent_estimator_from_estimator(fake_estimator)
    op = PAULI_IDENTITY
    state_1 = ComputationalBasisState(1)
    state_2 = ComputationalBasisState(1, bits=1)
    assert concurrent_estimator([op], [state_1]) == [_Estimate(value=0 + 0j)]
    assert concurrent_estimator([op], [state_1, state_2]) == [
        _Estimate(0 + 0j),
        _Estimate(1 + 0j),
    ]
    assert concurrent_estimator([op, op], [state_1]) == [
        _Estimate(0 + 0j),
        _Estimate(0 + 0j),
    ]

    op_1 = Operator({pauli_label("X0"): 1, pauli_label("Y0"): 1})
    assert concurrent_estimator([op, op_1], [state_1, state_2]) == [
        _Estimate(value=0 + 0j),
        _Estimate(value=2 + 0j),
    ]


def test_create_estimator_from_concurrent_estimator() -> None:
    estimator = create_estimator_from_concurrent_estimator(fake_concurrent_estimator)
    op = PAULI_IDENTITY
    state = ComputationalBasisState(1, bits=1)
    assert estimator(op, state) == _Estimate(value=1 + 0j)


def test_create_general_esimtators_from_estimator() -> None:
    general_estimators: GeneralQuantumEstimators[
        CircuitQuantumState, ParametricCircuitQuantumState
    ] = create_general_estimators_from_estimator(fake_estimator)

    op_0 = PAULI_IDENTITY
    op_1 = Operator({pauli_label("X0"): 1, pauli_label("Y0"): 1})
    state_0 = ComputationalBasisState(1)
    state_1 = ComputationalBasisState(1, bits=1)

    estimator = general_estimators.estimator
    assert estimator(op_0, state_0) == _Estimate(value=0 + 0j)
    assert estimator(op_0, state_1) == _Estimate(value=1 + 0j)

    concurrent_estimator = general_estimators.concurrent_estimator
    assert concurrent_estimator([op_0], [state_0]) == [_Estimate(value=0 + 0j)]
    assert concurrent_estimator([op_0], [state_0, state_1]) == [
        _Estimate(value=0 + 0j),
        _Estimate(value=1 + 0j),
    ]
    assert concurrent_estimator([op_0, op_1], [state_1]) == [
        _Estimate(value=1 + 0j),
        _Estimate(value=2 + 0j),
    ]
    assert concurrent_estimator([op_0, op_1], [state_0, state_1]) == [
        _Estimate(value=0 + 0j),
        _Estimate(value=2 + 0j),
    ]

    param_circuit = UnboundParametricQuantumCircuit(1)
    param_circuit.add_ParametricRX_gate(0)
    param_circuit.add_ParametricRY_gate(0)
    param_circuit.add_ParametricRZ_gate(0)
    param_state = ParametricCircuitQuantumState(1, param_circuit)

    parametric_estimator = general_estimators.parametric_estimator
    assert parametric_estimator(op_0, param_state, [0, 1, 2]) == _Estimate(value=3 + 0j)
    assert parametric_estimator(op_0, param_state, [3, 4, 5]) == _Estimate(
        value=12 + 0j
    )

    concurrent_param_estimator = general_estimators.concurrent_parametric_estimator
    assert concurrent_param_estimator(op_0, param_state, [[0, 1, 2], [3, 4, 5]]) == [
        _Estimate(value=3 + 0j),
        _Estimate(value=12 + 0j),
    ]


def test_create_general_esimtators_from_concurrent_estimator() -> None:
    general_estimators: GeneralQuantumEstimators[
        CircuitQuantumState, ParametricCircuitQuantumState
    ] = create_general_estimators_from_concurrent_estimator(fake_concurrent_estimator)

    op_0 = PAULI_IDENTITY
    op_1 = Operator({pauli_label("X0"): 1, pauli_label("Y0"): 1})
    state_0 = ComputationalBasisState(1)
    state_1 = ComputationalBasisState(1, bits=1)

    estimator = general_estimators.estimator
    assert estimator(op_0, state_0) == _Estimate(value=0 + 0j)
    assert estimator(op_0, state_1) == _Estimate(value=1 + 0j)

    concurrent_estimator = general_estimators.concurrent_estimator
    assert concurrent_estimator([op_0], [state_0]) == [_Estimate(value=0 + 0j)]
    assert concurrent_estimator([op_0], [state_0, state_1]) == [
        _Estimate(value=0 + 0j),
        _Estimate(value=1 + 0j),
    ]
    assert concurrent_estimator([op_0, op_1], [state_1]) == [
        _Estimate(value=1 + 0j),
        _Estimate(value=2 + 0j),
    ]
    assert concurrent_estimator([op_0, op_1], [state_0, state_1]) == [
        _Estimate(value=0 + 0j),
        _Estimate(value=2 + 0j),
    ]

    param_circuit = UnboundParametricQuantumCircuit(1)
    param_circuit.add_ParametricRX_gate(0)
    param_circuit.add_ParametricRY_gate(0)
    param_circuit.add_ParametricRZ_gate(0)
    param_state = ParametricCircuitQuantumState(1, param_circuit)

    parametric_estimator = general_estimators.parametric_estimator
    assert parametric_estimator(op_0, param_state, [0, 1, 2]) == _Estimate(value=3 + 0j)
    assert parametric_estimator(op_0, param_state, [3, 4, 5]) == _Estimate(
        value=12 + 0j
    )

    concurrent_param_estimator = general_estimators.concurrent_parametric_estimator
    assert concurrent_param_estimator(op_0, param_state, [[0, 1, 2], [3, 4, 5]]) == [
        _Estimate(value=3 + 0j),
        _Estimate(value=12 + 0j),
    ]


def test_create_parametric_estimator_from_concurrent_estimator() -> None:
    parametric_estimator: ParametricQuantumEstimator[
        ParametricCircuitQuantumState
    ] = create_parametric_estimator_from_concurrent_estimator(fake_concurrent_estimator)

    param_circuit = UnboundParametricQuantumCircuit(1)
    param_circuit.add_ParametricRX_gate(0)
    param_circuit.add_ParametricRY_gate(0)
    param_circuit.add_ParametricRZ_gate(0)
    param_state = ParametricCircuitQuantumState(1, param_circuit)

    op = PAULI_IDENTITY
    assert parametric_estimator(op, param_state, [0, 1, 2]) == _Estimate(value=3 + 0j)
    assert parametric_estimator(op, param_state, [3, 4, 5]) == _Estimate(value=12 + 0j)


def test_create_concurrent_parametric_estimator_from_concurrent_estimator() -> None:
    concurrent_param_estimator: ConcurrentParametricQuantumEstimator[
        ParametricCircuitQuantumState
    ] = create_concurrent_parametric_estimator_from_concurrent_estimator(
        fake_concurrent_estimator
    )

    param_circuit = UnboundParametricQuantumCircuit(1)
    param_circuit.add_ParametricRX_gate(0)
    param_circuit.add_ParametricRY_gate(0)
    param_circuit.add_ParametricRZ_gate(0)
    param_state = ParametricCircuitQuantumState(1, param_circuit)

    op = PAULI_IDENTITY
    assert concurrent_param_estimator(op, param_state, [[0, 1, 2], [3, 4, 5]]) == [
        _Estimate(value=3 + 0j),
        _Estimate(value=12 + 0j),
    ]
