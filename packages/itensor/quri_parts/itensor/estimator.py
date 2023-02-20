import os
from collections.abc import Collection, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Union

import juliacall
from juliacall import Main as jl
from quri_parts.core.estimator import (
    ConcurrentParametricQuantumEstimator,
    ConcurrentQuantumEstimator,
    Estimatable,
    Estimate,
    ParametricQuantumEstimator,
    QuantumEstimator,
    create_parametric_estimator,
)
from quri_parts.core.operator import Operator, zero
from quri_parts.core.state import (
    CircuitQuantumState,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
    QuantumStateVector,
)
from quri_parts.core.utils.concurrent import execute_concurrently
from typing_extensions import TypeAlias

from .circuit import convert_circuit, convert_parametric_circuit
from .operator import convert_operator

if TYPE_CHECKING:
    from concurrent.futures import Executor

path = os.getcwd()
library_path = os.path.join(path, "packages/itensor/quri_parts/itensor/library.jl")

jl.seval("using ITensors")
include_statement = 'include("' + library_path + '")'
jl.seval(include_statement)


class _Estimate(NamedTuple):
    value: complex
    error: float = 0.0


#: A type alias for state classes supported by Qulacs estimators.
#: Qulacs estimators support both of circuit states and state vectors.
QulacsStateT: TypeAlias = Union[CircuitQuantumState, QuantumStateVector]

#: A type alias for parametric state classes supported by Qulacs estimators.
#: Qulacs estimators support both of circuit states and state vectors.
QulacsParametricStateT: TypeAlias = Union[
    ParametricCircuitQuantumState, ParametricQuantumStateVector
]


def _estimate(operator: Estimatable, state: QulacsStateT) -> Estimate[complex]:
    if operator == zero():
        return _Estimate(value=0.0)
    qubits = state.qubit_count
    s: juliacall.VectorValue = jl.siteinds("Qubit", qubits)
    psi: juliacall.AnyValue = jl.initState(s, qubits)

    # create ITensor circuit
    circuit = convert_circuit(state.circuit, s)

    # create ITensor operator
    op = convert_operator(operator, s)

    # calculate expectation value
    psi = jl.apply(circuit, psi)
    exp: float = jl.expectation(psi, op)

    return _Estimate(value=exp)


def create_itensor_mps_estimator() -> QuantumEstimator[QulacsStateT]:
    """Returns a :class:`~QuantumEstimator` that uses ITensor MPS simulator
    to calculate expectation values."""

    return _estimate


def _sequential_estimate(
    _: Any, op_state_tuples: Sequence[tuple[Estimatable, QulacsStateT]]
) -> Sequence[Estimate[complex]]:
    return [_estimate(operator, state) for operator, state in op_state_tuples]


def _sequential_estimate_single_state(
    state: QulacsStateT, operators: Sequence[Estimatable]
) -> Sequence[Estimate[complex]]:
    qubits = state.qubit_count
    s: juliacall.VectorValue = jl.siteinds("Qubit", qubits)
    psi: juliacall.AnyValue = jl.initState(s, qubits)
    circuit = convert_circuit(state.circuit, s)
    psi = jl.apply(circuit, psi)
    results = []
    for op in operators:
        itensor_op = convert_operator(op, s)
        results.append(_Estimate(value=jl.expectation(psi, itensor_op)))
    return results


def _concurrent_estimate(
    _sequential_estimate: Callable[
        [Any, Sequence[tuple[Estimatable, QulacsStateT]]], Sequence[Estimate[complex]]
    ],
    _sequential_estimate_single_state: Callable[
        [QulacsStateT, Sequence[Estimatable]], Sequence[Estimate[complex]]
    ],
    operators: Collection[Estimatable],
    states: Collection[QulacsStateT],
    executor: Optional["Executor"],
    concurrency: int = 1,
) -> Sequence[Estimate[complex]]:
    num_ops = len(operators)
    num_states = len(states)

    if num_ops == 0:
        raise ValueError("No operator specified.")

    if num_states == 0:
        raise ValueError("No state specified.")

    if num_ops > 1 and num_states > 1 and num_ops != num_states:
        raise ValueError(
            f"Number of operators ({num_ops}) does not match"
            f"number of states ({num_states})."
        )

    if num_states == 1:
        return execute_concurrently(
            _sequential_estimate_single_state,
            next(iter(states)),
            operators,
            executor,
            concurrency,
        )
    else:
        if num_ops == 1:
            operators = [next(iter(operators))] * num_states
        return execute_concurrently(
            _sequential_estimate, None, zip(operators, states), executor, concurrency
        )


def create_itensor_mps_concurrent_estimator(
    executor: Optional["Executor"] = None, concurrency: int = 1
) -> ConcurrentQuantumEstimator[QulacsStateT]:
    """Returns a :class:`~ConcurrentQuantumEstimator` that uses ITensor MPS
    simulator to calculate expectation values."""

    def estimator(
        operators: Collection[Estimatable],
        states: Collection[QulacsStateT],
    ) -> Iterable[Estimate[complex]]:
        return _concurrent_estimate(
            _sequential_estimate,
            _sequential_estimate_single_state,
            operators,
            states,
            executor,
            concurrency,
        )

    return estimator


def _sequential_parametric_estimate(
    op_state: tuple[Estimatable, QulacsParametricStateT],
    params: Sequence[Sequence[float]],
) -> Sequence[Estimate[complex]]:
    operator, state = op_state
    n_qubits = state.qubit_count
    s = jl.siteinds("Qubit", n_qubits)

    op = convert_operator(operator, s)

    parametric_circuit = state.parametric_circuit
    estimates = []

    for param in params:
        psi: juliacall.AnyValue = jl.initState(s, n_qubits)
        circuit = convert_parametric_circuit(parametric_circuit, s, param)
        psi = jl.apply(circuit, psi)
        estimates.append(_Estimate(value=jl.expectation(psi, op)))
    return estimates


def create_itensor_mps_parametric_estimator() -> ParametricQuantumEstimator[
    QulacsParametricStateT
]:
    def estimator(
        operator: Estimatable, state: QulacsParametricStateT, param: Sequence[float]
    ) -> Estimate[complex]:
        ests = _sequential_parametric_estimate((operator, state), [param])
        return ests[0]

    return estimator


def create_itensor_mps_concurrent_parametric_estimator(
    executor: Optional["Executor"] = None, concurrency: int = 1
) -> ConcurrentParametricQuantumEstimator[QulacsParametricStateT]:
    def estimator(
        operator: Estimatable,
        state: QulacsParametricStateT,
        params: Sequence[Sequence[float]],
    ) -> Sequence[Estimate[complex]]:
        return execute_concurrently(
            _sequential_parametric_estimate,
            (operator, state),
            params,
            executor,
            concurrency,
        )

    return estimator


if __name__ == "__main__":
    from quri_parts.core.operator import pauli_label
    from quri_parts.core.state import ComputationalBasisState

    pauli = pauli_label("Z0 Z2 Z5")
    state = ComputationalBasisState(6, bits=0b110010)
    estimator = create_itensor_mps_estimator()
    estimate = estimator(pauli, state)
    assert estimate.value == -1
    assert estimate.error == 0

    operator = Operator(
        {
            pauli_label("Z0 Z2 Z5"): 0.25,
            pauli_label("Z1 Z2 Z4"): 0.5j,
        }
    )
    state = ComputationalBasisState(6, bits=0b110010)
    estimator = create_itensor_mps_estimator()
    estimate = estimator(operator, state)
    print(estimate.value)
    assert estimate.value == -0.25 + 0.5j
    assert estimate.error == 0
