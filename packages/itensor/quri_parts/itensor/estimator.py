import os
from typing import NamedTuple, Union

import juliacall
from juliacall import Main as jl
from quri_parts.core.estimator import Estimatable, Estimate, QuantumEstimator
from quri_parts.core.operator import Operator, zero
from quri_parts.core.state import (
    CircuitQuantumState,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
    QuantumStateVector,
)
from typing_extensions import TypeAlias

from .circuit import convert_circuit
from .operator import convert_operator

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
