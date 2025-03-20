from typing import Protocol

from quri_algo.problem.operators.interface import OperatorT


class OperatorFunctionEstimation(Protocol[OperatorT]):
    """Represents a problem that requires estimation of an expectation value of
    an operator function with respect to a state."""

    operator: OperatorT
    