import pytest
from quri_parts.core.operator import pauli_label
from quri_parts.core.state import ComputationalBasisState

from quri_parts.itensor.estimator import create_itensor_mps_estimator


def test_estimate_pauli_label() -> None:
    pauli = pauli_label("Z0 Z2 Z5")
    state = ComputationalBasisState(6, bits=0b110010)
    estimator = create_itensor_mps_estimator()
    estimate = estimator(pauli, state)
    assert estimate.value == -1
    assert estimate.error == 0
