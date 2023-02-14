import pytest

from quri_parts.circuit import QuantumCircuit
from quri_parts.itensor.sampler import (
    create_itensor_mps_sampler,
)


def circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(4)
    circuit.add_X_gate(0)
    circuit.add_H_gate(1)
    circuit.add_Z_gate(2)
    circuit.add_Y_gate(3)
    return circuit


class TestITensorMPSSampler:
    @pytest.mark.parametrize("qubits", [4, 12])
    @pytest.mark.parametrize("shots", [800, 1200, 2**12 + 100])
    def test_sampler(self, qubits: int, shots: int) -> None:
        circuit = QuantumCircuit(qubits)
        for i in range(qubits):
            circuit.add_H_gate(i)

        sampler = create_itensor_mps_sampler()
        counts = sampler(circuit, shots)

        assert set(counts.keys()).issubset(range(2**qubits))
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == shots
