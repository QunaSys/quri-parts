import unittest

from numpy import pi
from pytket.backends.backend import Backend
from pytket.extensions.qiskit import AerBackend  # type: ignore

from quri_parts.backend import SamplingJob, SamplingResult
from quri_parts.circuit import QuantumCircuit
from quri_parts.tket.sampler import TKetSamplingJob  # type: ignore


class TestTKetSamplingBackend(unittest.TestCase):
    backend: Backend
    sampling_job: SamplingJob
    circuit: QuantumCircuit
    result: SamplingResult

    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = AerBackend()

        cls.circuit = QuantumCircuit(4)
        cls.circuit.add_X_gate(0)
        cls.circuit.add_X_gate(1)
        cls.circuit.add_CNOT_gate(0, 1)
        cls.circuit.add_X_gate(0)
        cls.circuit.add_CNOT_gate(0, 1)
        cls.circuit.add_RX_gate(0, 3.87 * pi)
        cls.circuit.add_U3_gate(2, 3.87 * pi, 1.23 * pi, -0.9 * pi)
        cls.circuit.add_U2_gate(1, 1.23 * pi, -0.9 * pi)
        cls.circuit.add_Identity_gate(0)
        cls.circuit.add_SWAP_gate(0, 3)
        cls.circuit.add_UnitaryMatrix_gate([3], [[0, 1], [1, 0]])

        cls.sampling_job = TKetSamplingJob(
            circuit=cls.circuit, n_shots=10000, backend=cls.backend
        )
        cls.result = cls.sampling_job.result()

    def test_sample(self) -> None:
        assert sum(self.result.counts.values()) == 10000
        assert set(self.result.counts.keys()) == {
            0b0000,
            0b0001,
            0b0010,
            0b0011,
            0b0100,
            0b0101,
            0b0110,
            0b0111,
        }
