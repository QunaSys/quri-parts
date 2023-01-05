# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "sAS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import pytest
from quri_parts.backend import CompositeSamplingJob
from quri_parts.circuit import NonParametricQuantumCircuit, QuantumCircuit
from quri_parts.circuit.transpile import CircuitTranspiler

from qiskit.circuit import QuantumCircuit as QiskitCircuit
from qiskit.providers.fake_provider.fake_backend import FakeBackendV2
from qiskit.providers.fake_provider import FakeMelbourneV2
from qiskit.result import Result as QiskitResult
from qiskit.test import QiskitTestCase

from quri_parts.qiskit.backend import QiskitSamplingBackend, QiskitSamplingResult


def circuit_converter(
    _: NonParametricQuantumCircuit,
    transpiler: Optional[CircuitTranspiler] = None
) -> QiskitCircuit:
    circuit = QiskitCircuit(4)
    circuit.h(0)
    circuit.i(1)
    circuit.h(2)
    circuit.i(3)
    return circuit


def circuit_transpiler(_: NonParametricQuantumCircuit) -> NonParametricQuantumCircuit:
    circuit = QuantumCircuit(4)
    circuit.add_X_gate(0)
    circuit.add_H_gate(1)
    circuit.add_Z_gate(2)
    circuit.add_Y_gate(3)
    return circuit


class TestSamplingResult(QiskitTestCase):
    def setUp(self):
        self.base_result_args = dict(
            backend_name="test_backend",
            backend_version="1.0.0",
            qobj_id="id-137",
            job_id="job-137",
            success=True,
            status="Successed"
        )

        super().setUp()

#    def test_measured_qubits(self) -> None:
#        raw_counts = {"0x0": 2, "0x2": 2}
#        qiskit_result = QiskitResult(results=[raw_counts], **self.base_result_args)
#        result = QiskitSamplingResult(qiskit_result)
#        counts = result.counts
#        assert counts == {0b00: 2, 0b10: 2}


class TestSamplingBackend(QiskitTestCase):
    def setUp(self):
        self.backend = FakeMelbourneV2()
        super().setUp()

    def test_sample(self) -> None:
        device = self.backend
        backend = QiskitSamplingBackend(device, circuit_converter)
        job = backend.sample(QuantumCircuit(4), 1000)
        counts = job.result().counts

        #assert set(counts.keys()) in {i for i in range(16)}
        assert sum(counts.values()) == 1000

    def test_default_circuit_converter(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.add_X_gate(0)
        circuit.add_H_gate(1)
        circuit.add_Z_gate(2)

        device = self.backend
        backend = QiskitSamplingBackend(device)
        job = backend.sample(circuit, 1000)
        counts = job.result().counts

        #assert set(counts.keys()) in {i for i in range(16)}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 1000

    def test_circuit_transpiler(self) -> None:
        circuit = QuantumCircuit(4)
        backend = QiskitSamplingBackend(
            self.backend, circuit_transpiler=circuit_transpiler
        )  # With default circuit_converter.
        job = backend.sample(circuit, 1000)
        counts = job.result().counts

        #assert set(counts.keys()) in {i for i in range(16)}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 1000

    def test_min_shots(self) -> None:
        backend = QiskitSamplingBackend(self.backend, circuit_converter)
        backend._min_shots = 1000
        job = backend.sample(QuantumCircuit(4), 50)
        counts = job.result().counts

        #assert set(counts.keys()) in {i for i in range(16)}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 1000

        backend = QiskitSamplingBackend(
            self.backend, circuit_converter, enable_shots_roundup=False
        )
        backend._min_shots = 1000
        with pytest.raises(ValueError):
            job = backend.sample(QuantumCircuit(4), 50)

    def test_max_shots(self) -> None:
        backend = QiskitSamplingBackend(self.backend, circuit_converter)
        backend._max_shots = 1000
        job = backend.sample(QuantumCircuit(4), 2100)
        counts = job.result().counts

        #assert set(counts.keys()) in {i for i in range(16)}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 2100
        assert isinstance(job, CompositeSamplingJob)
        assert sorted(sum(j.result().counts.values()) for j in job.jobs) == [
            100,
            1000,
            1000,
        ]

        backend = QiskitSamplingBackend(self.backend, circuit_converter)
        backend._max_shots = 1000
        backend._min_shots = 200
        job = backend.sample(QuantumCircuit(4), 2100)
        counts = job.result().counts

        #assert set(counts.keys()) in {i for i in range(16)}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 2200
        assert isinstance(job, CompositeSamplingJob)
        assert sorted(sum(j.result().counts.values()) for j in job.jobs) == [
            200,
            1000,
            1000,
        ]

        backend = QiskitSamplingBackend(
            self.backend, circuit_converter, enable_shots_roundup=False
        )
        backend._max_shots = 1000
        backend._min_shots = 200
        job = backend.sample(QuantumCircuit(4), 2100)
        counts = job.result().counts

        #assert set(counts.keys()) in {i for i in range(16)}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 2000
        assert isinstance(job, CompositeSamplingJob)
        assert sorted(sum(j.result().counts.values()) for j in job.jobs) == [1000, 1000]
