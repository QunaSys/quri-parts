# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "sAS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock
import pytest

from qiskit.test.reference_circuits import ReferenceCircuits
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
from qiskit_ibm_runtime.runtime_job import JobStatus, RuntimeJob

from quri_parts.backend import CompositeSamplingJob
from quri_parts.circuit import QuantumCircuit
from quri_parts.qiskit.backend import QiskitRuntimeSamplingBackend
from quri_parts.qiskit.backend.primitive import QiskitRuntimeSamplingJob

from .mock.ibm_runtime_service_mock import mock_get_backend


primitive_api = pytest.mark.skipif(
    "not config.getoption('--api')",
    reason="Running when --api tests are enabled",
)


class TestQiskitPrimitive:
    def test_fake_service_call(self) -> None:
        # Checking if fake backend works
        runtime_service = mock_get_backend("FakeVigo")
        service = runtime_service()
        with Session(service=service, backend="FakeVigo") as session:
            sampler = Sampler(session=session)
            bell = ReferenceCircuits.bell()

            bell = ReferenceCircuits.bell()
            _ = sampler.run(bell, shots=1000)

        service.run.assert_called_once()
        args_list = service.run.call_args_list
        assert len(args_list) == 1
        _, kwargs = args_list[0]
        shots = kwargs["inputs"]["run_options"]["shots"]
        assert shots == 1000

    def test_sampler_call(self) -> None:
        jjob = MagicMock(spec=RuntimeJob)

        def fake_run(program_id, inputs, **kwargs) -> RuntimeJob:  # type: ignore
            def _always_false() -> bool:
                return False

            def _status() -> JobStatus:
                return JobStatus.DONE

            jjob.status = _status
            jjob.running = _always_false

            return jjob

        runtime_service = mock_get_backend("FakeVigo")
        service = runtime_service()
        service.run = fake_run
        backend = service.backend()
        sampler = QiskitRuntimeSamplingBackend(backend=backend, service=service)

        circuit = QuantumCircuit(5)
        job = sampler.sample(circuit, 10)

        assert isinstance(job, QiskitRuntimeSamplingJob)

    def test_sampler_session(self) -> None:
        jjob = MagicMock(spec=RuntimeJob)

        def fake_run(program_id, inputs, **kwargs) -> RuntimeJob:  # type: ignore
            def _always_false() -> bool:
                return False

            def _status() -> JobStatus:
                return JobStatus.DONE

            def _job_id() -> str:
                return "aaa"

            jjob.status = _status
            jjob.running = _always_false
            jjob.job_id = _job_id

            return jjob

        runtime_service = mock_get_backend("FakeVigo")
        service = runtime_service()
        service.run = fake_run
        backend = service.backend()

        with QiskitRuntimeSamplingBackend(backend=backend, service=service) as sampler:
            circuit = QuantumCircuit(5)
            job = sampler.sample(circuit, 10)

        assert isinstance(job, QiskitRuntimeSamplingJob)

        # Checking if the session is closed
        service._api_client.close_session.assert_called_once_with("aaa")

    @primitive_api
    def test_sampler_live_simple(self) -> None:
        service = QiskitRuntimeService()
        backend = service.backend("ibmq_qasm_simulator")

        sampler = QiskitRuntimeSamplingBackend(backend=backend)

        circuit = QuantumCircuit(2)
        circuit.add_X_gate(0)
        job = sampler.sample(circuit, 10)

        assert isinstance(job, QiskitRuntimeSamplingJob)
        result = job.result()
        counts = result.counts

        expected_counts = {1: 10}
        assert counts == expected_counts

    @primitive_api
    def test_sampler_live_composite(self) -> None:
        service = QiskitRuntimeService()
        backend = service.backend("ibmq_qasm_simulator")

        sampler = QiskitRuntimeSamplingBackend(backend=backend)

        # Setting the max shots for the sake of testing
        sampler._max_shots = 5

        circuit = QuantumCircuit(2)
        circuit.add_X_gate(0)
        job = sampler.sample(circuit, 10)

        assert isinstance(job, CompositeSamplingJob)
        result = job.result()
        counts = result.counts

        expected_counts = {1: 10}
        assert counts == expected_counts

    @primitive_api
    def test_sampler_live_simple_session(self) -> None:
        service = QiskitRuntimeService()
        backend = service.backend("ibmq_qasm_simulator")

        circuit = QuantumCircuit(2)
        circuit.add_X_gate(0)
        with QiskitRuntimeSamplingBackend(backend=backend, service=service) as sampler:
            job = sampler.sample(circuit, 10)

            assert isinstance(job, QiskitRuntimeSamplingJob)
            result = job.result()
            counts = result.counts

            expected_counts = {1: 10}
            assert counts == expected_counts

    @primitive_api
    def test_sampler_live_composite_session(self) -> None:
        service = QiskitRuntimeService()
        backend = service.backend("ibmq_qasm_simulator")

        # Setting the max shots for the sake of testing

        circuit = QuantumCircuit(2)
        circuit.add_X_gate(0)

        with QiskitRuntimeSamplingBackend(backend=backend, service=service) as sampler:
            sampler._max_shots = 5
            job = sampler.sample(circuit, 10)

            assert isinstance(job, CompositeSamplingJob)
            result = job.result()
            counts = result.counts

            expected_counts = {1: 10}
            assert counts == expected_counts
