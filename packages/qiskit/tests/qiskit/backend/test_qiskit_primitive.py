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

from qiskit.test.reference_circuits import ReferenceCircuits
from qiskit_ibm_runtime import Sampler, Session
from qiskit_ibm_runtime.runtime_job import JobStatus, RuntimeJob

from quri_parts.circuit import QuantumCircuit
from quri_parts.qiskit.backend import QiskitRuntimeSamplingBackend, QiskitSamplingJob

from .mock.ibm_runtime_service_mock import mock_get_backend


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

        class Result:
            def job_id(self) -> str:
                return "aaa"

            def result(self) -> RuntimeJob:
                return jjob

        def fake_run(program_id, inputs, **kwargs) -> Result:  # type: ignore
            def _always_false() -> bool:
                return False

            jjob.status = JobStatus.DONE
            jjob.running = _always_false

            return Result()

        runtime_service = mock_get_backend("FakeVigo")
        service = runtime_service()
        service.run = fake_run
        backend = service.backend()
        sampler = QiskitRuntimeSamplingBackend(backend=backend, service=service)

        circuit = QuantumCircuit(5)
        job = sampler.sample(circuit, 10)

        assert isinstance(job, QiskitSamplingJob)

    def test_sampler_session(self) -> None:
        jjob = MagicMock(spec=RuntimeJob)

        class Result:
            def job_id(self) -> str:
                return "aaa"

            def result(self) -> RuntimeJob:
                return jjob

        def fake_run(program_id, inputs, **kwargs) -> Result:  # type: ignore
            def _always_false() -> bool:
                return False

            jjob.status = JobStatus.DONE
            jjob.running = _always_false

            return Result()

        runtime_service = mock_get_backend("FakeVigo")
        service = runtime_service()
        service.run = fake_run
        backend = service.backend()

        with QiskitRuntimeSamplingBackend(backend=backend, service=service) as sampler:
            circuit = QuantumCircuit(5)
            job = sampler.sample(circuit, 10)

        assert isinstance(job, QiskitSamplingJob)

        # Checking if the session is closed
        service._api_client.close_session.assert_called_once_with("aaa")
