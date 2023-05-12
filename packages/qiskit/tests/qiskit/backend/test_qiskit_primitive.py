# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "sAS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from qiskit_ibm_runtime import Sampler, Session
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService

def test_sampler_connection():
    service = QiskitRuntimeService()
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure_all()

    backend = service.backend('ibmq_qasm_simulator')
    with Session(service=service, backend=backend) as session:
        sampler = Sampler(session=session)
        job = sampler.run(circuits=bell)
