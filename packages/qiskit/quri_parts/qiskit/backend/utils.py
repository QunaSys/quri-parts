# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Mapping, Optional, Sequence

from qiskit.providers.backend import Backend, BackendV1, BackendV2

from quri_parts.backend import BackendError, SamplingJob
from quri_parts.backend.qubit_mapping import BackendQubitMapping, QubitMappedSamplingJob
from quri_parts.circuit.transpile import CircuitTranspiler, SequentialTranspiler
from quri_parts.qiskit.circuit import QiskitTranspiler


def distribute_backend_shots(
    n_shots: int,
    min_shots: int,
    max_shots: Optional[int],
    enable_shots_roundup: Optional[bool] = True,
) -> Sequence[int]:
    if max_shots is not None and n_shots > max_shots:
        shot_dist = [max_shots] * (n_shots // max_shots)
        remaining = n_shots % max_shots
        if remaining > 0:
            if remaining >= min_shots:
                shot_dist.append(remaining)
            elif enable_shots_roundup:
                shot_dist.append(min_shots)
    else:
        if n_shots >= min_shots or enable_shots_roundup:
            shot_dist = [max(n_shots, min_shots)]
        else:
            raise ValueError(
                f"n_shots is smaller than minimum shot count ({min_shots}) "
                "supported by the device. Try larger n_shots or use "
                "enable_shots_roundup=True when creating the backend."
            )
    return shot_dist


def get_backend_min_max_shot(backend: Backend) -> tuple[int, Optional[int]]:
    if isinstance(backend, BackendV1):
        max_shots = backend.configuration().max_shots
        if max_shots > 0:
            return 1, max_shots

    if not isinstance(backend, (BackendV1, BackendV2)):
        raise BackendError("Backend not supported.")

    return 1, None


def get_qubit_mapper_and_circuit_transpiler(
    qubit_mapping: Optional[Mapping[int, int]] = None,
    circuit_transpiler: Optional[CircuitTranspiler] = None,
) -> tuple[Callable[[SamplingJob], SamplingJob], CircuitTranspiler]:
    if circuit_transpiler is None:
        circuit_transpiler = QiskitTranspiler()

    if qubit_mapping:
        mapper = BackendQubitMapping(qubit_mapping)
        circuit_transpiler = SequentialTranspiler(
            [circuit_transpiler, mapper.circuit_transpiler]
        )
        composite_job_qubit_mapper: Callable[
            [SamplingJob], SamplingJob
        ] = lambda job: QubitMappedSamplingJob(
            job, mapper
        )  # noqa: E731
        return composite_job_qubit_mapper, circuit_transpiler
    else:
        simple_job_qubit_mapper: Callable[
            [SamplingJob], SamplingJob
        ] = lambda job: job  # noqa: E731
        return simple_job_qubit_mapper, circuit_transpiler
