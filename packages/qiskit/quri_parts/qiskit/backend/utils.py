# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence

from quri_parts.backend import CompositeSamplingJob, SamplingJob
from quri_parts.backend.qubit_mapping import BackendQubitMapping, QubitMappedSamplingJob


def shot_distributer(
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


def job_processor(
    jobs: Sequence[SamplingJob],
    qubit_mapping: Optional[BackendQubitMapping] = None,
) -> SamplingJob:
    if qubit_mapping is not None:
        jobs = [QubitMappedSamplingJob(job, qubit_mapping) for job in jobs]
    if len(jobs) == 1:
        return jobs[0]
    else:
        return CompositeSamplingJob(jobs)
