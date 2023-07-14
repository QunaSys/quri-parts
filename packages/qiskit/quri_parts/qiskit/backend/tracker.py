# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
from typing import TYPE_CHECKING, Sequence

from qiskit_ibm_runtime.runtime_job import JobStatus

if TYPE_CHECKING:
    from .primitive import QiskitRuntimeSamplingJob


class TrackerStatus(enum.Enum):
    Done = enum.auto()
    Exceeded = enum.auto()
    Running = enum.auto()
    Empty = enum.auto()


class Tracker:
    def __init__(self, time_limit: float) -> None:
        self._status = TrackerStatus.Empty
        self._total_time = 0.0
        self._time_limit = time_limit

        self._running_jobs: dict[
            str, QiskitRuntimeSamplingJob
        ] = {}  # The key is job id string
        self._finished_jobs: dict[
            str, QiskitRuntimeSamplingJob
        ] = {}  # The key is job id string

    @property
    def status(self) -> TrackerStatus:
        return self._status

    @property
    def total_run_time(self) -> float:
        return self._total_time

    @property
    def available_time_left(self) -> float:
        return self._time_limit - self._total_time

    @property
    def finished_jobs(self) -> Sequence["QiskitRuntimeSamplingJob"]:
        return list(self._finished_jobs.values())

    @property
    def running_jobs(self) -> Sequence["QiskitRuntimeSamplingJob"]:
        return list(self._running_jobs.values())

    def track(
        self,
    ) -> tuple[TrackerStatus, Sequence["QiskitRuntimeSamplingJob"]]:
        finished_id = []
        jobs_to_be_cancelled: list["QiskitRuntimeSamplingJob"] = []

        # Scan through running jobs to see if there are finished ones.
        for job in self._running_jobs.values():
            job_id = job._qiskit_job.job_id()
            metrics = job._qiskit_job.metrics()
            finished = job._qiskit_job.status() == JobStatus.DONE
            if finished:
                finished_id.append(job_id)
                self._finished_jobs[job_id] = job
                self._total_time += metrics["usage"]["seconds"]

        # Remove finished jobs from running_jobs
        for job_id in finished_id:
            del self._running_jobs[job_id]

        # Modify tracker status
        if self.total_run_time >= self._time_limit:
            self._status = TrackerStatus.Exceeded
            jobs_to_be_cancelled.extend(list(self._running_jobs.values()))

        else:
            if len(self._running_jobs) == 0 and len(self._finished_jobs) != 0:
                self._status = TrackerStatus.Done
            else:
                assert (
                    self._status == TrackerStatus.Running
                    or self._status == TrackerStatus.Empty
                )

        return self._status, jobs_to_be_cancelled

    def add_job_for_tracking(self, runtime_job: "QiskitRuntimeSamplingJob") -> None:
        assert (
            runtime_job._qiskit_job.job_id() not in self._running_jobs
            and runtime_job._qiskit_job.job_id() not in self._finished_jobs
        ), "This job is already submitted before."

        if self._status == TrackerStatus.Done or self._status == TrackerStatus.Empty:
            self._status = TrackerStatus.Running

        self._running_jobs[runtime_job._qiskit_job.job_id()] = runtime_job
