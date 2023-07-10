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

from quri_parts.qiskit.backend.primitive import QiskitRuntimeSamplingJob

PRICE_PER_SECOND = 16


class TrackerStatus(enum.Enum):
    Done = enum.auto()
    Exceeded = enum.auto()
    Running = enum.auto()
    Empty = enum.auto()


class Tracker:
    def __init__(self, price_limit: float) -> None:
        # Make all attributes un-accessible
        self.__price_limit = price_limit
        self.__status = TrackerStatus.Empty
        self.__running_jobs: dict[
            str, QiskitRuntimeSamplingJob
        ] = {}  # The key is job id string
        self.__finished_jobs: dict[
            str, QiskitRuntimeSamplingJob
        ] = {}  # The key is job id string
        self.__total_time = 0.0

    @property
    def status(self):
        return self.__status

    @property
    def total_run_time(self):
        return self.__total_time

    def add_job(self, runtime_job: QiskitRuntimeSamplingJob) -> None:
        assert (
            runtime_job._qiskit_job.job_id() not in self.__running_jobs
            and runtime_job._qiskit_job.job_id() not in self.__finished_jobs
        )

        if self.status is TrackerStatus.Done or self.status is TrackerStatus.Empty:
            self.__status = TrackerStatus.Running
        else:
            # Run track before submitting.
            # If price limit is exceeded, cancel the newly submitted job
            # and throw error.
            try:
                self.track()
            except RuntimeError:
                runtime_job._qiskit_job.cancel()
                raise RuntimeError(
                    "Cannot submit this job as price limit of"
                    f"{self.__price_limit} USD is already exceeded."
                )

        self.__running_jobs[runtime_job._qiskit_job.job_id()] = runtime_job

    def track(self) -> None:
        finished_id = []
        # Scan through running jobs to see if there are finished ones.
        for job in self.__running_jobs.values():
            job_id = job._qiskit_job.job_id()
            metrics = job._qiskit_job.metrics()
            finished = metrics["timestamps"]["finished"] is not None
            if finished:
                finished_id.append(job_id)
                self.__finished_jobs[job_id] = job
                self.__total_time += metrics["usage"]["second"]

        # Remove finished jobs form running_jobs
        for job_id in finished_id:
            del self.__running_jobs[job_id]

        # Modify tracker status
        if self.__total_time * PRICE_PER_SECOND > self.__price_limit:
            self.__status = TrackerStatus.Exceeded
            for job in self.__running_jobs.values():
                job._qiskit_job.cancel()
            raise RuntimeError(f"Price limit of {self.__price_limit} USD exceeded.")

        elif len(self.__running_jobs) == 0:
            self.__status == TrackerStatus.Done

    def run_track(self):
        while self.__status is TrackerStatus.Running:
            self.track()
