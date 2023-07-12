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


class TrackerStatus(enum.Enum):
    Done = enum.auto()
    Exceeded = enum.auto()
    Running = enum.auto()
    Empty = enum.auto()


class Tracker:
    def __init__(self) -> None:
        # Make all attributes un-accessible
        self.__status = TrackerStatus.Empty
        self.__total_time = 0.0

    @property
    def status(self) -> TrackerStatus:
        return self.__status

    def set_status(self, new_status: TrackerStatus) -> None:
        self.__status = new_status

    @property
    def total_run_time(self) -> float:
        return self.__total_time

    def add_new_job_execution_time(self, time_increment: float) -> None:
        self.__total_time += time_increment
