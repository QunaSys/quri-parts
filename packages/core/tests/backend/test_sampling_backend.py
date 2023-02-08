# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from typing import cast
from unittest import mock

import pytest

from quri_parts.backend import (
    CompositeSamplingJob,
    SamplingCounts,
    SamplingJobState,
    TERMINAL_STATES,
    async_state,
)


def create_mock_job(
    counts: SamplingCounts = {}, state: SamplingJobState = SamplingJobState.COMPLETED
) -> mock.Mock:
    job = mock.Mock()
    result = job.result.return_value
    result.counts = counts

    async def final_state(
        raise_failure: bool = False, raise_cancel: bool = False
    ) -> SamplingJobState:
        if state in TERMINAL_STATES:
            return await async_state(state, raise_failure, raise_cancel)
        else:
            # Return a never finished Future
            return await cast(
                asyncio.Future[SamplingJobState],
                asyncio.get_running_loop().create_future(),
            )

    job.async_final_state = final_state

    return job


class TestCompositeSamplingJob:
    def test_result(self) -> None:
        counts = [
            {0: 100, 1: 200, 2: 300, 3: 400},
            {0: 200, 1: 400, 3: 100, 4: 500},
        ]
        jobs = [create_mock_job(c) for c in counts]
        composite_job = CompositeSamplingJob(jobs=jobs)
        assert composite_job.result().counts == {0: 300, 1: 600, 2: 300, 3: 500, 4: 500}

    @pytest.mark.asyncio
    async def test_async_final_state_complete(self) -> None:
        states = [SamplingJobState.COMPLETED] * 3
        jobs = [create_mock_job(state=s) for s in states]
        composite_job = CompositeSamplingJob(jobs)
        state = await composite_job.async_final_state()
        assert state == SamplingJobState.COMPLETED

    @pytest.mark.asyncio
    async def test_async_final_state_failed(self) -> None:
        states = [
            SamplingJobState.COMPLETED,
            SamplingJobState.RUNNING,
            SamplingJobState.FAILED,
        ]
        jobs = [create_mock_job(state=s) for s in states]
        composite_job = CompositeSamplingJob(jobs)
        state = await composite_job.async_final_state()
        assert state == SamplingJobState.FAILED

    @pytest.mark.asyncio
    async def test_async_final_state_cancelled(self) -> None:
        states = [
            SamplingJobState.COMPLETED,
            SamplingJobState.RUNNING,
            SamplingJobState.CANCELLED,
        ]
        jobs = [create_mock_job(state=s) for s in states]
        composite_job = CompositeSamplingJob(jobs)
        state = await composite_job.async_final_state()
        assert state == SamplingJobState.CANCELLED
