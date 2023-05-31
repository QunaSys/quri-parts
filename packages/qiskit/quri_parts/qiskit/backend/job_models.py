import json
from collections.abc import MutableMapping
from typing import Sequence

from pydantic.dataclasses import dataclass
from pydantic.json import pydantic_encoder

from quri_parts.backend import SamplingCounts, SamplingJob, SamplingResult


@dataclass
class QiskitSavedDataSamplingResult(SamplingResult):
    """Raw_data takes in saved raw data in `qiskit_result.get_counts()`
    format."""

    raw_data: dict[str, int]

    @property
    def counts(self) -> SamplingCounts:
        """Convert the raw data to format that conforms to rest of the
        codes."""
        measurements: MutableMapping[int, int] = {}
        for result in self.raw_data:
            measurements[int(result, 2)] = self.raw_data[result]
        return measurements


@dataclass
class QiskitSavedDataSamplingJob(SamplingJob):
    circuit_str: str
    n_shots: int
    saved_result: QiskitSavedDataSamplingResult

    def result(self) -> QiskitSavedDataSamplingResult:
        return self.saved_result


def convert_saved_jobs_sequence_to_str(
    saved_data_seq: Sequence[QiskitSavedDataSamplingJob],
) -> str:
    return json.dumps(saved_data_seq, default=pydantic_encoder)
