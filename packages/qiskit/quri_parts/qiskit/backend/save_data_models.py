from collections.abc import MutableMapping

from pydantic.dataclasses import dataclass

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
