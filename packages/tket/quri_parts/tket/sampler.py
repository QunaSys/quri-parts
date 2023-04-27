from collections.abc import MutableMapping

from pytket import Circuit  # type: ignore
from pytket.backends.backendresult import BackendResult
from pytket.extensions.qiskit import AerBackend  # type: ignore

from quri_parts.backend import SamplingCounts, SamplingJob, SamplingResult


class TKetSamplingResult(SamplingResult):
    """A result of a Qiskit sampling job."""

    def __init__(self, tket_result: BackendResult):
        if not isinstance(tket_result, BackendResult):
            raise ValueError(
                "Only pytket.backends.backendresult.BackendResult is supported"
            )
        self._tket_result = tket_result

    @property
    def counts(self) -> SamplingCounts:
        qiskit_counts = self._tket_result.get_counts()
        measurements: MutableMapping[int, int] = {}
        for result in qiskit_counts:
            result_label = int("".join(map(str, result)), 2)
            measurements[result_label] = qiskit_counts[result]
        return measurements


class TKetSamplingJob(SamplingJob):
    """A job for a Qiskit sampling measurement."""

    def __init__(self, circuit: Circuit, n_shots: int):
        circuit_ = circuit.copy().measure_all()
        self.backend = AerBackend()
        self.compiled_circ = self.backend.get_compiled_circuit(circuit_)
        self.handle = self.backend.process_circuit(self.compiled_circ, n_shots=n_shots)

    def result(self) -> SamplingResult:
        tket_result = self.backend.get_result(self.handle)
        return TKetSamplingResult(tket_result)
