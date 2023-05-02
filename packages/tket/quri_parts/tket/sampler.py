from collections.abc import MutableMapping
from typing import Union

from pytket import Circuit  # type: ignore
from pytket.backends.backend import Backend
from pytket.backends.backendresult import BackendResult

from quri_parts.backend import SamplingCounts, SamplingJob, SamplingResult
from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.tket.circuit import convert_circuit  # type: ignore


class TKetSamplingResult(SamplingResult):
    """A result of a TKet sampling job."""

    def __init__(self, tket_result: BackendResult):
        if not isinstance(tket_result, BackendResult):
            raise ValueError(
                "Only pytket.backends.backendresult.BackendResult is supported"
            )
        self._tket_result = tket_result

    @property
    def counts(self) -> SamplingCounts:
        """Note that the qubit on the most left-hand side is the zeroth
        qubit."""
        tket_counts = self._tket_result.get_counts()
        measurements: MutableMapping[int, int] = {}
        for result in tket_counts:
            result_label = int("".join(map(str, result)), 2)
            measurements[result_label] = tket_counts[result]
        return measurements


class TKetSamplingJob(SamplingJob):
    """A job for a TKet sampling measurement."""

    def __init__(
        self,
        circuit: Union[Circuit, NonParametricQuantumCircuit],
        n_shots: int,
        backend: Backend,
    ):
        if isinstance(circuit, NonParametricQuantumCircuit):
            circuit = convert_circuit(circuit)
        circuit_ = circuit.copy().measure_all()
        self.backend = backend
        self.compiled_circ = self.backend.get_compiled_circuit(circuit_)
        self.handle = self.backend.process_circuit(self.compiled_circ, n_shots=n_shots)

    def result(self) -> SamplingResult:
        tket_result = self.backend.get_result(self.handle)
        return TKetSamplingResult(tket_result)
