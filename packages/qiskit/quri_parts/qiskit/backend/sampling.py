# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any, Optional

# import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.providers import Job
from qiskit.providers.backend import Backend
from qiskit.providers.ibmq import IBMQBackend
from qiskit.providers.ibmq.job import IBMQJob
from qiskit.result import Result
from qiskit_aer import AerJob, AerSimulator
from qiskit_aer.backends.aerbackend import AerBackend

from quri_parts.backend import (
    BackendError,
    CompositeSamplingJob,
    MeasurementCounts,
    SamplingBackend,
    SamplingJob,
    SamplingResult,
)
from quri_parts.circuit import CircuitTranspiler, NonParametricQuantumCircuit
from quri_parts.qiskit.circuit import QiskitTranspiler, convert_circuit


class QiskitSamplingResult(SamplingResult):
    """A result of a Qiskit sampling job."""

    def __init__(self, qiskit_result: Result):
        if not isinstance(qiskit_result, Result):
            raise ValueError("Only qiskit.result.Result is supported")

        self._qiskit_result = qiskit_result

    @property
    def counts(self) -> MeasurementCounts:
        qiskit_counts = self._qiskit_result.get_counts()
        measurements: Mapping[int, int] = {}
        if qiskit_counts is None:
            raise BackendError("No valid measurement results retrieved.")
        for result in qiskit_counts:
            measurements[int(result[2:], 16)] = qiskit_counts.pop(result) \
                # type: ignore
        return measurements


class QiskitSamplingJob(SamplingJob):
    """A job for a Qiskit sampling measurement."""

    def __init__(self, qiskit_job: Job):
        self._qiskit_job = qiskit_job

    def result(self) -> SamplingResult:
        if isinstance(self._qiskit_job, IBMQJob):
            qiskit_result: Result = self._qiskit_job.results
        if isinstance(self._qiskit_job, AerJob):
            qiskit_result: Result = self._qiskit_job.result()
        return QiskitSamplingResult(qiskit_result)


class QiskitSamplingBackend(SamplingBackend):
    """A Qiskit backend for a sampling measurement.

    Args:
        backend: A Qiskit :class:`qiskit.providers.backend.Backend` \
            for circuit execution.
        circuit_converter: A function converting \
            :class:`~quri_parts.circuit.NonParametricQuantumCircuit` to \
            a Qiskit :class:`qiskit.circuit.QuantumCircuit`.
        enable_shots_roundup: If True, when a number of shots specified to \
            :meth:`~sample` is smaller than the minimum number of shots supported by \
            the device, it is rounded up to the minimum. In this case, it is possible \
            that shots more than specified are used. If it is strictly not allowed to \
            exceed the specified shot count, set this argument to False.
        run_kwargs: Additional keyword arguments for \
            :meth:`braket.devices.Device.run` method.
    """

    def __init__(
        self,
        backend: Backend,
        circuit_converter: Callable[
            [NonParametricQuantumCircuit, Optional[CircuitTranspiler]], QuantumCircuit
        ] = convert_circuit,
        circuit_transpiler: Optional[CircuitTranspiler] = None,
        enable_shots_roundup: bool = True,
        run_kwargs: Mapping[str, Any] = {},
    ):
        self._backend = backend
        self._circuit_converter = circuit_converter
        if circuit_transpiler is None:
            circuit_transpiler = QiskitTranspiler()
        self._circuit_transpiler = circuit_transpiler
        self._enable_shots_roundup = enable_shots_roundup
        self._run_kwargs = run_kwargs

        self._min_shots = 1
        self._max_shots: Optional[int] = None
        if isinstance(backend, IBMQBackend):
            max_shots = backend.configuration.max_shots
            if max_shots > 0:
                self._max_shots = max_shots
        elif isinstance(backend, AerBackend):
            if isinstance(backend, AerSimulator):
                pass
            else:
                raise BackendError("Backend not supported")

    def sample(self, circuit: NonParametricQuantumCircuit, n_shots: int) -> SamplingJob:
        if not n_shots >= 1:
            raise ValueError("n_shots should be a positive integer.")
        if self._max_shots is not None and n_shots > self._max_shots:
            shot_dist = [self._max_shots] * (n_shots // self._max_shots)
            remaining = n_shots % self._max_shots
            if remaining >= self._min_shots or self._enable_shots_roundup:
                shot_dist.append(max(remaining, self._min_shots))
        else:
            if n_shots >= self._min_shots or self._enable_shots_roundup:
                shot_dist = [max(n_shots, self._min_shots)]
            else:
                raise ValueError(
                    f"n_shots is smaller than minimum shot count ({self._min_shots}) "
                    "supported by the device. Try larger n_shots or use "
                    "enable_shots_roundup=True when creating the backend."
                )

        qiskit_circuit = self._circuit_converter(circuit, self._circuit_transpiler)
        tasks = []
        try:
            if isinstance(self._backend, AerSimulator)\
                 or isinstance(self._backend, IBMQBackend):
                for s in shot_dist:
                    tasks.append(
                        self._backend.run(qiskit_circuit, shots=s, **self._run_kwargs)
                    )
            else:
                raise BackendError("Backend not supported")

        except Exception as e:
            for t in tasks:
                try:
                    t.cancel()
                except Exception:
                    # Ignore cancel errors
                    pass
            raise BackendError("Qiskit Device.run failed") from e

        if len(tasks) == 1:
            return QiskitSamplingJob(tasks[0])
        else:
            return CompositeSamplingJob(tuple(QiskitSamplingJob(t) for t in tasks))
