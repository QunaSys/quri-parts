import os
from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional

import juliacall
from juliacall import Main as jl
from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.core.sampling import ConcurrentSampler, MeasurementCounts, Sampler
from quri_parts.core.utils.concurrent import execute_concurrently

from quri_parts.itensor.circuit import convert_circuit

if TYPE_CHECKING:
    from concurrent.futures import Executor


path = os.getcwd()
library_path = os.path.join(path, "packages/itensor/quri_parts/itensor/library.jl")

jl.seval("using ITensors")
include_statement = 'include("' + library_path + '")'
print(include_statement)
jl.seval(include_statement)


def _sample(circuit: NonParametricQuantumCircuit, shots: int) -> MeasurementCounts:
    qubits = circuit.qubit_count
    s: juliacall.VectorValue = jl.siteinds("Qubit", qubits)
    psi: juliacall.AnyValue = jl.initState(s, qubits)
    qs_circuit = convert_circuit(circuit, s)
    psi = jl.apply(qs_circuit, psi)
    result: list[int] = []
    result = jl.sampling(psi, shots)
    return Counter(result)


def create_itensor_mps_sampler() -> Sampler:
    """Returns a :class:`~Sampler` that uses ITensor mps simulator for
    sampling."""
    return _sample


def _sample_sequentially(
    _: Any, circuit_shots_tuples: Iterable[tuple[NonParametricQuantumCircuit, int]]
) -> Iterable[MeasurementCounts]:
    return [_sample(circuit, shots) for circuit, shots in circuit_shots_tuples]


def _sample_concurrently(
    circuit_shots_tuples: Iterable[tuple[NonParametricQuantumCircuit, int]],
    executor: Optional["Executor"],
    concurrency: int = 1,
) -> Iterable[MeasurementCounts]:
    return execute_concurrently(
        _sample_sequentially, None, circuit_shots_tuples, executor, concurrency
    )


# Segmentation fault probably caused by juliacall
def create_itensor_mps_concurrent_sampler(
    executor: Optional["Executor"] = None, concurrency: int = 1
) -> ConcurrentSampler:
    """Returns a :class:`~ConcurrentSampler` that uses ITensor mps simulator
    for sampling."""

    def sampler(
        circuit_shots_tuples: Iterable[tuple[NonParametricQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        return _sample_concurrently(circuit_shots_tuples, executor, concurrency)

    return sampler
