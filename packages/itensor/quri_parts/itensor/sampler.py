import os
from collections import Counter

import juliacall
from juliacall import Main as jl
from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.core.sampling import MeasurementCounts, Sampler

from quri_parts.itensor.circuit import convert_circuit

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
