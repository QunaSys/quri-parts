from typing import Optional

from qiskit.compiler import transpile
from qiskit.providers import Backend

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.transpile import CircuitTranspilerProtocol
from quri_parts.qiskit.circuit import circuit_from_qiskit, convert_circuit


class QiskitOptimizationTranspiler(CircuitTranspilerProtocol):
    def __init__(self, backend: Backend, optimization_level: Optional[int] = None):
        self._backend = backend
        self._optimization_level = optimization_level

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        qiskit_circ = convert_circuit(circuit)
        optimized_qiskit_circ = transpile(
            qiskit_circ,
            backend=self._backend,
            optimization_level=self._optimization_level,
        )
        return circuit_from_qiskit(optimized_qiskit_circ)
