from typing import Optional

from qiskit import transpile
from qiskit.providers import Backend

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.transpile import CircuitTranspilerProtocol
from quri_parts.qiskit.circuit import circuit_from_qiskit, convert_circuit


class QiskitOptimizationTranspiler(CircuitTranspilerProtocol):
    def __init__(
        self,
        backend: Optional[Backend] = None,
        basis_gates: Optional[list[str]] = None,
        optimization_level: Optional[int] = None,
    ):
        self._backend = backend
        self._basis_gates = basis_gates
        self._optimization_level = optimization_level

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        qiskit_circ = convert_circuit(circuit)
        optimized_qiskit_circ = transpile(
            qiskit_circ,
            backend=self._backend,
            basis_gates=self._basis_gates,
            optimization_level=self._optimization_level,
        )
        return circuit_from_qiskit(optimized_qiskit_circ)


__all__ = [
    "QiskitOptimizationTranspiler",
]
