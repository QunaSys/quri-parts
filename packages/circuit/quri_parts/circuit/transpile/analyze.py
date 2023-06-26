from collections import defaultdict
from collections.abc import Mapping, Sequence
import itertools as it

from quri_parts.circuit import NonParametricQuantumCircuit, gate_names
from quri_parts.circuit.gate_names import NonParametricGateNameType


def gate_weighted_depth(
    circuit: NonParametricQuantumCircuit,
    weights: Mapping[NonParametricGateNameType, int],
    default_weight: int = 1,
) -> int:
    qubit_depths: dict[int, int] = defaultdict()
    for gate in circuit.gates:
        qubits = tuple(gate.control_indices) + tuple(gate.target_indices)
        depth = weights.get(gate.name, default_weight) + max(
            qubit_depths[q] for q in qubits
        )
    for q in qubits:
        qubit_depths[q] = depth
    return max(qubit_depths.values()) if qubit_depths else 0


def gate_count(
    circuit: NonParametricQuantumCircuit,
    qubit_indices: Sequence[int] = (),
    target_gates: Sequence[NonParametricGateNameType] = (),
) -> int:
    count = 0
    target_qubits = set(qubit_indices)
    target_gates = set(target_gates)
    for gate in circuit.gates:
        qubits = set(tuple(gate.control_indices) + tuple(gate.target_indices))
        if gate.name in target_gates and qubits <= target_qubits:
            count += 1
    return count


def qubit_couplings(
    circuit: NonParametricQuantumCircuit,
) -> Sequence[tuple[int, int]]:
    coupling = set()
    for gate in circuit:
        qubits = tuple(gate.control_indices) + tuple(gate.target_indices)
        if len(qubits) < 2:
            continue
        coupling |= set(it.product(qubits, repeat=2))
    return list(coupling)
