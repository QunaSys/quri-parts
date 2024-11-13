import numpy as np
from quri_parts.circuit.gates import PauliRotation
from collections.abc import Sequence
from typing import Optional, Sequence, List, Tuple

from quri_parts.circuit import (
    ImmutableLinearMappedParametricQuantumCircuit,
    LinearMappedParametricQuantumCircuit,
)

from quri_parts.algo.ansatz.two_local import (
    EntanglementPatternType,
    EntLayerMakerArg,
    RotLayerMakerArg,
    TwoLocal,
    build_entangler_map,
)


def Rxx_gate(index, theta):
    gate = PauliRotation(index, (1, 1), theta)  # X gate: pauli_ids=1
    return gate


def _add_rz_gates(circuit, arg):
    layer_index, qubit_index = arg
    theta = circuit.add_parameter(f"theta_{layer_index}_{qubit_index}")
    circuit.add_ParametricRZ_gate(qubit_index, theta)


def _add_rxx_rz_gates(circuit, arg):
    layer_index, (i, j) = arg
    circuit.add_gate(Rxx_gate((i, j), np.pi / 2))
    circuit.add_ParametricRZ_gate(i, circuit.add_parameter(f"theta_{layer_index}_{i}"))
    circuit.add_ParametricRZ_gate(j, circuit.add_parameter(f"theta_{layer_index}_{j}"))
    circuit.add_gate(Rxx_gate((i, j), -np.pi / 2))


class BrickworkStructuredAnsatz(ImmutableLinearMappedParametricQuantumCircuit):
    """Brikwork-structured ansatz.

    Ref:
    Mizuta, K., Nakagawa, Y. O., Mitarai, K., & Fujii, K. (2022).
    Local Variational Quantum Compilation of Large-Scale Hamiltonian Dynamics. P
    RX Quantum, 3(4). https://doi.org/10.1103/PRXQuantum.3.040302


    Args:
        qubit_count: Number of qubits.
        depth: Number of depths.
        rotation_indices: Qubit indices specifying on which qubits each rotation layer
          acts.
        entangler_map_seq: Qubit index pairs specifying on which qubit pairs each
          entanglement layer acts.
    """

    _add_rotation_gates = _add_rz_gates
    _add_entanglement_gates = _add_rxx_rz_gates

    def __init__(
        self,
        qubit_count: int,
        depth: int,
        rotation_indices: Optional[Sequence[int]] = None,
        entangler_map_seq: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
    ):
        if rotation_indices is None:
            rotation_indices = list(range(qubit_count))

        if entangler_map_seq is None:
            entangler_map_seq = self._build_entangler_map_seq(qubit_count, depth)

        circuit = TwoLocal(
            qubit_count,
            layer_pattern="e" * 2 * depth,
            rot_layer_maker=self.__class__._add_rotation_gates,
            ent_layer_maker=self.__class__._add_entanglement_gates,
            rotation_indices=rotation_indices,
            entangler_map_seq=entangler_map_seq,
        )
        super().__init__(circuit)
        # self.depth = depth

    def _build_entangler_map_seq(self, qubit_count, depth):
        entangler_map_seq = []

        for i in range(2 * depth):
            if i % 2 == 0:
                layer = [(j, j + 1) for j in range(0, qubit_count - 1, 2)]
            elif i % 2 == 1:
                layer = [(j, j + 1) for j in range(1, qubit_count - 1, 2)]

            entangler_map_seq.append(layer)

        return entangler_map_seq
