from collections.abc import Sequence
from typing import Optional, Tuple

import numpy as np
from quri_parts.circuit import (
    ImmutableLinearMappedParametricQuantumCircuit,
    LinearMappedParametricQuantumCircuit,
    PauliRotation,
    QuantumGate,
)

from .two_local import (
    EntanglementPatternType,
    EntLayerMakerArg,
    TwoLocal,
    build_entangler_map,
)


def Rxx_gate(target_indices: Sequence[int], angle: float) -> QuantumGate:
    gate = PauliRotation(target_indices, (1, 1), angle)
    return gate


def _add_rxx_rz_gates(
    circuit: LinearMappedParametricQuantumCircuit, arg: EntLayerMakerArg
) -> None:
    layer_index, (i, j) = arg
    theta_0, theta_1 = circuit.add_parameters(
        f"theta_{layer_index}_{j}", f"phi_{layer_index}_{j}"
    )
    circuit.add_gate(Rxx_gate((i, j), np.pi / 2))
    circuit.add_ParametricRZ_gate(i, {theta_0: 1.0, theta_1: 1.0})
    circuit.add_ParametricRZ_gate(j, {theta_0: -1.0, theta_1: 1.0})
    circuit.add_gate(Rxx_gate((i, j), -np.pi / 2))


class Z2SymmetryPreserving(ImmutableLinearMappedParametricQuantumCircuit):
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

    _add_entanglement_gates = _add_rxx_rz_gates

    def __init__(
        self,
        qubit_count: int,
        reps: int,
        entangler_map_seq: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
    ):
        if entangler_map_seq is None:
            entangler_map_seq = build_entangler_map(
                qubit_count, [EntanglementPatternType.LINEAR] * reps
            )

        circuit = TwoLocal(
            qubit_count,
            layer_pattern="e" * reps,
            rot_layer_maker=lambda c, a: None,
            ent_layer_maker=self.__class__._add_entanglement_gates,
            rotation_indices=[],
            entangler_map_seq=entangler_map_seq,
        )
        super().__init__(circuit)
        self.reps = reps
