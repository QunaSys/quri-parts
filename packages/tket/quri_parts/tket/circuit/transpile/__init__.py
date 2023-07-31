# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping
from typing import Optional

from pytket import OpType, passes
from pytket.backends import Backend

from quri_parts.circuit import NonParametricQuantumCircuit, gate_names
from quri_parts.circuit.transpile import CircuitTranspilerProtocol
from quri_parts.tket.circuit import circuit_from_tket, convert_circuit

_qp_tket_gate_name_map: Mapping[str, OpType] = {
    gate_names.Identity: OpType.noop,
    gate_names.X: OpType.X,
    gate_names.Y: OpType.Y,
    gate_names.Z: OpType.Z,
    gate_names.H: OpType.H,
    gate_names.S: OpType.S,
    gate_names.Sdag: OpType.Sdg,
    gate_names.SqrtX: OpType.SX,
    gate_names.SqrtXdag: OpType.SXdg,
    gate_names.T: OpType.T,
    gate_names.Tdag: OpType.Tdg,
    gate_names.RX: OpType.Rx,
    gate_names.RY: OpType.Ry,
    gate_names.RZ: OpType.Rz,
    gate_names.U1: OpType.U1,
    gate_names.U2: OpType.U2,
    gate_names.U3: OpType.U3,
    gate_names.CNOT: OpType.CX,
    gate_names.CZ: OpType.CZ,
    gate_names.SWAP: OpType.SWAP,
    gate_names.TOFFOLI: OpType.CCX,
}


class TketTranspiler(CircuitTranspilerProtocol):
    def __init__(
        self,
        backend: Optional[Backend] = None,
        basis_gates: Optional[list[str]] = None,
        optimization_level: int = 2,
    ):
        self._backend = backend
        self._basis_gates = basis_gates
        if optimization_level < 0 or optimization_level > 2:
            raise ValueError("optimization_level must be 0, 1, or 2.")
        self._optimization_level = optimization_level

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        tket_circ = convert_circuit(circuit)

        if self._backend is not None:
            self._backend.default_compilation_pass(
                optimisation_level=self._optimization_level
            ).apply(tket_circ)
            return circuit_from_tket(tket_circ)

        pass_list = []
        if self._optimization_level == 1:
            pass_list.append(passes.SynthesiseTket())  # type: ignore
        elif self._optimization_level == 2:
            pass_list.append(passes.FullPeepholeOptimise())  # type: ignore

        if self._basis_gates is not None:
            pass_list.append(
                passes.auto_rebase_pass(
                    {_qp_tket_gate_name_map[name] for name in self._basis_gates}
                )
            )

        passes.SequencePass(pass_list).apply(tket_circ)  # type: ignore
        return circuit_from_tket(tket_circ)


__all__ = [
    "TketTranspiler",
]
