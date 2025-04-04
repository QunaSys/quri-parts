# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence

import numpy as np

from quri_parts.circuit import QuantumGate, gate_names, gates
from quri_parts.circuit.transpile import GateKindDecomposer
from quri_parts.qiskit.circuit.gates import ECR


class CNOT2ECRTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposed CNOT gates into sequences of X,
    SqrtX, RZ, and ECR gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.CNOT]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        control, target = gate.control_indices[0], gate.target_indices[0]
        return [
            gates.X(control),
            gates.RZ(control, -np.pi / 2.0),
            gates.SqrtX(target),
            ECR(control, target),
        ]
