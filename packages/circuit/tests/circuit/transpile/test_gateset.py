# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from quri_parts.circuit import QuantumCircuit, QuantumGate, gates
from quri_parts.circuit.transpile import (
    CliffordConversionTranspiler,
    GateSetConversionTranspiler,
    RotationConversionTranspiler,
    RX2RYRZTranspiler,
    RX2RZHTranspiler,
    RY2RXRZTranspiler,
    RY2RZHTranspiler,
    RZ2RXRYTranspiler,
)


def _gates_close(x: QuantumGate, y: QuantumGate) -> bool:
    return (
        x.name == y.name
        and x.target_indices == y.target_indices
        and x.control_indices == y.control_indices
        and np.allclose(x.params, y.params)
        and x.pauli_ids == y.pauli_ids
        and np.allclose(x.unitary_matrix, y.unitary_matrix)
    )
