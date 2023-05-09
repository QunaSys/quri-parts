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

from quri_parts.circuit import (
    NonParametricQuantumCircuit,
    QuantumCircuit,
    QuantumGate,
    gate_names,
)

from .transpiler import CircuitTranspilerProtocol


class FuseRotationTranspiler(CircuitTranspilerProtocol):
    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        xs = list(circuit.gates)
        ys: list[QuantumGate] = []

        while xs:
            x = xs.pop(0)
            if not ys:
                ys.append(x)
                continue
            y = ys.pop(-1)

            if (
                x.name in [gate_names.RX, gate_names.RY, gate_names.RZ]
                and x.name == y.name
                and x.target_indices == y.target_indices
            ):
                theta = (x.params[0] + y.params[0]) % (2 * np.pi)
                ys.append(
                    QuantumGate(
                        name=x.name,
                        target_indices=x.target_indices,
                        params=[theta],
                    )
                )
            else:
                ys.extend([y, x])

        ret = QuantumCircuit(circuit.qubit_count)
        ret.extend(ys)
        return ret
