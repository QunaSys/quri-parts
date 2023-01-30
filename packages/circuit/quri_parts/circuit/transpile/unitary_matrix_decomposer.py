# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cmath
import math
from collections.abc import Sequence

import numpy as np

from quri_parts.circuit import QuantumGate, gate_names, gates

from .transpiler import GateDecomposer


class SingleQubitUnitaryMatrix2RYRZ(GateDecomposer):
    def is_target_gate(self, gate: QuantumGate) -> bool:
        return gate.name == gate_names.UnitaryMatrix and len(gate.target_indices) == 1

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        theta = self._su2_decompose(gate.unitary_matrix)
        # phase = math.cos(theta[0] / 2) - 1j * math.sin(theta[0] / 2)

        target = gate.target_indices[0]
        return [
            gates.RZ(target, theta[1]),
            gates.RY(target, theta[2]),
            gates.RZ(target, theta[3]),
        ]

    def _su2_decompose(
        self, v: Sequence[Sequence[complex]], eps: float = 1.0e-15
    ) -> Sequence[float]:
        # if abs(v[0][1]) < eps and abs(v[1][0]) < eps and abs(v[0][0] * v[1][1]) > 0:
        if abs(v[0][1]) < eps or abs(v[1][0]) < eps:
            theta = np.zeros(4)
            theta[0] = (1j * cmath.log(v[0][0] * v[1][1])).real
            theta[1] = (1j * cmath.log(v[0][0] / v[1][1])).real
            # theta[1] = (2j * cmath.log(v[0][0] / v[1][1])).real
            theta[2] = 0
            theta[3] = 0

            m = np.zeros(4)
            a = (cmath.phase(v[0][0]) + (theta[0] + theta[1]) / 2) / (-np.pi / 2)
            b = (cmath.phase(v[1][1]) + (theta[0] - theta[1]) / 2) / (-np.pi / 2)
            m[0] = (a + b) / 2
            m[1] = (a - b) / 2
            theta += np.pi * m

            return tuple(theta)
        # elif abs(v[0][0]) < eps and abs(v[1][1]) < eps and abs(v[0][1] * v[1][0]) > 0:
        elif abs(v[0][0]) < eps or abs(v[1][1]) < eps:
            theta = np.zeros(4)
            theta[0] = (1j * cmath.log(-v[1][0] * v[0][1])).real
            theta[1] = (1j * cmath.log(-v[1][0] / v[0][1])).real
            theta[2] = np.pi
            theta[3] = 0

            m = np.zeros(4)
            a = (cmath.phase(v[1][0]) + (theta[0] + theta[1]) / 2) / (-np.pi / 2)
            b = (cmath.phase(-v[0][1]) + (theta[0] - theta[1]) / 2) / (-np.pi / 2)
            m[0] = (a + b) / 2
            m[1] = (a - b) / 2
            theta += np.pi * m

            return tuple(theta)
        else:
            theta = np.zeros(4)
            theta[0] = (1j * cmath.log(v[0][0] * v[1][1] - v[1][0] * v[0][1])).real
            theta[1] = (0.5j * cmath.log(-v[0][0] * v[1][0] / (v[1][1] * v[0][1]))).real
            if abs(v[0][0] * v[1][1] + v[1][0] * v[0][1]) > 1:
                theta[2] = 0
            else:
                theta[2] = math.acos(abs(v[0][0] * v[1][1] + v[1][0] * v[0][1]))
            theta[3] = (0.5j * cmath.log(-v[0][0] * v[0][1] / (v[1][1] * v[1][0]))).real
            # theta[1] = (1j * cmath.log(-v[0][0] * v[1][0] / (v[1][1] * v[0][1]))).real
            # theta[2] = math.acos(0.5j * abs(v[0][0] * v[1][1] + v[1][0] * v[0][1]))
            # theta[3] = (1j * cmath.log(-v[0][0] * v[0][1] / (v[1][1] * v[1][0]))).real

            m = np.zeros(4)
            if abs(abs(v[0][0]) - math.cos(theta[2] / 2)) > abs(
                abs(v[0][0]) - math.sin(theta[2] / 2)
            ):
                theta[2] = np.pi - theta[2]
            a = (cmath.phase(v[0][0]) + (theta[0] + theta[3] + theta[1]) / 2) / (
                -np.pi / 4
            )
            b = (cmath.phase(v[1][0]) + (theta[0] - theta[3] + theta[1]) / 2) / (
                -np.pi / 4
            )
            c = (cmath.phase(-v[0][1]) + (theta[0] + theta[3] - theta[1]) / 2) / (
                -np.pi / 4
            )
            # d = (cmath.phase(v[1][1]) + (theta[0] - theta[3] - theta[1]) / 2) / (
            #     -np.pi / 4
            # )
            m[0] = (b + c) / 2
            m[1] = (a - c) / 2
            m[3] = (a - b) / 2
            theta += np.pi / 2 * m

            return tuple(theta)
