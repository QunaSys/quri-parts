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

from qulacs import QuantumGateBase
from qulacs.gate import CPTP, DenseMatrix, Identity, Pauli, Probabilistic

from quri_parts.circuit.noise import AbstractKrausNoise, PauliNoise, ProbabilisticNoise


def create_kraus_gate(
    qubits: Sequence[int], noise: AbstractKrausNoise
) -> QuantumGateBase:
    return CPTP([DenseMatrix(qubits, kraus) for kraus in noise.kraus_operators])


def create_pauli_noise_gate(
    qubits: Sequence[int], noise: PauliNoise
) -> QuantumGateBase:
    qubits = noise.qubit_indices
    pauli_list, prob_list = noise.pauli_list, list(noise.prob_list)

    pauli_gates = [Pauli(qubits, pauli) for pauli in pauli_list]
    if sum(prob_list) < 1.0:
        pauli_gates.append(Identity(len(qubits)))
        prob_identity = 1.0 - sum(prob_list)
        prob_list.append(prob_identity)

    return Probabilistic(prob_list, pauli_gates)


def create_probabilistic_gate(
    qubits: Sequence[int], noise: ProbabilisticNoise
) -> QuantumGateBase:
    dense_matrices = [DenseMatrix(qubits, matrix) for matrix in noise.gate_matrices]
    return Probabilistic(noise.prob_list, dense_matrices)
