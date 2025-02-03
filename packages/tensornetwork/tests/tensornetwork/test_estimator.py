# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.core.operator import pauli_label, Operator
from quri_parts.core.state import GeneralCircuitQuantumState, ComputationalBasisState
from quri_parts.circuit import QuantumCircuit
from quri_parts.tensornetwork.estimator import (
    create_tensornetwork_estimator,
)


class TestTensorNetworkEstimator:
    def test_create_tensornetwork_estimator(self) -> None:
        pauli = pauli_label("Z0 Z2 Z5")
        state = ComputationalBasisState(6, bits=0b110010)
        estimator = create_tensornetwork_estimator()
        estimate = estimator(pauli, state)
        assert estimate.value == -1
        assert estimate.error == 0

    def test_estimate_operator(self) -> None:
        operator = Operator(
            {
                pauli_label("Z0 Z2 Z5"): 0.25,
                pauli_label("Z1 Z2 Z4"): 0.5j,
            }
        )
        state = ComputationalBasisState(6, bits=0b110010)
        estimator = create_tensornetwork_estimator()
        estimate = estimator(operator, state)
        assert estimate.value == -0.25 + 0.5j
        assert estimate.error == 0

    def test_estimate_operator_with_applied_circuit(self) -> None:
        operator = Operator(
            {
                pauli_label("Z0 Z2 Z5"): 0.25,
                pauli_label("Z1 Z2 Z4"): 0.5j,
            }
        )
        estimator = create_tensornetwork_estimator()
        circuit = QuantumCircuit(6)
        circuit.add_X_gate(1)
        circuit.add_X_gate(4)
        circuit.add_X_gate(5)

        state = GeneralCircuitQuantumState(6, circuit)
        estimate = estimator(operator, state)
        assert estimate.value == -0.25 + 0.5j
        assert estimate.error == 0

