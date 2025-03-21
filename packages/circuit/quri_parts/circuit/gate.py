# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#: Parametric quantum gate.
#:
#: Not intended for direct use. Every gate is created through factory
#: methods. A ParametricQuantumGate object contains information of gate
#: name, control qubit, target qubit, and pauli ids.
#: Non-parametric quantum gate.
#:
#: Not intended for direct use. Every gate is created by factory
#: methods. A QuantumGate object contains information of gate name,
#: control qubit, target qubit, classical bits, parameters, and pauli
#: ids.
from quri_parts.rust.circuit.gate import ParametricQuantumGate, QuantumGate

__all__ = [
    "ParametricQuantumGate",
    "QuantumGate",
]
