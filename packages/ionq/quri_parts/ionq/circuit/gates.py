# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal

from quri_parts.circuit import QuantumGate
from quri_parts.ionq.circuit import gate_names


class GPiFactory:
    name: Literal["GPi"] = gate_names.GPi

    def __call__(self, target_index: int, phi: float) -> QuantumGate:
        return QuantumGate(
            name=self.name,
            target_indices=(target_index,),
            params=(phi,),
        )


GPi = GPiFactory()


class GPi2Factory:
    name: Literal["GPi2"] = gate_names.GPi2

    def __call__(self, target_index: int, phi: float) -> QuantumGate:
        return QuantumGate(
            name=self.name,
            target_indices=(target_index,),
            params=(phi,),
        )


GPi2 = GPi2Factory()


class XXFactory:
    name: Literal["XX"] = gate_names.XX

    def __call__(
        self, target_index0: int, target_index1: int, phi: float
    ) -> QuantumGate:
        return QuantumGate(
            name=self.name,
            target_indices=(target_index0, target_index1),
            params=(phi,),
        )


XX = XXFactory()


class MSFactory:
    name: Literal["MS"] = gate_names.MS

    def __call__(
        self, target_index0: int, target_index1: int, phi0: float, phi1: float
    ) -> QuantumGate:
        return QuantumGate(
            name=self.name,
            target_indices=(target_index0, target_index1),
            params=(phi0, phi1),
        )


MS = MSFactory()
