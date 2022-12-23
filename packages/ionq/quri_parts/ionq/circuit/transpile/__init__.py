# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

from quri_parts.circuit.transpile import (
    CircuitTranspiler,
    RotationSetTranspiler,
    SequentialTranspiler,
)

from .ionq_native_transpiler import CNOT2RXRYXXTranspiler, IonQNativeTranspiler

#: CircuitTranspiler to transpile a QuantumCircuit into another
#: QuantumCircuit contaiing only GPi, GPi2, and MS gates.
#: Note that the converted circuit contains IonQ native gates and is only assumed
#: to be executed after conversion.
IonQSetTranspiler: Callable[[], CircuitTranspiler] = lambda: SequentialTranspiler(
    [
        RotationSetTranspiler(),
        CNOT2RXRYXXTranspiler(),
        IonQNativeTranspiler(),
    ]
)


__all__ = [
    "IonQSetTranspiler",
    "IonQNativeTranspiler",
    "CNOT2RXRYXXTranspiler",
]
