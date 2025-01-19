# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit.transpile import (
    PauliDecomposeTranspiler,
    PauliRotationDecomposeTranspiler,
    SequentialTranspiler,
)
from quri_parts.qiskit.circuit import QiskitSetTranspiler


def test_qiskit_transpiler() -> None:
    transpiler = QiskitSetTranspiler()
    assert isinstance(transpiler, SequentialTranspiler)
    assert [type(x) for x in transpiler._transpilers] == [
        PauliDecomposeTranspiler,
        PauliRotationDecomposeTranspiler,
    ]
