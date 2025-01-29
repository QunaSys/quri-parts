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

from typing import Sequence
from abc import ABC
from tensornetwork import Node

_I_LIST = [[1.0, 0.0], [0.0, 1.0]]
_X_LIST = [[0.0, 1.0], [1.0, 0.0]]
_Y_LIST = [
    [0.0, 1.0j],
    [-1.0j, 0.0],
]  # This looks transposed, but it is the correct form
_Z_LIST = [[1.0, 0.0], [0.0, -1.0]]
_H_LIST = [[1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], [1.0 / np.sqrt(2), -1.0 / np.sqrt(2)]]
_S_LIST = [[1.0, 0.0], [0.0, 1.0j]]
_SDAG_LIST = [[1.0, 0.0], [0.0, -1.0j]]
_SQRTX_LIST = [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]]
_SQRTXDAG_LIST = [[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]]
_SQRTY_LIST = [[0.5 + 0.5j, 0.5 + 0.5j], [-0.5 - 0.5j, 0.5 + 0.5j]]
_SQRTYDAG_LIST = [[0.5 - 0.5j, -0.5 + 0.5j], [0.5 - 0.5j, 0.5 - 0.5j]]
_SQRTZ_LIST = [[1.0, 0.0], [0.0, 1.0j]]
_SQRTZDAG_LIST = [[1.0, 0.0], [0.0, -1.0j]]
_T_LIST = [[1.0, 0.0], [0.0, (1.0 + 1.0j) / np.sqrt(2)]]
_TDAG_LIST = [[1.0, 0.0], [0.0, (1.0 - 1.0j) / np.sqrt(2)]]


class QuantumGate(Node, ABC):
    """:class:`~QuantumGate` class is a base class that wraps
    :class:`~Node`."""

    def __init__(self, data: Sequence[complex], name, backend):
        if backend == "numpy":
            tensor = np.array(data)
        else:
            raise ValueError("Invalid backend selected for tensor network: ", backend)
        super().__init__(tensor, name=name, backend=backend)


class I(QuantumGate):
    def __init__(self, backend="numpy"):
        data = _I_LIST
        name = "I"

        super().__init__(data, name, backend)


class X(QuantumGate):
    def __init__(self, backend="numpy"):
        data = _X_LIST
        name = "X"

        super().__init__(data, name, backend)


class Y(QuantumGate):
    def __init__(self, backend="numpy"):
        data = _Y_LIST
        name = "Y"

        super().__init__(data, name, backend)


class Z(QuantumGate):
    def __init__(self, backend="numpy"):
        data = _Z_LIST
        name = "Z"

        super().__init__(data, name, backend)


class H(QuantumGate):
    def __init__(self, backend="numpy"):
        data = _H_LIST
        name = "H"

        super().__init__(data, name, backend)


class S(QuantumGate):
    def __init__(self, backend="numpy"):
        data = _S_LIST
        name = "S"

        super().__init__(data, name, backend)


class Sdag(QuantumGate):
    def __init__(self, backend="numpy"):
        data = _SDAG_LIST
        name = "Sdag"

        super().__init__(data, name, backend)


class SqrtX(QuantumGate):
    def __init__(self, backend="numpy"):
        data = _SQRTX_LIST
        name = "SqrtX"

        super().__init__(data, name, backend)


class SqrtXdag(QuantumGate):
    def __init__(self, backend="numpy"):
        data = _SQRTXDAG_LIST
        name = "SqrtXdag"

        super().__init__(data, name, backend)


class SqrtY(QuantumGate):
    def __init__(self, backend="numpy"):
        data = _SQRTY_LIST
        name = "SqrtY"

        super().__init__(data, name, backend)


class SqrtYdag(QuantumGate):
    def __init__(self, backend="numpy"):
        data = _SQRTYDAG_LIST
        name = "SqrtYdag"

        super().__init__(data, name, backend)


class SqrtZ(QuantumGate):
    def __init__(self, backend="numpy"):
        data = _SQRTZ_LIST
        name = "SqrtZ"

        super().__init__(data, name, backend)


class SqrtZdag(QuantumGate):
    def __init__(self, backend="numpy"):
        data = _SQRTZDAG_LIST
        name = "SqrtZdag"

        super().__init__(data, name, backend)


class T(QuantumGate):
    def __init__(self, backend="numpy"):
        data = _T_LIST
        name = "T"

        super().__init__(data, name, backend)


class Tdag(QuantumGate):
    def __init__(self, backend="numpy"):
        data = _TDAG_LIST
        name = "Tdag"

        super().__init__(data, name, backend)


class Rx(Node):
    ...


class Ry(Node):
    ...


class Rz(Node):
    ...


class U1(Node):
    ...


class U2(Node):
    ...


class U3(Node):
    ...


class CNOT(Node):
    ...


class CZ(Node):
    ...


class SWAP(Node):
    ...


class Toffoli(Node):
    ...
