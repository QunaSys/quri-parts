# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import numpy.typing as npt
from tensornetwork import Node

_I_LIST: Sequence[Sequence[complex]] = [[1.0, 0.0], [0.0, 1.0]]
_X_LIST: Sequence[Sequence[complex]] = [[0.0, 1.0], [1.0, 0.0]]
_Y_LIST: Sequence[Sequence[complex]] = [
    [0.0, 1.0j],
    [-1.0j, 0.0],
]  # This looks transposed, but it is the correct form
_Z_LIST: Sequence[Sequence[complex]] = [[1.0, 0.0], [0.0, -1.0]]
_H_LIST: Sequence[Sequence[complex]] = [
    [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)],
    [1.0 / np.sqrt(2), -1.0 / np.sqrt(2)],
]
_S_LIST: Sequence[Sequence[complex]] = [[1.0, 0.0], [0.0, 1.0j]]
_SDAG_LIST: Sequence[Sequence[complex]] = [[1.0, 0.0], [0.0, -1.0j]]
_SQRTX_LIST: Sequence[Sequence[complex]] = [
    [0.5 + 0.5j, 0.5 - 0.5j],
    [0.5 - 0.5j, 0.5 + 0.5j],
]
_SQRTXDAG_LIST: Sequence[Sequence[complex]] = [
    [0.5 - 0.5j, 0.5 + 0.5j],
    [0.5 + 0.5j, 0.5 - 0.5j],
]
_SQRTY_LIST: Sequence[Sequence[complex]] = [
    [0.5 + 0.5j, 0.5 + 0.5j],
    [-0.5 - 0.5j, 0.5 + 0.5j],
]
_SQRTYDAG_LIST: Sequence[Sequence[complex]] = [
    [0.5 - 0.5j, -0.5 + 0.5j],
    [0.5 - 0.5j, 0.5 - 0.5j],
]
_SQRTZ_LIST: Sequence[Sequence[complex]] = [[1.0, 0.0], [0.0, 1.0j]]
_SQRTZDAG_LIST: Sequence[Sequence[complex]] = [[1.0, 0.0], [0.0, -1.0j]]
_T_LIST: Sequence[Sequence[complex]] = [[1.0, 0.0], [0.0, (1.0 + 1.0j) / np.sqrt(2)]]
_TDAG_LIST: Sequence[Sequence[complex]] = [[1.0, 0.0], [0.0, (1.0 - 1.0j) / np.sqrt(2)]]
_CNOT_LIST: Sequence[Sequence[Sequence[Sequence[complex]]]] = [
    [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 1.0], [0.0, 0.0]],
    ],
    [
        [[0.0, 0.0], [0.0, 1.0]],
        [[0.0, 0.0], [1.0, 0.0]],
    ],
]
_CZ_LIST: Sequence[Sequence[Sequence[Sequence[complex]]]] = [
    [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 1.0], [0.0, 0.0]],
    ],
    [
        [[0.0, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [0.0, -1.0]],
    ],
]
_SWAP_LIST: Sequence[Sequence[Sequence[Sequence[complex]]]] = [
    [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0]],
    ],
    [
        [[0.0, 1.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0]],
    ],
]
_TOFFOLI_LIST: Sequence[Sequence[Sequence[Sequence[Sequence[Sequence[complex]]]]]] = [
    [
        [
            [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
        ],
        [
            [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
        ],
    ],
    [
        [
            [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]],
        ],
        [
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]],
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
        ],
    ],
]


class QuantumGate(Node, ABC):  # type: ignore
    """:class:`~QuantumGate` class is a base class that wraps
    :class:`~Node`."""

    def __init__(
        self, unitary_matrix: npt.NDArray[np.complex128], name: str, backend: str
    ) -> None:
        tensor = unitary_matrix
        super().__init__(tensor, name=name, backend=backend)


class SingleQubitGate(QuantumGate, ABC):
    """:class:`~SingleQubitGate` class is a base class that facilitates one qubit
    gates."""

    def __init__(
        self, data: Sequence[Sequence[complex]], name: str, backend: str
    ) -> None:
        tensor = np.array(data, dtype=np.complex128)
        super().__init__(tensor, name=name, backend=backend)


class SingleQubitRotationGate(QuantumGate, ABC):
    """:class:`~SingleQubitRotationGate` class is a base class that facilitates
    single qubit rotation gates."""

    @abstractmethod
    def rotation(self, angles: Sequence[float]) -> npt.NDArray[np.complex128]:
        pass

    def __init__(self, angles: Sequence[float], name: str, backend: str) -> None:
        self.angles = angles
        unitary_matrix = self.rotation(angles)
        super().__init__(unitary_matrix, name=name, backend=backend)


class SingleQubitPauliRotationGate(SingleQubitRotationGate, ABC):
    """:class:`~SingleQubitPauliRotationGate` class is a base class that
    facilitates single qubit Pauli rotation gates."""

    pauli: Sequence[Sequence[complex]]

    def rotation(self, angles: Sequence[float]) -> npt.NDArray[np.complex128]:
        assert len(angles) == 1
        return np.transpose(
            np.cos(angles[0] / 2.0) * np.eye(2, dtype=np.complex128)
            - 1j * np.sin(angles[0] / 2.0) * np.array(self.pauli, dtype=np.complex128)
        )  # transpose is taken here to conform to tensor notation

    def __init__(self, angle: float, name: str, backend: str) -> None:
        super().__init__([angle], name=name, backend=backend)


class TwoQubitGate(QuantumGate, ABC):
    """:class:`~TwoQubitGate` class is a base class that facilitates two qubit
    gates."""

    def __init__(
        self,
        data: Sequence[Sequence[Sequence[Sequence[complex]]]],
        name: str,
        backend: str,
    ) -> None:
        if backend == "numpy":
            tensor = np.array(data, dtype=np.complex128)
        else:
            raise ValueError("Invalid backend selected for tensor network: ", backend)
        super().__init__(tensor, name=name, backend=backend)


class ThreeQubitGate(QuantumGate, ABC):
    """:class:`~ThreeQubitGate` class is a base class that facilitates three
    qubit gates."""

    def __init__(
        self,
        data: Sequence[Sequence[Sequence[Sequence[Sequence[Sequence[complex]]]]]],
        name: str,
        backend: str,
    ) -> None:
        if backend == "numpy":
            tensor = np.array(data, dtype=np.complex128)
        else:
            raise ValueError("Invalid backend selected for tensor network: ", backend)
        super().__init__(tensor, name=name, backend=backend)


class I(SingleQubitGate):  # noqa: E742
    def __init__(self, backend: str = "numpy") -> None:
        unitary_matrix = _I_LIST
        name = "I"

        super().__init__(unitary_matrix, name, backend)


class X(SingleQubitGate):
    def __init__(self, backend: str = "numpy") -> None:
        unitary_matrix = _X_LIST
        name = "X"

        super().__init__(unitary_matrix, name, backend)


class Y(SingleQubitGate):
    def __init__(self, backend: str = "numpy") -> None:
        unitary_matrix = _Y_LIST
        name = "Y"

        super().__init__(unitary_matrix, name, backend)


class Z(SingleQubitGate):
    def __init__(self, backend: str = "numpy") -> None:
        unitary_matrix = _Z_LIST
        name = "Z"

        super().__init__(unitary_matrix, name, backend)


class H(SingleQubitGate):
    def __init__(self, backend: str = "numpy") -> None:
        unitary_matrix = _H_LIST
        name = "H"

        super().__init__(unitary_matrix, name, backend)


class S(SingleQubitGate):
    def __init__(self, backend: str = "numpy") -> None:
        unitary_matrix = _S_LIST
        name = "S"

        super().__init__(unitary_matrix, name, backend)


class Sdag(SingleQubitGate):
    def __init__(self, backend: str = "numpy") -> None:
        unitary_matrix = _SDAG_LIST
        name = "Sdag"

        super().__init__(unitary_matrix, name, backend)


class SqrtX(SingleQubitGate):
    def __init__(self, backend: str = "numpy") -> None:
        unitary_matrix = _SQRTX_LIST
        name = "SqrtX"

        super().__init__(unitary_matrix, name, backend)


class SqrtXdag(SingleQubitGate):
    def __init__(self, backend: str = "numpy") -> None:
        unitary_matrix = _SQRTXDAG_LIST
        name = "SqrtXdag"

        super().__init__(unitary_matrix, name, backend)


class SqrtY(SingleQubitGate):
    def __init__(self, backend: str = "numpy") -> None:
        unitary_matrix = _SQRTY_LIST
        name = "SqrtY"

        super().__init__(unitary_matrix, name, backend)


class SqrtYdag(SingleQubitGate):
    def __init__(self, backend: str = "numpy") -> None:
        unitary_matrix = _SQRTYDAG_LIST
        name = "SqrtYdag"

        super().__init__(unitary_matrix, name, backend)


class SqrtZ(SingleQubitGate):
    def __init__(self, backend: str = "numpy") -> None:
        unitary_matrix = _SQRTZ_LIST
        name = "SqrtZ"

        super().__init__(unitary_matrix, name, backend)


class SqrtZdag(SingleQubitGate):
    def __init__(self, backend: str = "numpy") -> None:
        unitary_matrix = _SQRTZDAG_LIST
        name = "SqrtZdag"

        super().__init__(unitary_matrix, name, backend)


class T(SingleQubitGate):
    def __init__(self, backend: str = "numpy") -> None:
        unitary_matrix = _T_LIST
        name = "T"

        super().__init__(unitary_matrix, name, backend)


class Tdag(SingleQubitGate):
    def __init__(self, backend: str = "numpy") -> None:
        unitary_matrix = _TDAG_LIST
        name = "Tdag"

        super().__init__(unitary_matrix, name, backend)


class Rx(SingleQubitPauliRotationGate):
    pauli = _X_LIST

    def __init__(self, angle: float, backend: str = "numpy") -> None:
        name = "RX"

        super().__init__(angle, name, backend)


class Ry(SingleQubitPauliRotationGate):
    pauli = _Y_LIST

    def __init__(self, angle: float, backend: str = "numpy") -> None:
        name = "RY"

        super().__init__(angle, name, backend)


class Rz(SingleQubitPauliRotationGate):
    pauli = _Z_LIST

    def __init__(self, angle: float, backend: str = "numpy") -> None:
        name = "RZ"

        super().__init__(angle, name, backend)


class U1(SingleQubitRotationGate):
    def rotation(self, angles: Sequence[float]) -> npt.NDArray[np.complex128]:
        assert len(angles) == 1
        return np.array(
            [[1.0, 0.0], [0.0, np.exp(1j * angles[0])]], dtype=np.complex128
        )

    def __init__(self, angles: Sequence[float], backend: str = "numpy") -> None:
        name = "U1"

        super().__init__(angles, name, backend)


class U2(SingleQubitRotationGate):
    def rotation(self, angles: Sequence[float]) -> npt.NDArray[np.complex128]:
        assert len(angles) == 2
        return np.array(
            [
                [1.0 / np.sqrt(2), np.exp(1j * angles[0]) / np.sqrt(2)],
                [
                    -np.exp(-1j * angles[1]) / np.sqrt(2),
                    np.exp(1j * (angles[0] + angles[1])) / np.sqrt(2),
                ],
            ],
            dtype=np.complex128,
        )

    def __init__(self, angles: Sequence[float], backend: str = "numpy") -> None:
        name = "U2"

        super().__init__(angles, name, backend)


class U3(SingleQubitRotationGate):
    def rotation(self, angles: Sequence[float]) -> npt.NDArray[np.complex128]:
        assert len(angles) == 3
        return np.array(
            [
                [
                    np.cos(angles[0] / 2.0),
                    np.exp(1j * angles[1]) * np.sin(angles[0] / 2.0),
                ],
                [
                    -np.exp(1j * angles[2]) * np.sin(angles[0] / 2.0),
                    np.exp(1j * (angles[1] + angles[2])) * np.cos(angles[0] / 2.0),
                ],
            ],
            dtype=np.complex128,
        )

    def __init__(self, angles: Sequence[float], backend: str = "numpy") -> None:
        name = "U3"

        super().__init__(angles, name, backend)


class CNOT(TwoQubitGate):
    def __init__(self, backend: str = "numpy") -> None:
        data = _CNOT_LIST
        name = "CNOT"

        super().__init__(data, name, backend)


class CZ(TwoQubitGate):
    def __init__(self, backend: str = "numpy") -> None:
        data = _CZ_LIST
        name = "CZ"

        super().__init__(data, name, backend)


class SWAP(TwoQubitGate):
    def __init__(self, backend: str = "numpy") -> None:
        data = _SWAP_LIST
        name = "SWAP"

        super().__init__(data, name, backend)


class Toffoli(ThreeQubitGate):
    def __init__(self, backend: str = "numpy") -> None:
        data = _TOFFOLI_LIST
        name = "SWAP"

        super().__init__(data, name, backend)
