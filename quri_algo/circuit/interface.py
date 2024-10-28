# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Optional

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.transpile import CircuitTranspiler

from quri_algo.problem import ProblemT

from .utils.transpile import apply_transpiler


@dataclass
class CircuitFactory(ABC):
    qubit_count: int = field(init=False)
    transpiler: Optional[CircuitTranspiler] = field(default=None, kw_only=True)

    @apply_transpiler  # type: ignore
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> NonParametricQuantumCircuit:
        ...


@dataclass
class PartialCircuitFactory(CircuitFactory, ABC):
    @apply_transpiler  # type: ignore
    @abstractmethod
    def __call__(
        self, idx0: int, idx1: int, *args: Any, **kwargs: Any
    ) -> NonParametricQuantumCircuit:
        ...


@dataclass
class ProblemCircuitFactory(Generic[ProblemT], CircuitFactory, ABC):
    """Represents a circuit that encodes a unitary operator into a circuit."""

    encoded_problem: ProblemT
