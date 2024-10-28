# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TypeVar

from quri_parts.core.operator import Operator

from .interface import Problem


@dataclass
class HamiltonianInput(Problem):
    """Represents an encoded Hamiltonian."""

    ...


HamiltonianT = TypeVar("HamiltonianT", bound="HamiltonianInput")


@dataclass
class QubitHamiltonianInput(HamiltonianInput):
    """Represents the Hamiltonian in its qubit Hamiltonian form."""

    qubit_hamiltonian: Operator
