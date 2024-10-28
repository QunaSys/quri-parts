# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from dataclasses import dataclass
from typing import TypeVar


@dataclass
class Problem(ABC):
    """Problem to be encoded into a circuit.

    It can be a time evolution operator or the one in Proposition 2 of
    https://arxiv.org/pdf/1802.00171.
    """

    n_state_qubit: int


ProblemT = TypeVar("ProblemT", bound="Problem")
