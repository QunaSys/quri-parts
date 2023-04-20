# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    import numpy.typing as npt

from quri_parts.circuit import (
    NonParametricQuantumCircuit,
    UnboundParametricQuantumCircuitProtocol,
)

from . import ComputationalBasisState
from .state import GeneralCircuitQuantumState, QuantumState
from .state_parametric import ParametricCircuitQuantumState
from .state_vector import QuantumStateVector, StateVectorType
from .state_vector_parametric import ParametricQuantumStateVector


def apply_circuit(
    state: QuantumState,
    circuit: Optional[
        Union[NonParametricQuantumCircuit, UnboundParametricQuantumCircuitProtocol]
    ] = None,
) -> QuantumState:
    ...


def quantum_state(
    n_qubits: int,
    vec: Optional[Union[StateVectorType, "npt.ArrayLike"]] = None,
    bits: int = 0,
    circuit: Optional[
        Union[NonParametricQuantumCircuit, UnboundParametricQuantumCircuitProtocol]
    ] = None,
) -> QuantumState:
    if bits != 0:
        if circuit is None:
            return ComputationalBasisState(n_qubits, bits=bits)
        elif isinstance(circuit, NonParametricQuantumCircuit):
            ...
            # cb_state = ComputationalBasisState(n_qubits, bits=bits)
            # combined_circuit = NonParametricQuantumCircuit(n_qubits)
            # combined_circuit += cb_state.circuit
            # combined_circuit += circuit
            # return GeneralCircuitQuantumState(n_qubits, combined_circuit)
        else:
            # cb_state = ComputationalBasisState(n_qubits, bits=bits)
            # combined_circuit =
            ...
    if vec is None:
        if circuit is None or isinstance(circuit, NonParametricQuantumCircuit):
            return GeneralCircuitQuantumState(n_qubits, circuit)
        else:
            return ParametricCircuitQuantumState(n_qubits, circuit)
    else:
        if circuit is None or isinstance(circuit, NonParametricQuantumCircuit):
            return QuantumStateVector(n_qubits, vec, circuit)
        else:
            return ParametricQuantumStateVector(n_qubits, circuit, vec)


# quantum_state(3, circuit, vec)
# s = quantum_state(3, vec)
# s = apply_circuit(circuit, s)

# s = quantum_state(3, bits=0b101)
# s = apply_circuit(circuit, s)
