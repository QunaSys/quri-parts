# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Optional, Union, cast

if TYPE_CHECKING:
    import numpy.typing as npt

from quri_parts.circuit import (
    NonParametricQuantumCircuit,
    UnboundParametricQuantumCircuitBase,
)

from . import ComputationalBasisState
from .state import GeneralCircuitQuantumState, QuantumState
from .state_parametric import ParametricCircuitQuantumState
from .state_vector import QuantumStateVector, StateVectorType
from .state_vector_parametric import ParametricQuantumStateVector


def apply_circuit(
    state: QuantumState,
    circuit: Union[NonParametricQuantumCircuit, UnboundParametricQuantumCircuitBase],
) -> QuantumState:
    if isinstance(state, GeneralCircuitQuantumState):
        combined_circuit = state.circuit * circuit
        return quantum_state(state.qubit_count, circuit=combined_circuit)
    elif isinstance(state, QuantumStateVector):
        combined_circuit = state.circuit * circuit
        return quantum_state(
            state.qubit_count, vector=state.vector, circuit=combined_circuit
        )
    elif isinstance(state, ParametricCircuitQuantumState):
        combined_circuit = (
            cast(UnboundParametricQuantumCircuitBase, state.parametric_circuit)
            * circuit
        )
        return quantum_state(state.qubit_count, circuit=combined_circuit)
    elif isinstance(state, ParametricQuantumStateVector):
        combined_circuit = (
            cast(UnboundParametricQuantumCircuitBase, state.parametric_circuit)
            * circuit
        )
        return quantum_state(
            state.qubit_count, vector=state.vector, circuit=combined_circuit
        )
    else:
        raise ValueError(f"Unsupported state type: {state}")


def quantum_state(
    n_qubits: int,
    vector: Optional[Union[StateVectorType, "npt.ArrayLike"]] = None,
    bits: int = 0,
    circuit: Optional[
        Union[NonParametricQuantumCircuit, UnboundParametricQuantumCircuitBase]
    ] = None,
) -> QuantumState:
    if vector is None:
        cb_state = ComputationalBasisState(n_qubits, bits=bits)
        if circuit is None:
            return cb_state
        else:
            comb_circuit = cb_state.circuit * circuit
            return circuit_quantum_state(n_qubits, comb_circuit)
    if bits != 0:
        raise ValueError("")
    return quantum_state_vector(n_qubits, vector, circuit)


def circuit_quantum_state(
    n_qubits: int,
    circuit: Optional[
        Union[NonParametricQuantumCircuit, UnboundParametricQuantumCircuitBase]
    ] = None,
) -> QuantumState:
    if circuit is None or isinstance(circuit, NonParametricQuantumCircuit):
        return GeneralCircuitQuantumState(n_qubits, circuit)
    else:
        return ParametricCircuitQuantumState(n_qubits, circuit)


def quantum_state_vector(
    n_qubits: int,
    vector: Union[StateVectorType, "npt.ArrayLike"],
    circuit: Optional[
        Union[NonParametricQuantumCircuit, UnboundParametricQuantumCircuitBase]
    ] = None,
) -> QuantumState:
    if circuit is None or isinstance(circuit, NonParametricQuantumCircuit):
        return QuantumStateVector(n_qubits, vector=vector, circuit=circuit)
    else:
        return ParametricQuantumStateVector(n_qubits, vector=vector, circuit=circuit)
