# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Union

from numpy import cfloat, float64, zeros
from numpy.typing import NDArray
from pytket.circuit import StatePreparationBox  # type: ignore
from typing_extensions import TypeAlias

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.core.state import GeneralCircuitQuantumState, QuantumStateVector
from quri_parts.tket.circuit import convert_circuit

State: TypeAlias = Union[
    GeneralCircuitQuantumState, QuantumStateVector, NDArray[cfloat]
]


def create_tket_simulator() -> (
    Callable[[NonParametricQuantumCircuit, State, str], QuantumStateVector]
):
    """Create a simulator that acts the circuit onto the state vector.

    Probably need to add an "endian" argument in the QuantumStateVector
    constructor
    """

    def simulator(
        circuit: NonParametricQuantumCircuit, init_state: State, endian: str = "little"
    ) -> QuantumStateVector:
        n_qubits = circuit.qubit_count

        if isinstance(init_state, (GeneralCircuitQuantumState, QuantumStateVector)):
            init_state_vector = evaluate_state_to_vector(init_state).vector
            return QuantumStateVector(
                n_qubits=n_qubits,
                vector=run_circuit(circuit=circuit, init_state=init_state_vector),
            )

        return QuantumStateVector(
            n_qubits=n_qubits,
            vector=run_circuit(
                circuit=circuit, init_state=init_state, input_endian=endian
            ),
        )

    return simulator


def evaluate_state_to_vector(
    state: Union[GeneralCircuitQuantumState, QuantumStateVector]
) -> QuantumStateVector:
    """Convert GeneralCircuitQuantumState or QuantumStateVector to
    QuantumStateVector that only contains the state vector."""
    n_qubits = state.qubit_count

    if isinstance(state, QuantumStateVector):
        init_state_vector = state.vector
    elif isinstance(state, GeneralCircuitQuantumState):
        init_state_vector = zeros(2**n_qubits, dtype=complex)
        init_state_vector[0] = 1.0
    else:
        raise TypeError(
            "the input state should be either a GeneralCircuitQuantumState\
             or a QuantumStateVector"
        )

    out_state_vector = run_circuit(state.circuit, init_state_vector)

    quri_vector_state = QuantumStateVector(n_qubits=n_qubits, vector=out_state_vector)
    return quri_vector_state


def _endian_convention_conversion_matrix(n_qubit: int) -> NDArray[float64]:
    """Generate the matrix to convert between different endian conventions of
    the state vector."""
    conversion_matrix = zeros((2**n_qubit, 2**n_qubit))

    for i in range(2**n_qubit):
        little_bin = bin(i)[2:].zfill(n_qubit)
        big_bin = bin(i)[2:].zfill(n_qubit)[::-1]
        conversion_matrix[int(big_bin, 2), int(little_bin, 2)] = 1.0

    return conversion_matrix


def run_circuit(
    circuit: NonParametricQuantumCircuit,
    init_state: NDArray[cfloat],
    input_endian: str = "little",
) -> NDArray[cfloat]:
    """Act a NonParametricQuantumCircuit onto a state vector and returns a new
    state vector.

    Args:
        input_endian: the endian of the input inital state vector
            "little" (default): The statevector is converted from the convention
                where the most right-hand side qubit is the zeroth qubit
            "big": The statevector is converted from the convention where the most
                left-hand side qubit is the zeroth qubit
    """

    if len(init_state) != 2**circuit.qubit_count:
        raise ValueError("Inconsistent qubit length between circuit and state")

    assert input_endian.lower() in ("little", "big")

    if input_endian == "little":
        endian_conversion_matrix = _endian_convention_conversion_matrix(
            circuit.qubit_count
        )
        init_state = endian_conversion_matrix @ init_state

    tket_circuit = convert_circuit(circuit)
    state_prep_box = StatePreparationBox(init_state)
    state_prep_circuit = state_prep_box.get_circuit()
    total_circuit = state_prep_circuit.add_circuit(
        tket_circuit.copy(), tket_circuit.qubits
    )

    new_state_vector: NDArray[cfloat] = total_circuit.get_statevector()

    if input_endian == "little":
        return endian_conversion_matrix @ new_state_vector

    return new_state_vector
