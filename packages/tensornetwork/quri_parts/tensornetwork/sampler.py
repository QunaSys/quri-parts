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

from quri_parts.core.sampling import Sampler, MeasurementCounts
from quri_parts.core.state import GeneralCircuitQuantumState
from quri_parts.circuit import QuantumCircuit
from quri_parts.tensornetwork.state import TensorNetworkState, convert_state


def tensor_network_sample(circuit: QuantumCircuit, shots: int) -> MeasurementCounts:
    """Returns the probabilities multiplied by the specific shot count."""
    state = GeneralCircuitQuantumState(circuit.qubit_count, circuit)
    tensor_network_state = convert_state(state)
    tensor_network_state = tensor_network_state.contract()

    tensor = np.reshape(tensor_network_state._container.pop().tensor, (2**circuit.qubit_count))
    probabilities = np.abs(tensor)**2
    samples = {b: shots*probabilities[b] for b in range(2**circuit.qubit_count)}

    return samples


def create_tensornetwork_ideal_sampler() -> Sampler:
    """Returns a :class:`~Sampler` that uses TensorNetwork simulator for
    returning the probabilities multiplied by the specific shot count."""

    def sample(circuit, shots):
        return tensor_network_sample(circuit, shots)

    return sample