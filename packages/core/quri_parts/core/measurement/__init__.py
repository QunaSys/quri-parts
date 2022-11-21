# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .bitwise_commuting_pauli import (
    bitwise_commuting_pauli_measurement,
    bitwise_commuting_pauli_measurement_circuit,
    bitwise_pauli_reconstructor_factory,
    individual_pauli_measurement,
)
from .interface import (
    CommutablePauliSetMeasurement,
    CommutablePauliSetMeasurementFactory,
    CommutablePauliSetMeasurementTuple,
    PauliMeasurementCircuitGeneration,
    PauliReconstructor,
    PauliReconstructorFactory,
)

#: PauliMeasurementCircuitGeneration represents a function that generates a circuit
#: (a gate list) for measuring mutually commuting Pauli operators.
PauliMeasurementCircuitGeneration = PauliMeasurementCircuitGeneration

#: PauliReconstructor represents a function that reconstructs a value of a Pauli
#:  operator from a measurement result of its measurement circuit.
PauliReconstructor = PauliReconstructor

#: PauliReconstructorFactory represents a factory function that returns
#: a :class:`~PauliReconstructor` for a given Pauli operator.
PauliReconstructorFactory = PauliReconstructorFactory

#: Represents a function that performs grouping of Pauli operators into groups of
#: commutable Pauli operators and returns measurement schemes for them.
CommutablePauliSetMeasurementFactory = CommutablePauliSetMeasurementFactory

__all__ = [
    "PauliMeasurementCircuitGeneration",
    "PauliReconstructor",
    "PauliReconstructorFactory",
    "CommutablePauliSetMeasurement",
    "CommutablePauliSetMeasurementTuple",
    "CommutablePauliSetMeasurementFactory",
    "bitwise_commuting_pauli_measurement_circuit",
    "bitwise_pauli_reconstructor_factory",
    "bitwise_commuting_pauli_measurement",
    "individual_pauli_measurement",
]
