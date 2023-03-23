# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence

from quri_parts.circuit import (
    LinearMappedUnboundParametricQuantumCircuit,
    Parameter,
    ParameterOrLinearFunction,
)


def add_orbital_rotation_gate(
    circuit: LinearMappedUnboundParametricQuantumCircuit,
    qubit_indices: Sequence[int],
    param_fn: ParameterOrLinearFunction,
) -> LinearMappedUnboundParametricQuantumCircuit:
    """Add four-qubit spatial orbital rotation gate
    :math:`\\textrm{QNP}_{\\textrm{OR}}(\\phi)` that conserves the number of particles.

    Ref.
        Anselmetti et al.,
        Local, expressive, quantum-number-preserving VQE ansätze for fermionic systems,
        2021 New J. Phys. 23 113010, DOI `10.1088/1367-2630/ac2cb3
        <https://iopscience.iop.org/article/10.1088/1367-2630/ac2cb3>`_

        See also PennyLane's documentations,
        `qml.OrbitalRotation <https://docs.pennylane.ai/en/stable/code/api/
        pennylane.OrbitalRotation.html>`_

    Args:
        circuit: :class:`LinearMappedUnboundParametricQuantumCircuit` that the orbital \
        rotation gate is added to.
        phi: :attr:`circuit`\\ 's input parameter.
        coeff: Parameter coefficient.
        const: Constant for :class:`out_params` of linear mapping.
    """
    if isinstance(param_fn, Parameter):
        param_fn = {param_fn: 0.5}
    else:
        param_fn = {param: 0.5 * val for param, val in param_fn.items()}

    circuit.add_H_gate(qubit_indices[3])
    circuit.add_H_gate(qubit_indices[2])
    circuit.add_CNOT_gate(qubit_indices[3], qubit_indices[1])
    circuit.add_CNOT_gate(qubit_indices[2], qubit_indices[0])
    circuit.add_ParametricRY_gate(qubit_indices[3], param_fn)
    circuit.add_ParametricRY_gate(qubit_indices[2], param_fn)
    circuit.add_ParametricRY_gate(qubit_indices[1], param_fn)
    circuit.add_ParametricRY_gate(qubit_indices[0], param_fn)
    circuit.add_CNOT_gate(qubit_indices[3], qubit_indices[1])
    circuit.add_CNOT_gate(qubit_indices[2], qubit_indices[0])
    circuit.add_H_gate(qubit_indices[3])
    circuit.add_H_gate(qubit_indices[2])

    return circuit
