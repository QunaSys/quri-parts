# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import lru_cache
from typing import Union

import numpy as np
from quri_parts.circuit import (
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuit,
    NonParametricQuantumCircuit,
    QuantumCircuit,
    inverse_circuit,
)
from quri_parts.circuit.transpile import CircuitTranspiler
from quri_parts.core.operator import (
    PAULI_IDENTITY,
    Operator,
    pauli_label,
    trotter_suzuki_decomposition,
)

from quri_algo.circuit.time_evolution.interface import TimeEvolutionCircuitFactory
from quri_algo.circuit.utils.transpile import apply_transpiler
from quri_algo.problem import QubitHamiltonianInput

from .interface import (
    ControlledTimeEvolutionCircuitFactory,
    PartialTimeEvolutionCircuitFactory,
)


def get_shifted_hamiltonian(hamiltonian: Operator, shift: int) -> Operator:
    hamiltonian_shifted = Operator()
    for op, coef in hamiltonian.items():
        if op == PAULI_IDENTITY:
            hamiltonian_shifted.add_term(PAULI_IDENTITY, coef)
            continue
        idx_id_iterable = [
            (idx + shift, id) for idx, id in zip(*op.index_and_pauli_id_list)
        ]
        hamiltonian_shifted.add_term(pauli_label(idx_id_iterable), coef)

    return hamiltonian_shifted


def get_trotter_time_evolution_operator(
    hamiltonian: Operator,
    n_state_qubits: int,
    n_trotter: int = 1,
    trotter_order: int = 1,
) -> ImmutableLinearMappedUnboundParametricQuantumCircuit:
    circuit = LinearMappedUnboundParametricQuantumCircuit(n_state_qubits)
    t = circuit.add_parameter("t")

    if trotter_order == 1:
        for _ in range(n_trotter):
            for op, coef in hamiltonian.items():
                if op == PAULI_IDENTITY:
                    continue
                circuit.add_ParametricPauliRotation_gate(
                    *op.index_and_pauli_id_list, {t: 2 * coef.real / n_trotter}
                )
        return circuit.freeze()

    assert trotter_order % 2 == 0, "Trotter order must be 1 or an even number."
    decomposition = trotter_suzuki_decomposition(hamiltonian, 1.0, trotter_order // 2)

    for _ in range(n_trotter):
        for term in decomposition:
            if term.pauli == PAULI_IDENTITY:
                continue
            circuit.add_ParametricPauliRotation_gate(
                *term.pauli.index_and_pauli_id_list,
                {t: 2 * term.coefficient.real / n_trotter},
            )
    return circuit.freeze()


class TrotterTimeEvolutionCircuitFactory(
    TimeEvolutionCircuitFactory[QubitHamiltonianInput]
):
    def __init__(
        self,
        encoded_problem: QubitHamiltonianInput,
        n_trotter: int,
        trotter_order: int = 1,
        *,
        transpiler: CircuitTranspiler | None = None,
    ):
        self.n_trotter = n_trotter
        self.encoded_problem = encoded_problem
        self.qubit_count = self.encoded_problem.n_state_qubit
        self.trotter_order = trotter_order
        self.transpiler = transpiler
        assert isinstance(self.encoded_problem, QubitHamiltonianInput)
        self._param_evo_circuit = get_trotter_time_evolution_operator(
            self.encoded_problem.qubit_hamiltonian,
            self.encoded_problem.n_state_qubit,
            self.n_trotter,
            self.trotter_order,
        )

    @apply_transpiler  # type: ignore
    def __call__(self, evolution_time: float) -> NonParametricQuantumCircuit:
        return self._param_evo_circuit.bind_parameters([evolution_time])


def get_trotter_controlled_time_evolution_operator(
    hamiltonian: Operator,
    n_state_qubits: int,
    n_trotter: int = 1,
) -> ImmutableLinearMappedUnboundParametricQuantumCircuit:
    r"""This part uses the following formulae:

    .. math::
        P_0 \otimes I + P_1 \otimes e^{i\theta P} &= e^{-i \frac{\theta}{2} Z \otimes P} e^{i \frac{\theta}{2} I \otimes P}\\
        P_0 \otimes I + P_1 \otimes e^{i\theta P} &= \text{PauliRotation}(Z \otimes P, \theta) \text{PauliRotation}(I \otimes P, -\theta)\\
        P_0 \otimes I + P_1 \otimes e^{-i \frac{c}{T} P}
        &= \text{PauliRotation}(Z \otimes P, -\frac{c}{T}) \text{PauliRotation}(I \otimes P, \frac{c}{T})
    """
    circuit = LinearMappedUnboundParametricQuantumCircuit(n_state_qubits + 1)
    t = circuit.add_parameter("t")
    hamiltonian_shifted = get_shifted_hamiltonian(hamiltonian, 1)
    for _ in range(n_trotter):
        for op, coef in hamiltonian_shifted.items():
            if op == PAULI_IDENTITY:
                circuit.add_ParametricRZ_gate(0, {t: -coef.real / n_trotter})
                continue
            circuit.add_ParametricPauliRotation_gate(
                *op.index_and_pauli_id_list, {t: coef.real / n_trotter}
            )
            extended_op = pauli_label(str(op) + f" Z{0}")
            circuit.add_ParametricPauliRotation_gate(
                *extended_op.index_and_pauli_id_list,
                {t: -coef.real / n_trotter},
            )
    return circuit.freeze()


class TrotterControlledTimeEvolutionCircuitFactory(
    ControlledTimeEvolutionCircuitFactory[QubitHamiltonianInput]
):
    def __init__(
        self,
        encoded_problem: QubitHamiltonianInput,
        n_trotter: int,
        trotter_order: int = 1,
        *,
        transpiler: CircuitTranspiler | None = None,
    ):
        self.n_trotter = n_trotter
        self.encoded_problem = encoded_problem
        self.qubit_count = encoded_problem.n_state_qubit
        self.transpiler = transpiler
        assert isinstance(self.encoded_problem, QubitHamiltonianInput)
        self._param_evo_circuit = get_trotter_controlled_time_evolution_operator(
            self.encoded_problem.qubit_hamiltonian,
            self.encoded_problem.n_state_qubit,
            self.n_trotter,
        )

    @apply_transpiler  # type: ignore
    def __call__(self, evolution_time: float) -> NonParametricQuantumCircuit:
        return self._param_evo_circuit.bind_parameters([evolution_time])


class TrotterPartialTimeEvolutionCircuitFactory(PartialTimeEvolutionCircuitFactory):
    def __init__(
        self,
        encoded_problem: QubitHamiltonianInput,
        n_trotter: int,
        *,
        transpiler: CircuitTranspiler | None = None,
    ):
        self.encoded_problem = encoded_problem
        self.n_trotter = n_trotter
        self.transpiler = transpiler

    @lru_cache(maxsize=None)
    def get_partial_time_evolution_circuit(
        self, idx0: int, idx1: int
    ) -> ImmutableLinearMappedUnboundParametricQuantumCircuit:
        local_hamiltonian_input = self.get_local_hamiltonian_input(idx0, idx1)
        return get_trotter_time_evolution_operator(
            local_hamiltonian_input.qubit_hamiltonian,
            local_hamiltonian_input.n_state_qubit,
            self.n_trotter,
        )

    @apply_transpiler  # type: ignore
    def __call__(
        self, idx0: int, idx1: int, evolution_time: float
    ) -> NonParametricQuantumCircuit:
        return self.get_partial_time_evolution_circuit(idx0, idx1).bind_parameters(
            [evolution_time]
        )

    def __hash__(self) -> int:
        return (
            self.n_trotter
        )  # The arguments to a cached function must all be hashable, including `self`


def get_evolution_trotter_step(
    time_step: float, evolution_time: float
) -> tuple[int, int]:
    assert time_step > 0, "time step must be greater than 0."
    step = np.abs(evolution_time) / time_step
    if not np.isclose(np.round(step, 12), int(step)):
        raise ValueError(
            f"Evolution time {evolution_time} is not an integer "
            f"muliple of time step {time_step}."
        )
    return int(np.round(step, 12)), int(np.sign(evolution_time))


class FixedIntervalTrotterTimeEvolution(
    TimeEvolutionCircuitFactory[QubitHamiltonianInput]
):
    def __init__(
        self,
        encoded_problem: QubitHamiltonianInput,
        time_step: float,
        trotter_order: int = 1,
        *,
        transpiler: CircuitTranspiler | None = None,
    ):
        self.encoded_problem = encoded_problem
        self.qubit_count = encoded_problem.n_state_qubit
        self.time_step = time_step
        self.transpiler = transpiler

        self._single_time_step_circuit = get_trotter_time_evolution_operator(
            encoded_problem.qubit_hamiltonian,
            encoded_problem.n_state_qubit,
            trotter_order=trotter_order,
        ).bind_parameters([time_step])

    @apply_transpiler  # type: ignore
    def __call__(self, evolution_time: float) -> NonParametricQuantumCircuit:
        n_repition, sgn = get_evolution_trotter_step(self.time_step, evolution_time)
        circuit = QuantumCircuit(self.qubit_count)
        for _ in range(n_repition):
            circuit.extend(self._single_time_step_circuit)
        return circuit if sgn > 0 else inverse_circuit(circuit)
