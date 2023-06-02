# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from quri_parts.chem.utils.excitations import excitations
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from quri_parts.openfermion.ansatz.uccsd import (
    TrotterSingletUCCSD,
    _construct_circuit,
    _construct_spin_symmetric_circuit,
)
from quri_parts.openfermion.transforms import (
    bravyi_kitaev,
    jordan_wigner,
    symmetry_conserving_bravyi_kitaev,
)
from quri_parts.openfermion.utils import (
    add_exp_excitation_gates_trotter_decomposition,
    add_exp_pauli_gates_from_linear_mapped_function,
)


class TestConstructCircuit:
    def test_construct_circuit_w_singles_trotter1(self) -> None:
        n_spin_orbitals = 4
        n_electrons = 2
        fermion_qubit_mapping = jordan_wigner
        trotter_number = 1
        use_singles = True

        circuit = _construct_circuit(
            n_spin_orbitals,
            n_electrons,
            fermion_qubit_mapping,
            trotter_number,
            use_singles,
        )
        expected_circuit = LinearMappedUnboundParametricQuantumCircuit(n_spin_orbitals)
        params = expected_circuit.add_parameters("param1", "param2", "param3")
        op_mapper = fermion_qubit_mapping.get_of_operator_mapper()
        s_excs, d_excs = excitations(n_spin_orbitals, n_electrons)
        add_exp_excitation_gates_trotter_decomposition(
            expected_circuit, d_excs, [params[-1]], op_mapper, 1 / trotter_number
        )
        add_exp_excitation_gates_trotter_decomposition(
            expected_circuit, s_excs, params[:-1], op_mapper, 1 / trotter_number
        )
        assert circuit.parameter_count == expected_circuit.parameter_count
        assert circuit._circuit.gates == expected_circuit._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
        bound_circuit = circuit.bind_parameters(param_vals)
        expected_bound_circuit = expected_circuit.bind_parameters(param_vals)
        assert bound_circuit == expected_bound_circuit

    def test_construct_circuit_wo_singles_trotter1(self) -> None:
        n_spin_orbitals = 4
        n_electrons = 2
        fermion_qubit_mapping = jordan_wigner
        trotter_number = 1
        use_singles = False

        circuit = _construct_circuit(
            n_spin_orbitals,
            n_electrons,
            fermion_qubit_mapping,
            trotter_number,
            use_singles,
        )
        expected_circuit = LinearMappedUnboundParametricQuantumCircuit(n_spin_orbitals)
        param = expected_circuit.add_parameter("param")
        op_mapper = fermion_qubit_mapping.get_of_operator_mapper()
        _, d_excs = excitations(n_spin_orbitals, n_electrons)
        add_exp_excitation_gates_trotter_decomposition(
            expected_circuit, d_excs, [param], op_mapper, 1 / trotter_number
        )
        assert circuit.parameter_count == expected_circuit.parameter_count
        assert circuit._circuit.gates == expected_circuit._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
        bound_circuit = circuit.bind_parameters(param_vals)
        expected_bound_circuit = expected_circuit.bind_parameters(param_vals)
        assert bound_circuit == expected_bound_circuit

    def test_construct_circuit_w_singles_trotter2_scbk(self) -> None:
        n_spin_orbitals = 4
        n_electrons = 2
        fermion_qubit_mapping = symmetry_conserving_bravyi_kitaev
        use_singles = True
        trotter_number = 2

        circuit = _construct_circuit(
            n_spin_orbitals,
            n_electrons,
            fermion_qubit_mapping,
            trotter_number,
            use_singles,
        )
        n_qubits = fermion_qubit_mapping.n_qubits_required(n_spin_orbitals)
        op_mapper = fermion_qubit_mapping.get_of_operator_mapper(
            n_spin_orbitals, n_electrons
        )
        expected_circuit = LinearMappedUnboundParametricQuantumCircuit(n_qubits)
        params = expected_circuit.add_parameters("param1", "param2", "param3")
        s_excs, d_excs = excitations(n_spin_orbitals, n_electrons)
        add_exp_excitation_gates_trotter_decomposition(
            expected_circuit, d_excs, [params[-1]], op_mapper, 1 / trotter_number
        )
        add_exp_excitation_gates_trotter_decomposition(
            expected_circuit, s_excs, params[:-1], op_mapper, 1 / trotter_number
        )
        add_exp_excitation_gates_trotter_decomposition(
            expected_circuit, d_excs, [params[-1]], op_mapper, 1 / trotter_number
        )
        add_exp_excitation_gates_trotter_decomposition(
            expected_circuit, s_excs, params[:-1], op_mapper, 1 / trotter_number
        )
        assert circuit.parameter_count == expected_circuit.parameter_count
        assert circuit._circuit.gates == expected_circuit._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
        bound_circuit = circuit.bind_parameters(param_vals)
        expected_bound_circuit = expected_circuit.bind_parameters(param_vals)
        assert bound_circuit == expected_bound_circuit


class TestConstructSpinSymmetricCircuit:
    def test_construct_circuit_w_singles_trotter1(self) -> None:
        n_spin_orbitals = 8
        n_electrons = 4
        fermion_qubit_mapping = jordan_wigner
        trotter_number = 1
        use_singles = True

        circuit = _construct_spin_symmetric_circuit(
            n_spin_orbitals,
            n_electrons,
            fermion_qubit_mapping,
            trotter_number,
            use_singles,
        )
        expected_circuit = LinearMappedUnboundParametricQuantumCircuit(n_spin_orbitals)
        d_0_0_2_2 = expected_circuit.add_parameter("d_0_0_2_2")
        d_0_0_2_3 = expected_circuit.add_parameter("d_0_0_2_3")
        d_0_0_3_3 = expected_circuit.add_parameter("d_0_0_3_3")
        d_0_1_2_2 = expected_circuit.add_parameter("d_0_1_2_2")
        d_0_1_2_3 = expected_circuit.add_parameter("d_0_1_2_3")
        d_0_1_3_2 = expected_circuit.add_parameter("d_0_1_3_2")
        d_0_1_3_3 = expected_circuit.add_parameter("d_0_1_3_3")
        d_1_1_2_2 = expected_circuit.add_parameter("d_1_1_2_2")
        d_1_1_2_3 = expected_circuit.add_parameter("d_1_1_2_3")
        d_1_1_3_3 = expected_circuit.add_parameter("d_1_1_3_3")
        s_0_2 = expected_circuit.add_parameter("s_0_2")
        s_0_3 = expected_circuit.add_parameter("s_0_3")
        s_1_2 = expected_circuit.add_parameter("s_1_2")
        s_1_3 = expected_circuit.add_parameter("s_1_3")

        op_mapper = fermion_qubit_mapping.get_of_operator_mapper()

        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (0, 4), {s_0_2: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (0, 6), {s_0_3: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (1, 5), {s_0_2: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (1, 7), {s_0_3: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (2, 4), {s_1_2: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (2, 6), {s_1_3: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (3, 5), {s_1_2: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (3, 7), {s_1_3: 1}, op_mapper, 1
        )

        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (0, 1, 5, 4), {d_0_0_2_2: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (0, 1, 7, 4), {d_0_0_2_3: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (0, 1, 5, 6), {d_0_0_2_3: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (0, 1, 7, 6), {d_0_0_3_3: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (0, 3, 5, 4), {d_0_1_2_2: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (0, 3, 7, 4), {d_0_1_2_3: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (0, 3, 5, 6), {d_0_1_3_2: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (0, 3, 7, 6), {d_0_1_3_3: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (2, 1, 5, 4), {d_0_1_2_2: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (2, 1, 7, 4), {d_0_1_3_2: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (2, 1, 5, 6), {d_0_1_2_3: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (2, 1, 7, 6), {d_0_1_3_3: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (2, 3, 5, 4), {d_1_1_2_2: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (2, 3, 7, 4), {d_1_1_2_3: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (2, 3, 5, 6), {d_1_1_2_3: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (2, 3, 7, 6), {d_1_1_3_3: 1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (0, 2, 6, 4), {d_0_1_2_3: 1, d_0_1_3_2: -1}, op_mapper, 1
        )
        add_exp_pauli_gates_from_linear_mapped_function(
            expected_circuit, (1, 3, 7, 5), {d_0_1_2_3: 1, d_0_1_3_2: -1}, op_mapper, 1
        )

        assert circuit.parameter_count == expected_circuit.parameter_count == 14
        assert circuit._circuit.gates == expected_circuit._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
        bound_circuit = circuit.bind_parameters(param_vals)
        expected_bound_circuit = expected_circuit.bind_parameters(param_vals)
        assert bound_circuit == expected_bound_circuit


class TestTrotterSingletUCCSD:
    def test_trotter_singlet_uccsd_w_singles_jw(self) -> None:
        n_spin_orbitals = 4
        n_electrons = 2
        trotter_number = 1
        ansatz = TrotterSingletUCCSD(
            n_spin_orbitals, n_electrons, trotter_number=trotter_number
        )
        expected_ansatz = _construct_circuit(
            n_spin_orbitals,
            n_electrons,
            jordan_wigner,
            trotter_number=trotter_number,
            use_singles=True,
        )
        assert ansatz.parameter_count == expected_ansatz.parameter_count
        assert ansatz._circuit.gates == expected_ansatz._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(ansatz.parameter_count)]
        bound_ansatz = ansatz.bind_parameters(param_vals)
        expected_bound_ansatz = expected_ansatz.bind_parameters(param_vals)
        assert bound_ansatz == expected_bound_ansatz

    def test_trotter_singlet_uccsd_wo_singles_bk(self) -> None:
        n_spin_orbitals = 4
        n_electrons = 2
        trotter_number = 1
        ansatz = TrotterSingletUCCSD(
            n_spin_orbitals,
            n_electrons,
            bravyi_kitaev,
            trotter_number=trotter_number,
            use_singles=False,
        )
        expected_ansatz = _construct_circuit(
            n_spin_orbitals,
            n_electrons,
            bravyi_kitaev,
            trotter_number=trotter_number,
            use_singles=False,
        )
        assert ansatz.parameter_count == expected_ansatz.parameter_count
        assert ansatz._circuit.gates == expected_ansatz._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(ansatz.parameter_count)]
        bound_ansatz = ansatz.bind_parameters(param_vals)
        expected_bound_ansatz = expected_ansatz.bind_parameters(param_vals)
        assert bound_ansatz == expected_bound_ansatz

    def test_trotter_singlet_uccsd_scbk_trotter2(self) -> None:
        n_spin_orbitals = 4
        n_electrons = 2
        trotter_number = 2
        ansatz = TrotterSingletUCCSD(
            n_spin_orbitals,
            n_electrons,
            symmetry_conserving_bravyi_kitaev,
            trotter_number=trotter_number,
        )
        expected_ansatz = _construct_circuit(
            n_spin_orbitals,
            n_electrons,
            symmetry_conserving_bravyi_kitaev,
            trotter_number=trotter_number,
            use_singles=True,
        )
        assert ansatz.parameter_count == expected_ansatz.parameter_count
        assert ansatz._circuit.gates == expected_ansatz._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(ansatz.parameter_count)]
        bound_ansatz = ansatz.bind_parameters(param_vals)
        expected_bound_ansatz = expected_ansatz.bind_parameters(param_vals)
        assert bound_ansatz == expected_bound_ansatz

    def test_singlet_uccsd_invalid_input(self) -> None:
        with pytest.raises(ValueError):
            TrotterSingletUCCSD(4, 3)
        with pytest.raises(ValueError):
            TrotterSingletUCCSD(4, 4)

    def test_spin_symmetric_uccsd_trotter_1(self) -> None:
        n_spin_orbitals = 8
        n_electrons = 4
        trotter_number = 1
        ansatz = TrotterSingletUCCSD(
            n_spin_orbitals,
            n_electrons,
            trotter_number=trotter_number,
            spin_symmetric=True,
        )
        expected_ansatz = _construct_spin_symmetric_circuit(
            n_spin_orbitals,
            n_electrons,
            jordan_wigner,
            trotter_number=trotter_number,
            use_singles=True,
        )
        assert ansatz.parameter_count == expected_ansatz.parameter_count
        assert ansatz._circuit.gates == expected_ansatz._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(ansatz.parameter_count)]
        bound_ansatz = ansatz.bind_parameters(param_vals)
        expected_bound_ansatz = expected_ansatz.bind_parameters(param_vals)
        assert bound_ansatz == expected_bound_ansatz

    def test_spin_symmetric_uccsd_trotter_2(self) -> None:
        n_spin_orbitals = 8
        n_electrons = 4
        trotter_number = 1
        ansatz = TrotterSingletUCCSD(
            n_spin_orbitals,
            n_electrons,
            trotter_number=trotter_number,
            spin_symmetric=True,
            fermion_qubit_mapping=bravyi_kitaev,
        )
        expected_ansatz = _construct_spin_symmetric_circuit(
            n_spin_orbitals,
            n_electrons,
            bravyi_kitaev,
            trotter_number=trotter_number,
            use_singles=True,
        )
        assert ansatz.parameter_count == expected_ansatz.parameter_count
        assert ansatz._circuit.gates == expected_ansatz._circuit.gates
        param_vals = [0.1 * (i + 1) for i in range(ansatz.parameter_count)]
        bound_ansatz = ansatz.bind_parameters(param_vals)
        expected_bound_ansatz = expected_ansatz.bind_parameters(param_vals)
        assert bound_ansatz == expected_bound_ansatz
