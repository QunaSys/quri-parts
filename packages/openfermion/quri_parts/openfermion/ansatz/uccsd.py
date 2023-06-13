# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.chem.utils.excitations import (
    DoubleExcitation,
    SingleExcitation,
    excitations,
    spin_symmetric_excitations,
)
from quri_parts.circuit import (
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuit,
)

from ..transforms import OpenFermionQubitMapping, jordan_wigner
from ..utils import (
    add_exp_excitation_gates_trotter_decomposition,
    add_exp_pauli_gates_from_linear_mapped_function,
)
from ..utils.add_exp_excitation_gates_trotter_decomposition import _create_operator


class UCCSD(ImmutableLinearMappedUnboundParametricQuantumCircuit):
    """Unitary coupled-cluster singles and doubles (UCCSD) ansatz. The ansatz
    consists of the exponentials of single excitation and double excitation
    operator decomposed by first-order Trotter product formula. Note that the ansatz
    only supports singlet state and the occupied orbitals are the lowest
    :attr:`n_fermions` spin orbitals. The decomposition using Trotter product formula
    is executed for each qubit operators obtained by mapping excitation operators.

    Args:
        n_spin_orbitals: Number of spin orbitals.
        n_fermions: Number of fermions.
        fermion_qubit_mapping: Mapping from :class:`FermionOperator` to
          :class:`Operator`
        trotter_number: Number for first-order Trotter product formula.
        use_singles: If ``True``, single-excitation gates are applied.
        spin_symmetric: If ``True``, the ansatz will be spin symmetric.
            Parameters for the spin symmetric ansatz are named according to the spatial
            transition amplitude.
            - For single excitations, parameter named s_I_A denotes the excitation
            is from occupied spatial orbital I to virtial spatial orbital A.
            - For double excitations, parameter named d_I_J_A_B denotes the excitation
            is from occupied spin orbital (I, ↑), (J, ↓) to virtual spin orbitals
            (A, ↑), (B, ↓).

    Note:
        Spin symmetric ansatz:
            1. Certain excitation operators will share the same circuit parameters.
                - For single excitation:
                    - :math:`c_{A↑}^† c_{I↑}` and :math:`c_{A↓}^† c_{I↓}` share the same
                    transition amplitude t_I^A, thus sharing the same circuit parameter
                    t.
                - For mixed spin double excitation:
                    - :math:`c_{A↑}^† c_{B↓}^† c_{J↓} c_{I↑}` and :math:`c_{A↑}^† c_{B↓}^† c_{J↓} c_{I↑}`
                    share the same excitaion amplitude :math:`t_{I↑, J↓}^{A↑, B↓}`, thus sharing
                    the same circuit parameter d_I_J_A_B.
                    - All the circuit parameters for double excitation are fixed by
                    the mixed spin double excitation mode.
                - For same spin double excitation:
                    - :math:`c_{A↑}^† c_{B↑}^† c_{J↑} c_{I↑}` and :math:`c_{A↓}^† c_{B↓}^† c_{J↓} c_{I↓}`
                    share the same excitation amplitude :math:`t_{I↑, J↓}^{A↑, B↓} - t_{I↓, J↑}^{A↑, B↓}`,
                    where the corresponding parameters d_I_J_A_B and d_I_J_B_A are already added to the
                    circuit when constructing the mixed spin excitation modes.
            2. Parameter count:
                For a system with n_o occupied spatial orbitals and n_v virtual spatial orbitals:
                - Single excitations amount for n_o * n_v parameters
                - Double excitations amount for n_o * n_v + n_o * n_v * (n_o * n_v - 1)/2 parameters.
    """  # noqa: E501

    def __init__(
        self,
        n_spin_orbitals: int,
        n_fermions: int,
        fermion_qubit_mapping: OpenFermionQubitMapping = jordan_wigner,
        trotter_number: int = 1,
        use_singles: bool = True,
        delta_sz: int = 0,
        singlet_excitation: bool = False,
    ):
        n_vir_sorbs = n_spin_orbitals - n_fermions

        if n_fermions % 2:
            raise ValueError("Number of electrons must be even for SingletUCCSD.")

        if n_vir_sorbs <= 0:
            raise ValueError("Number of virtual orbitals must be a non-zero integer.")

        if delta_sz != 0 and singlet_excitation:
            raise ValueError("Singlet excitation is only allowed when ΔSz = 0")

        circuit = (
            _construct_singlet_excitation_circuit(
                n_spin_orbitals,
                n_fermions,
                fermion_qubit_mapping,
                trotter_number,
                use_singles,
            )
            if singlet_excitation
            else _construct_circuit(
                n_spin_orbitals,
                n_fermions,
                fermion_qubit_mapping,
                trotter_number,
                use_singles,
                delta_sz,
            )
        )

        super().__init__(circuit)


def _construct_circuit(
    n_spin_orbitals: int,
    n_fermions: int,
    fermion_qubit_mapping: OpenFermionQubitMapping,
    trotter_number: int,
    use_singles: bool,
    delta_sz: int = 0,
) -> LinearMappedUnboundParametricQuantumCircuit:
    n_qubits = fermion_qubit_mapping.n_qubits_required(n_spin_orbitals)

    s_excs, d_excs = excitations(n_spin_orbitals, n_fermions, delta_sz=delta_sz)

    circuit = LinearMappedUnboundParametricQuantumCircuit(n_qubits)
    if use_singles:
        s_exc_params = [
            circuit.add_parameter(f"theta_s_{i}") for i in range(len(s_excs))
        ]
    d_exc_params = [circuit.add_parameter(f"theta_d_{i}") for i in range(len(d_excs))]
    op_mapper = fermion_qubit_mapping.get_of_operator_mapper(
        n_spin_orbitals, n_fermions
    )
    for _ in range(trotter_number):
        add_exp_excitation_gates_trotter_decomposition(
            circuit, d_excs, d_exc_params, op_mapper, 1 / trotter_number
        )
        if use_singles:
            add_exp_excitation_gates_trotter_decomposition(
                circuit, s_excs, s_exc_params, op_mapper, 1 / trotter_number
            )

    return circuit


def _construct_singlet_excitation_circuit(
    n_spin_orbitals: int,
    n_fermions: int,
    fermion_qubit_mapping: OpenFermionQubitMapping,
    trotter_number: int,
    use_singles: bool,
) -> LinearMappedUnboundParametricQuantumCircuit:
    n_qubits = fermion_qubit_mapping.n_qubits_required(n_spin_orbitals)
    circuit = LinearMappedUnboundParametricQuantumCircuit(n_qubits)

    op_mapper = fermion_qubit_mapping.get_of_operator_mapper(
        n_spin_orbitals, n_fermions
    )

    (
        s_params,
        s_exc_param_fn_map,
        d_params,
        d_exc_param_fn_map,
    ) = singlet_excitation_parameters(n_spin_orbitals, n_fermions)

    added_parameter_map = {}

    all_param_names = s_params + d_params if use_singles else d_params

    for param_name in all_param_names:
        param = circuit.add_parameter(param_name)
        added_parameter_map[param_name] = param

    for _ in range(trotter_number):
        if use_singles:
            for exc, fnc_list in s_exc_param_fn_map.items():
                param_fnc = {
                    added_parameter_map[name]: coeff for name, coeff in fnc_list
                }
                real_operator = _create_operator(exc, op_mapper) * -1j
                add_exp_pauli_gates_from_linear_mapped_function(
                    circuit, param_fnc, real_operator, 1 / trotter_number
                )

        for d_exc, fnc_list in d_exc_param_fn_map.items():
            param_fnc = {added_parameter_map[name]: coeff for name, coeff in fnc_list}
            real_operator = _create_operator(d_exc, op_mapper) * -1j
            add_exp_pauli_gates_from_linear_mapped_function(
                circuit,
                param_fnc,
                real_operator,
                1 / trotter_number,
            )

    return circuit


def singlet_excitation_parameters(
    n_spin_orbitals: int, n_fermions: int
) -> tuple[
    list[str],
    dict[SingleExcitation, list[tuple[str, float]]],
    list[str],
    dict[DoubleExcitation, list[tuple[str, float]]],
]:
    s_exc, d_exc = spin_symmetric_excitations(n_spin_orbitals, n_fermions)

    s_sz_symmetric_set = []
    s_exc_param_fn_map = {}

    # single excitation
    for i, a in s_exc:
        if (i % 2) != (a % 2):
            continue
        param_name = f"s_{i//2}_{a//2}"
        if param_name not in s_sz_symmetric_set:
            s_sz_symmetric_set.append(param_name)
        s_exc_param_fn_map[(i, a)] = [(param_name, 1.0)]

    d_sz_symmetric_set = []
    same_spin_recorder = []
    d_exc_param_fn_map = {}

    # double excitation (mixed spin)
    for i, j, b, a in d_exc:
        if i % 2 == j % 2 == b % 2 == a % 2:
            same_spin_recorder.append((i, j, b, a))
            continue
        # Convention: i j b^ a^ contracts with t[i, j, a, b]
        if f"d_{j//2}_{i//2}_{b//2}_{a//2}" not in d_sz_symmetric_set:
            param_name = f"d_{i//2}_{j//2}_{a//2}_{b//2}"
            d_sz_symmetric_set.append(param_name)
            d_exc_param_fn_map[(i, j, b, a)] = [(param_name, 1.0)]
        else:
            d_exc_param_fn_map[(i, j, b, a)] = [(f"d_{j//2}_{i//2}_{b//2}_{a//2}", 1.0)]

    # double excitation (same spin)
    for op in same_spin_recorder:
        i, j, b, a = op
        spa_i, spa_j, spa_b, spa_a = str(i // 2), str(j // 2), str(b // 2), str(a // 2)
        if (m_name := f"d_{spa_i}_{spa_j}_{spa_b}_{spa_a}") in d_sz_symmetric_set and (
            p_name := f"d_{spa_i}_{spa_j}_{spa_a}_{spa_b}"
        ) in d_sz_symmetric_set:
            d_exc_param_fn_map[op] = [(p_name, 1.0), (m_name, -1.0)]
        elif (
            m_name := f"d_{spa_j}_{spa_i}_{spa_a}_{spa_b}"
        ) in d_sz_symmetric_set and (
            p_name := f"d_{spa_j}_{spa_i}_{spa_b}_{spa_a}"
        ) in d_sz_symmetric_set:
            d_exc_param_fn_map[op] = [(p_name, 1.0), (m_name, -1.0)]
        else:
            print(d_sz_symmetric_set, d_exc_param_fn_map)
            raise Exception

    return (
        s_sz_symmetric_set,
        s_exc_param_fn_map,
        d_sz_symmetric_set,
        d_exc_param_fn_map,
    )
