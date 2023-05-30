from openfermion import FermionOperator
from openfermion import jordan_wigner as of_jordan_wigner
from typing import Any
from quri_parts.chem.utils.excitations import spin_symmetric_excitations
from quri_parts.circuit import (
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuit,
    Parameter,
)
from quri_parts.openfermion.operator.conversions import operator_from_openfermion_op


class SpinSymmetricUCCSD(ImmutableLinearMappedUnboundParametricQuantumCircuit):
    def __init__(self, n_spin_orbital: int, n_fermion: int):
        circuit = self.construct_circuit(n_spin_orbital, n_fermion)
        super().__init__(circuit)

    def construct_circuit(
        self, n_spin_orbital: int, n_fermion: int
    ) -> LinearMappedUnboundParametricQuantumCircuit:
        circuit = LinearMappedUnboundParametricQuantumCircuit(n_spin_orbital)
        (
            s_params,
            s_exc_param_fn_map,
            d_params,
            d_exc_param_fn_map,
        ) = spin_symmetric_parameters(n_spin_orbital, n_fermion)

        added_parameter_map = {}

        for param_name in s_params | d_params:
            param = circuit.add_parameter(param_name)
            added_parameter_map[param_name] = param

        for exc, fnc_list in s_exc_param_fn_map.items():
            param_fnc = {added_parameter_map[name]: coeff for name, coeff in fnc_list}
            add_exp_pauli_gates(circuit, exc, param_fnc)

        for exc, fnc_list in d_exc_param_fn_map.items():
            param_fnc = {added_parameter_map[name]: coeff for name, coeff in fnc_list}
            add_exp_pauli_gates(circuit, exc, param_fnc)

        return circuit


def spin_symmetric_parameters(
    n_spin_orbitals: int, n_fermions: int
):
    s_exc, d_exc = spin_symmetric_excitations(n_spin_orbitals, n_fermions)

    s_sz_symmetric_set = set()
    s_exc_param_fn_map = {}
    for i, a in s_exc:
        if (i % 2) != (a % 2):
            continue
        param_name = f"s_{i//2}_{a//2}"
        s_sz_symmetric_set.add(param_name)
        s_exc_param_fn_map[(i, a)] = [(param_name, 1)]

    d_sz_symmetric_set = set()
    same_spin_recorder = []
    d_exc_param_fn_map = {}

    for i, j, b, a in d_exc:
        if i % 2 == j % 2 == b % 2 == a % 2:
            same_spin_recorder.append((i, j, b, a))
            continue
        """Convention: i j b^ a^ contracts with t[i, j, a, b]
        """
        if f"d_{j//2}_{i//2}_{b//2}_{a//2}" not in d_sz_symmetric_set:
            param_name = f"d_{i//2}_{j//2}_{a//2}_{b//2}"
            d_sz_symmetric_set.add(param_name)
            d_exc_param_fn_map[(i, j, b, a)] = [(param_name, 1)]
        else:
            d_exc_param_fn_map[(i, j, b, a)] = [(f"d_{j//2}_{i//2}_{b//2}_{a//2}", 1)]

    while len(same_spin_recorder) > 0:
        curr_op = same_spin_recorder.pop()
        i, j, b, a = curr_op
        spa_i, spa_j, spa_b, spa_a = str(i // 2), str(j // 2), str(b // 2), str(a // 2)
        if (m_name := f"d_{spa_i}_{spa_j}_{spa_b}_{spa_a}") in d_sz_symmetric_set and (
            p_name := f"d_{spa_i}_{spa_j}_{spa_a}_{spa_b}"
        ) in d_sz_symmetric_set:
            d_exc_param_fn_map[curr_op] = [(p_name, 1), (m_name, -1)]
        elif (
            m_name := f"d_{spa_j}_{spa_i}_{spa_a}_{spa_b}"
        ) in d_sz_symmetric_set and (
            p_name := f"d_{spa_j}_{spa_i}_{spa_b}_{spa_a}"
        ) in d_sz_symmetric_set:
            d_exc_param_fn_map[curr_op] = [(p_name, 1), (m_name, -1)]
        else:
            print(d_sz_symmetric_set, d_exc_param_fn_map)
            raise Exception

    return (
        s_sz_symmetric_set,
        s_exc_param_fn_map,
        d_sz_symmetric_set,
        d_exc_param_fn_map,
    )


def add_exp_pauli_gates(
    circuit: LinearMappedUnboundParametricQuantumCircuit,
    exc: tuple,
    param_fn: dict[Parameter, float],
) -> None:
    fermionic_generator = generate_fermionic_operator(exc)
    qubit_generator = of_jordan_wigner(fermionic_generator)
    qp_operator = operator_from_openfermion_op(qubit_generator)
    for pauli_str, op_coeff in qp_operator.items():
        qubit_idx, pauli_ids = [], []
        for idx, p in pauli_str:
            qubit_idx.append(idx)
            pauli_ids.append(p._value_)
        new_param_mapping = {
            param: -2 * op_coeff.imag * old_coeff
            for param, old_coeff in param_fn.items()
        }
        circuit.add_ParametricPauliRotation_gate(
            qubit_idx, pauli_ids, new_param_mapping
        )


def generate_fermionic_operator(exc_idx):
    if len(exc_idx) == 2:
        i, a = exc_idx
        operator = 0
        operator += FermionOperator(((a, 1), (i, 0)))
        operator -= FermionOperator(((i, 1), (a, 0)))
        return operator

    if len(exc_idx) == 4:
        i, j, b, a = exc_idx
        operator = 0
        operator += FermionOperator(((a, 1), (b, 1), (j, 0), (i, 0)))
        operator -= FermionOperator(((i, 1), (j, 1), (b, 0), (a, 0)))
        return operator
