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

from quri_parts.chem.utils.excitations import DoubleExcitation, SingleExcitation
from quri_parts.circuit import (
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuit,
    Parameter,
)
from quri_parts.openfermion.transforms import (
    OpenFermionQubitMapping,
    OpenFermionQubitOperatorMapper,
    jordan_wigner,
)
from quri_parts.openfermion.utils import add_exp_excitation_gates_trotter_decomposition


class KUpCCGSD(ImmutableLinearMappedUnboundParametricQuantumCircuit):
    """k-unitary pair coupled-cluster generalized singles and doubles
    (k-UpCCGSD) ansatz. The ansatz consists of the exponentials of generalized
    single excitation and pair double excitation operators decomposed by first-
    order Trotter product formula repeated k times. The term "generalized"
    means excitation operators created regardless of the occupied orbitals.
    Here the double excitations are limited to 'pair' excitation, which means
    two electrons from one spatial orbital to another.

    Ref:
        Lee et al., Generalized Unitary Coupled Cluster Wave functions for Quantum
        Computation, J. Chem. Theory Comput. 15, 311–324 (2019),
        `arXiv:1810.02327 <https://arxiv.org/abs/1810.02327>`_.

        PennyLane's documentations,
        `qml.kUpCCGSD <https://docs.pennylane.ai/en/stable/
        code/api/pennylane.kUpCCGSD.html>`_

    Args:
        n_spin_orbitals: Number of spin orbitals.
        n_fermions: Number of fermions.
        k: Number of repetitions.
        fermion_qubit_mapping: Mapping from :class:`FermionOperator` to
          :class:`Operator`
        trotter_number: Number for first-order Trotter product formula.
        delta_sz: Changes of spin in the excitation.
    """

    def __init__(
        self,
        n_spin_orbitals: int,
        n_fermions: int,
        k: int = 1,
        fermion_qubit_mapping: OpenFermionQubitMapping = jordan_wigner,
        trotter_number: int = 1,
        delta_sz: int = 0,
        singlet_excitation: bool = False,
    ):
        s_excs = _generalized_single_excitations(n_spin_orbitals, delta_sz)
        d_excs = _generalized_pair_double_excitations(n_spin_orbitals)

        n_qubits = fermion_qubit_mapping.n_qubits_required(n_spin_orbitals)
        circuit = LinearMappedUnboundParametricQuantumCircuit(n_qubits)

        op_mapper = fermion_qubit_mapping.get_of_operator_mapper(
            n_spin_orbitals, n_fermions
        )

        if not singlet_excitation:
            _construct_circuit(circuit, s_excs, d_excs, k, trotter_number, op_mapper)
        else:
            assert delta_sz == 0, ValueError(
                "delta_sz should be 0 when singlet_excitation is True"
            )
            _construct_singlet_excitation_circuit(
                circuit, s_excs, d_excs, k, trotter_number, op_mapper
            )

        super().__init__(circuit)


def _construct_circuit(
    circuit: LinearMappedUnboundParametricQuantumCircuit,
    s_excs: Sequence[SingleExcitation],
    d_excs: Sequence[DoubleExcitation],
    k: int,
    trotter_number: int,
    op_mapper: OpenFermionQubitOperatorMapper,
) -> None:
    for x in range(k):
        (
            independent_s_exc,
            independent_s_exc_name,
            independent_d_exc,
            independent_d_exc_name,
        ) = get_independent_excitations(s_excs, d_excs, x)

        independent_s_exc_params = [
            circuit.add_parameter(name) for name in independent_s_exc_name
        ]
        independent_d_exc_params = [
            circuit.add_parameter(name) for name in independent_d_exc_name
        ]

        for _ in range(trotter_number):
            add_exp_excitation_gates_trotter_decomposition(
                circuit,
                independent_s_exc,
                independent_s_exc_params,
                op_mapper,
                1 / trotter_number,
            )
            add_exp_excitation_gates_trotter_decomposition(
                circuit,
                independent_d_exc,
                independent_d_exc_params,
                op_mapper,
                1 / trotter_number,
            )


def _construct_singlet_excitation_circuit(
    circuit: LinearMappedUnboundParametricQuantumCircuit,
    s_excs: Sequence[SingleExcitation],
    d_excs: Sequence[DoubleExcitation],
    k: int,
    trotter_number: int,
    op_mapper: OpenFermionQubitOperatorMapper,
) -> None:
    for x in range(k):
        (
            independent_s_exc,
            _,
            independent_d_exc,
            independent_d_exc_name,
        ) = get_independent_excitations(s_excs, d_excs, x)

        spin_symmetric_names_recorder: dict[str, Parameter] = {}
        independent_s_exc_params = []
        for s_exc in independent_s_exc:
            start, end = s_exc
            param_name = f"spatialt1_{x}_{end//2}_{start//2}"
            if param_name not in spin_symmetric_names_recorder:
                param = circuit.add_parameter(param_name)
                spin_symmetric_names_recorder[param_name] = param
            independent_s_exc_params.append(spin_symmetric_names_recorder[param_name])

        independent_d_exc_params = [
            circuit.add_parameter(name) for name in independent_d_exc_name
        ]

        for _ in range(trotter_number):
            add_exp_excitation_gates_trotter_decomposition(
                circuit,
                independent_s_exc,
                independent_s_exc_params,
                op_mapper,
                1 / trotter_number,
            )
            add_exp_excitation_gates_trotter_decomposition(
                circuit,
                independent_d_exc,
                independent_d_exc_params,
                op_mapper,
                1 / trotter_number,
            )


def get_independent_excitations(
    s_excs: Sequence[SingleExcitation], d_excs: Sequence[DoubleExcitation], x: int
) -> tuple[list[SingleExcitation], list[str], list[DoubleExcitation], list[str]]:
    independent_s_exc = []
    independent_s_exc_name = []

    for s_exc in s_excs:
        p, q = s_exc
        if (q, p) in independent_s_exc:
            # if p^ q is in s_excs, q^ p is also in s_exc.
            continue
        independent_s_exc.append(s_exc)
        independent_s_exc_name.append(f"s_{x}_{q}_{p}")

    independent_d_exc = []
    independent_d_exc_name = []

    for d_exc in d_excs:
        p, q, r, s = d_exc
        if (r, s, p, q) in independent_d_exc:
            # if q^ p^ s r is in d_excs, s^ r^ q p is also in d_excs.
            continue
        independent_d_exc.append(d_exc)
        independent_d_exc_name.append(f"d_{x}_{s}_{r}_{q}_{p}")

    return (
        independent_s_exc,
        independent_s_exc_name,
        independent_d_exc,
        independent_d_exc_name,
    )


def _generalized_single_excitations(
    n_spin_orbitals: int, delta_sz: int
) -> Sequence[SingleExcitation]:
    sz = [0.5 - (i % 2) for i in range(n_spin_orbitals)]
    s_excitations = []
    for r in range(n_spin_orbitals):
        for p in range(n_spin_orbitals):
            if sz[p] - sz[r] == delta_sz and p != r:
                s_excitations.append((r, p))
    return s_excitations


def _generalized_pair_double_excitations(
    n_spin_orbitals: int,
) -> Sequence[DoubleExcitation]:
    """FermionOperator([(r, 0), (r + 1, 0), (p, 1), (p + 1, 1)])"""
    double_excitations = [
        (r, r + 1, p, p + 1)
        for r in range(0, n_spin_orbitals - 1, 2)
        for p in range(0, n_spin_orbitals - 1, 2)
        if p != r
    ]
    return double_excitations
