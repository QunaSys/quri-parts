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
from openfermion import FermionOperator as OpenFermionFermionOperator
from openfermion import (
    symmetry_conserving_bravyi_kitaev as of_symmetry_conserving_bravyi_kitaev,
)

from quri_parts.core.state import ComputationalBasisState
from quri_parts.openfermion.operator import (
    FermionOperator,
    operator_from_openfermion_op,
)
from quri_parts.openfermion.transforms import (
    bravyi_kitaev,
    jordan_wigner,
    symmetry_conserving_bravyi_kitaev,
)


class TestOperatorMapper:
    def test_scbk_op_mapper(self) -> None:
        n_spin_orbitals = 4
        n_fermions = 2

        const = FermionOperator("")
        f_op1 = FermionOperator("3^ 1")
        f_op2 = FermionOperator("2^ 0")
        f_op3 = FermionOperator("0^")
        f_op4 = FermionOperator("1^ 0")
        op_total = const + f_op1 + f_op2 + f_op3 + f_op4

        of_const = OpenFermionFermionOperator("")
        of_f_op1 = OpenFermionFermionOperator("3^ 1")
        of_f_op2 = OpenFermionFermionOperator("2^ 0")
        of_f_op3 = OpenFermionFermionOperator("0^")
        of_f_op4 = OpenFermionFermionOperator("1^ 0")
        of_op_total = of_const + of_f_op1 + of_f_op2 + of_f_op3 + of_f_op4

        of_op_symmetry_conserve = of_const + of_f_op1 + of_f_op2

        op_mapper = symmetry_conserving_bravyi_kitaev.get_of_operator_mapper(
            n_spin_orbitals, n_fermions
        )
        total_transformed = op_mapper(op_total)
        of_total_transformed = op_mapper(of_op_total)

        expected = operator_from_openfermion_op(
            of_symmetry_conserving_bravyi_kitaev(of_op_symmetry_conserve, 4, 2)
        )

        assert total_transformed == expected
        assert of_total_transformed == expected


class TestStateMapper:
    def test_jw_state_mapper(self) -> None:
        n_spin_orbitals = 4
        state_mapper = jordan_wigner.get_state_mapper(n_spin_orbitals)

        mapped = state_mapper([0, 1])
        assert mapped == ComputationalBasisState(n_spin_orbitals, bits=0b0011)

    def test_bk_state_mapper(self) -> None:
        n_spin_orbitals = 4
        state_mapper = bravyi_kitaev.get_state_mapper(n_spin_orbitals)

        # State transformation matrix for BK(n_spin_orbitals=4, n_fermions=2):
        # [[1, 0, 0, 0]
        #  [1, 1, 0, 0]
        #  [0, 0, 1, 0]
        #  [1, 1, 1, 1]]

        mapped = state_mapper([0, 1])
        assert mapped == ComputationalBasisState(n_spin_orbitals, bits=0b0001)

        mapped = state_mapper([0, 2])
        assert mapped == ComputationalBasisState(n_spin_orbitals, bits=0b0111)

        n_spin_orbitals = 8
        n_fermions = 4
        state_mapper = bravyi_kitaev.get_state_mapper(n_spin_orbitals, n_fermions)

        # State transformation matrix for BK(n_spin_orbitals=8, n_fermions=4):
        # [[1, 0, 0, 0, 0, 0, 0, 0]
        #  [1, 1, 0, 0, 0, 0, 0, 0]
        #  [0, 0, 1, 0, 0, 0, 0, 0]
        #  [1, 1, 1, 1, 0, 0, 0, 0]
        #  [0, 0, 0, 0, 1, 0, 0, 0]
        #  [0, 0, 0, 0, 1, 1, 0, 0]
        #  [0, 0, 0, 0, 0, 0, 1, 0]
        #  [1, 1, 1, 1, 1, 1, 1, 1]]

        mapped = state_mapper([0, 1, 2, 3])
        assert mapped == ComputationalBasisState(n_spin_orbitals, bits=0b00000101)

        mapped = state_mapper([0, 1, 4, 5])
        assert mapped == ComputationalBasisState(n_spin_orbitals, bits=0b00010001)

    def test_scbk_state_mapper(self) -> None:
        n_spin_orbitals = 4

        with pytest.raises(ValueError):
            symmetry_conserving_bravyi_kitaev.get_state_mapper(n_spin_orbitals)

        n_fermions = 2
        state_mapper = symmetry_conserving_bravyi_kitaev.get_state_mapper(
            n_spin_orbitals, n_fermions
        )

        # State transformation:
        # 1. Reorder to "all spin-up orbitals, then all spin-down orbitals"
        # 2. Apply the matrix for BK(n_spin_orbitals=4, n_fermions=2):
        # [[1, 0, 0, 0]
        #  [1, 1, 0, 0]
        #  [0, 0, 1, 0]
        #  [1, 1, 1, 1]]
        # 3. Drop 2nd and 4th qubits

        mapped = state_mapper([0, 1])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b11)

        mapped = state_mapper([0, 2])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b01)

        n_spin_orbitals = 8
        n_fermions = 4
        state_mapper = symmetry_conserving_bravyi_kitaev.get_state_mapper(
            n_spin_orbitals, n_fermions
        )

        # State transformation:
        # 1. Reorder to "all spin-up orbitals, then all spin-down orbitals"
        # 2. Apply the matrix for BK(n_spin_orbitals=8, n_fermions=4):
        # [[1, 0, 0, 0, 0, 0, 0, 0]
        #  [1, 1, 0, 0, 0, 0, 0, 0]
        #  [0, 0, 1, 0, 0, 0, 0, 0]
        #  [1, 1, 1, 1, 0, 0, 0, 0]
        #  [0, 0, 0, 0, 1, 0, 0, 0]
        #  [0, 0, 0, 0, 1, 1, 0, 0]
        #  [0, 0, 0, 0, 0, 0, 1, 0]
        #  [1, 1, 1, 1, 1, 1, 1, 1]]
        # 3. Drop 4th and 8th qubits

        mapped = state_mapper([0, 1, 2, 3])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b001001)

        mapped = state_mapper([0, 2, 4, 6])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b000101)

        # In this case, since qubit binary before dropping 2 qubits is larger than
        # 2**n_qubits-1, it is required to take bitwise-AND when creating the state
        mapped = state_mapper([0, 1, 4, 5])
        assert mapped == ComputationalBasisState(n_spin_orbitals - 2, bits=0b111111)
