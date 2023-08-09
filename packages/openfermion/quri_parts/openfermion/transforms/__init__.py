# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from collections.abc import Collection, Sequence
from typing import Callable, Optional, Protocol, Union

from openfermion.ops import FermionOperator, InteractionOperator, MajoranaOperator
from openfermion.transforms import bravyi_kitaev as of_bravyi_kitaev
from openfermion.transforms import get_fermion_operator
from openfermion.transforms import jordan_wigner as of_jordan_wigner
from openfermion.transforms import (
    symmetry_conserving_bravyi_kitaev as of_symmetry_conserving_bravyi_kitaev,
)
from typing_extensions import TypeAlias

from quri_parts.chem.transforms import (
    BravyiKitaev,
    FermionQubitMapping,
    FermionQubitStateMapper,
    JordanWigner,
    QubitFermionStateMapper,
    SymmetryConservingBravyiKitaev,
)
from quri_parts.core.operator import Operator, SinglePauli
from quri_parts.core.state import ComputationalBasisState
from quri_parts.core.utils.binary_field import BinaryArray, BinaryMatrix, inverse
from quri_parts.openfermion.operator import (
    has_particle_number_symmetry,
    operator_from_openfermion_op,
)

#: Interface for a function that maps a :class:`openfermion.ops.FermionOperator`,
#: :class:`openfermion.ops.InteractionOperator` or
#: :class:`openfermion.ops.MajoranaOperator` to
#: a :class:`Operator`.
OpenFermionQubitOperatorMapper: TypeAlias = Callable[
    [Union["FermionOperator", "InteractionOperator", "MajoranaOperator"]],
    "Operator",
]


def _inv_state_transformation_matrix(
    op_mapper: OpenFermionQubitOperatorMapper, n_spin_orbitals: int
) -> tuple[BinaryMatrix, Sequence[int]]:
    """Build a matrix that converts vector of fermion occupancy to binary
    representation on qubit.

    It is assumed that mapping method maps the occupancy to Z operator.
    For example, the matrix is (29) in http://arxiv.org/abs/1208.5986
    for the Bravyi-Kitaev method.
    """
    mat = []
    signs = []
    for number_op in (
        (1 - 2 * FermionOperator(f"{i}^ {i}")) for i in range(n_spin_orbitals)
    ):
        op = op_mapper(number_op)
        row = list(0 for _ in range(n_spin_orbitals))
        if len(op) > 1:
            raise ValueError(
                "This method is incompatible with a mapping that converts a number "
                "operator to a sum of multiple Pauli terms."
            )
        pauli_label, coef = next(iter(op.items()))

        assert coef == 1 or coef == -1
        signs.append(int(coef.real))

        for idx, pauli in pauli_label:
            if pauli != SinglePauli.Z:
                raise ValueError("The action must be Z.")
            row[idx] = 1
        mat.append(row)
    return BinaryMatrix(mat), signs


class OpenFermionQubitMapping(FermionQubitMapping, Protocol):
    """Mapping from Fermionic operators and states to :class:`Operator`s and
    states using OpenFermion."""

    @abstractmethod
    def get_of_operator_mapper(
        self, n_spin_orbitals: Optional[int] = None, n_fermions: Optional[int] = None
    ) -> OpenFermionQubitOperatorMapper:
        """Returns a function that maps an OpenFermion
        :class:`~openfermion.ops.FermionOperator`,
        :class:`~openfermion.ops.InteractionOperator` or
        :class:`~openfermion.ops.MajoranaOperator`
        to a :class:`openfermion.ops.QubitOperator`.

        Args:
            n_spin_orbitals:
                The number of spin orbitals to be mapped to qubits. Some mappings
                require this argument (e.g.
                :class:`~OpenFermionSymmetryConservingBravyiKitaev`) while others
                calculate it automatically when omitted.
            n_fermions:
                When specified, restrict the mapping to a subspace spanned by states
                containing the fixed number of Fermions. Some mappings require this
                argument (e.g. :class:`~OpenFermionSymmetryConservingBravyiKitaev`)
                while the others ignore it.
        """
        ...

    def get_state_mapper(
        self, n_spin_orbitals: int, n_fermions: Optional[int] = None
    ) -> FermionQubitStateMapper:
        inv_trans_mat, signs = _inv_state_transformation_matrix(
            self.get_of_operator_mapper(n_spin_orbitals, n_fermions),
            n_spin_orbitals,
        )
        trans_mat = inverse(inv_trans_mat)
        n_qubits = self.n_qubits_required(n_spin_orbitals)

        def mapper(
            occupied_indices: Collection[int],
        ) -> ComputationalBasisState:
            occ_list = [(i in occupied_indices) for i in range(n_spin_orbitals)]
            occ_list = [not b if signs[i] == -1 else b for i, b in enumerate(occ_list)]
            occupancy_vector = BinaryArray(occ_list)
            qubit_vector = trans_mat @ occupancy_vector
            return ComputationalBasisState(
                n_qubits=n_qubits, bits=qubit_vector.binary & (2**n_qubits - 1)
            )

        return mapper

    def get_inv_state_mapper(
        self,
        n_spin_orbitals: int,
        n_fermions: Optional[int] = None,
        n_up_spins: Optional[int] = None,
    ) -> QubitFermionStateMapper:
        inv_trans_mat, signs = _inv_state_transformation_matrix(
            self.get_of_operator_mapper(n_spin_orbitals, n_fermions),
            n_spin_orbitals,
        )
        n_qubits = self.n_qubits_required(n_spin_orbitals)

        def mapper(state: ComputationalBasisState) -> Collection[int]:
            bits = state.bits
            bit_array = [(bits & 1 << index) >> index for index in range(n_qubits)]
            bit_array = self._augment_dropped_bits(
                bit_array, n_spin_orbitals, n_fermions
            )
            qubit_vector = BinaryArray(bit_array)
            occupancy_vector = inv_trans_mat @ qubit_vector
            occupancy_set = [
                i
                for i, o in enumerate(occupancy_vector)
                if (o == 1 and signs[i] == 1) or (o == 0 and signs[i] == -1)
            ]
            return occupancy_set

        return mapper

    def _augment_dropped_bits(
        self,
        bit_array: list[int],
        n_spin_orbitals: int,
        n_fermions: Optional[int] = None,
    ) -> list[int]:
        """Returns a bit array which is augmented by adding qubits dropped by a
        :class:`FermionQubitMapping`."""
        return bit_array


class OpenFermionJordanWigner(JordanWigner, OpenFermionQubitMapping):
    """Jordan-Wigner transformation using OpenFermion."""

    def get_of_operator_mapper(
        self, n_spin_orbitals: Optional[int] = None, n_fermions: Optional[int] = None
    ) -> OpenFermionQubitOperatorMapper:
        """Returns a function that maps a
        :class:`~openfermion.ops.FermionOperator`,
        :class:`~openfermion.ops.InteractionOperator` or
        :class:`~openfermion.ops.MajoranaOperator`
        to a :class:`openfermion.ops.QubitOperator`
        with Jordan-Wigner transformation.

        Both the arguments (``n_spin_orbitals`` and ``n_fermions``) are ignored since
        the mapping does not depend on them.
        """

        def mapper(
            op: Union["FermionOperator", "InteractionOperator", "MajoranaOperator"]
        ) -> "Operator":
            return operator_from_openfermion_op(of_jordan_wigner(op))

        return mapper

    def get_state_mapper(
        self, n_spin_orbitals: int, n_fermions: Optional[int] = None
    ) -> FermionQubitStateMapper:
        """Returns a function that maps occupied spin orbital indices to a
        computational basis state of qubits with Jordan-Wigner transformation.

        Args:
            n_spin_orbitals:
                The number of spin orbitals to be mapped to qubits.
            n_fermions:
                This argument is ignored since the mapping does not depend on it.
        """
        return super().get_state_mapper(n_spin_orbitals, n_fermions)


jordan_wigner = OpenFermionJordanWigner()


class OpenFermionBravyiKitaev(BravyiKitaev, OpenFermionQubitMapping):
    """Bravyi-Kitaev transformation using OpenFermion."""

    def get_of_operator_mapper(
        self, n_spin_orbitals: Optional[int] = None, n_fermions: Optional[int] = None
    ) -> OpenFermionQubitOperatorMapper:
        """Returns a function that maps a
        :class:`~openfermion.ops.FermionOperator`,
        :class:`~openfermion.ops.InteractionOperator` or
        :class:`~openfermion.ops.MajoranaOperator`
        to a :class:`openfermion.ops.QubitOperator`
        with Bravyi-Kitaev transformation.

        Args:
            n_spin_orbitals:
                The number of spin orbitals to be mapped to qubits. When omitted, it is
                automatically calculated from the largest orbital index contained in
                the operator.
            n_fermions:
                This argument is ignored since the mapping does not depend on it.
        """

        def mapper(
            op: Union["FermionOperator", "InteractionOperator", "MajoranaOperator"]
        ) -> "Operator":
            return operator_from_openfermion_op(
                of_bravyi_kitaev(op, n_qubits=n_spin_orbitals)
            )

        return mapper

    def get_state_mapper(
        self, n_spin_orbitals: int, n_fermions: Optional[int] = None
    ) -> FermionQubitStateMapper:
        """Returns a function that maps occupied spin orbital indices to a
        computational basis state of qubits with Bravyi-Kitaev transformation.

        Args:
            n_spin_orbitals:
                The number of spin orbitals to be mapped to qubits.
            n_fermions:
                This argument is ignored since the mapping does not depend on it.
        """
        return super().get_state_mapper(n_spin_orbitals, n_fermions)


bravyi_kitaev = OpenFermionBravyiKitaev()


class OpenFermionSymmetryConservingBravyiKitaev(
    SymmetryConservingBravyiKitaev, OpenFermionQubitMapping
):
    """Symmetry-conserving Bravyi-Kitaev transformation described in
    arXiv:1701.08213, using OpenFermion.

    Note that in this mapping the spin orbital indices are first
    reordered to all spin-up orbitals, then all spin-down orbitals.
    Bravyi-Kitaev transoformation is applied after the reordering and
    then two qubits are dropped using conservation of particle number
    and spin.

    Any operators which don't have particle number and spin symmetry are
    converted to `Operator()`, whose expectation value is zero for all
    states.
    """

    def get_of_operator_mapper(
        self, n_spin_orbitals: Optional[int] = None, n_fermions: Optional[int] = None
    ) -> OpenFermionQubitOperatorMapper:
        """Returns a function that maps a
        :class:`~openfermion.ops.FermionOperator`,
        :class:`~openfermion.ops.InteractionOperator` or
        :class:`~openfermion.ops.MajoranaOperator`
        to a :class:`openfermion.ops.QubitOperator`
        with symmetry-conserving Bravyi-Kitaev transformation.

        Both the arguments (``n_spin_orbitals`` and ``n_fermions``) are required.
        """

        if n_spin_orbitals is None:
            raise ValueError("n_spin_orbitals is required.")
        if n_fermions is None:
            raise ValueError("n_fermions is required.")

        def mapper(
            op: Union["FermionOperator", "InteractionOperator", "MajoranaOperator"]
        ) -> "Operator":
            if not isinstance(op, FermionOperator):
                op = get_fermion_operator(op)

            openfermion_op = FermionOperator()
            for pauli_product, coef in op.terms.items():
                openfermion_op += FermionOperator(pauli_product, coef)

            operator = Operator()
            for single_term in openfermion_op.get_operators():
                if has_particle_number_symmetry(single_term, check_spin_symmetry=True):
                    operator += operator_from_openfermion_op(
                        of_symmetry_conserving_bravyi_kitaev(
                            single_term, n_spin_orbitals, n_fermions
                        )
                    )
            return operator

        return mapper

    def get_state_mapper(
        self, n_spin_orbitals: int, n_fermions: Optional[int] = None
    ) -> FermionQubitStateMapper:
        """Returns a function that maps occupied spin orbital indices to a
        computational basis state of qubits with symmetry-conserving Bravyi-
        Kitaev transformation.

        Args:
            n_spin_orbitals:
                The number of spin orbitals to be mapped to qubits.
            n_fermions:
                Restrict the mapping to a subspace spanned by states
                containing the fixed number of Fermions. This argument is required.
        """
        if n_fermions is None:
            raise ValueError("n_fermions is required.")

        return super().get_state_mapper(n_spin_orbitals, n_fermions)

    def get_inv_state_mapper(
        self,
        n_spin_orbitals: int,
        n_fermions: Optional[int] = None,
        n_up_spins: Optional[int] = None,
    ) -> QubitFermionStateMapper:
        if n_fermions is None:
            raise ValueError("n_fermions is required.")
        if n_up_spins is None:
            raise ValueError("n_up_spins is required.")
        if 2 * n_up_spins - n_fermions not in [0, 1]:
            raise ValueError("Current implementation only supports sz = 0.0 or 0.5.")
        return super().get_inv_state_mapper(n_spin_orbitals, n_fermions, n_up_spins)

    def _augment_dropped_bits(
        self,
        bit_array: list[int],
        n_spin_orbitals: int,
        n_fermions: Optional[int] = None,
    ) -> list[int]:
        # Add two qubits dropped by the fermion-to-qubit mapping.
        return bit_array + [0, 0]


symmetry_conserving_bravyi_kitaev = OpenFermionSymmetryConservingBravyiKitaev()
