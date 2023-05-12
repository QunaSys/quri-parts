import unittest

from numpy.random import randint

from quri_parts.chem.utils.excitations import (
    DoubleExcitation,
    SingleExcitation,
    excitations,
)
from quri_parts.circuit import QuantumGate
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.openfermion.ansatz import TrotterSingletUCCSD


def get_spin_symmetric_spatial_param_dict_and_circuit_param_list(
    spin_symmetric_uccsd_ansatz: TrotterSingletUCCSD,
) -> tuple[dict[SingleExcitation, int], dict[DoubleExcitation, int], list[int]]:
    spatial_single_exc_param_dict = {}
    spatial_double_exc_param_dict = {}
    spin_symmetric_param_list = []

    for param in spin_symmetric_uccsd_ansatz.param_mapping.in_params:
        name_split = param.name.split("_")
        if name_split[1] == "s":
            _, _, i_str, j_str = name_split
            i, j = int(i_str), int(j_str)
            spatial_single_exc_param_dict[i, j] = randint(-10, 10)
            spin_symmetric_param_list.append(spatial_single_exc_param_dict[i, j])

        if name_split[1] == "d":
            _, _, i_str, j_str, k_str, l_str = name_split
            i, j, k, l = int(i_str), int(j_str), int(k_str), int(l_str)  # noqa: E741
            spatial_double_exc_param_dict[i, j, k, l] = randint(-10, 10)
            spin_symmetric_param_list.append(spatial_double_exc_param_dict[i, j, k, l])

    return (
        spatial_single_exc_param_dict,
        spatial_double_exc_param_dict,
        spin_symmetric_param_list,
    )


def get_general_ansatz_parameter(
    n_spin_orbital: int,
    n_fermions: int,
    general_ansatz_signs_and_param_map_single_exc: dict[
        SingleExcitation, tuple[int, SingleExcitation]
    ],
    general_ansatz_signs_and_param_map_double_exc: dict[
        DoubleExcitation, tuple[int, DoubleExcitation]
    ],
    spatial_single_exc_param_dict: dict[SingleExcitation, int],
    spatial_double_exc_param_dict: dict[DoubleExcitation, int],
) -> list[int]:
    general_ansatz_param_list = []

    for s_exc in excitations(n_spin_orbital, n_fermions)[0]:
        sgn, single_exc_symmetric_key = general_ansatz_signs_and_param_map_single_exc[
            s_exc
        ]
        if len(single_exc_symmetric_key) == 2:
            i, j = single_exc_symmetric_key
            general_ansatz_param_list.append(
                sgn * spatial_single_exc_param_dict[i // 2, j // 2]
            )

    for d_exc in excitations(n_spin_orbital, n_fermions)[1]:
        sgn, double_exc_symmetric_key = general_ansatz_signs_and_param_map_double_exc[
            d_exc
        ]
        if len(double_exc_symmetric_key) == 4:
            i, j, k, l = double_exc_symmetric_key  # noqa: E741
            general_ansatz_param_list.append(
                sgn * spatial_double_exc_param_dict[i // 2, j // 2, k // 2, l // 2]
            )

    return general_ansatz_param_list


def convert_gate_exponent_to_operator(gate: QuantumGate) -> Operator:
    pauli_map = {1: "X", 2: "Y", 3: "Z"}
    pauli_string = ""
    for p, i in zip(gate.pauli_ids, gate.target_indices):
        pauli_string += pauli_map[p] + str(i) + " "
    return Operator({pauli_label(pauli_string): gate.params[0]})


def get_uccsd_generator(
    ansatz_circuit: TrotterSingletUCCSD, parameter: list[int]
) -> Operator:
    generator = Operator({})
    for g in ansatz_circuit.bind_parameters(parameter).gates:
        generator += convert_gate_exponent_to_operator(g)
    return generator


class TestSymmetric(unittest.TestCase):
    spin_symmetric_uccsd_ansatz: TrotterSingletUCCSD
    general_uccsd_ansatz: TrotterSingletUCCSD
    general_ansatz_signs_and_param_map_single_exc: dict[
        SingleExcitation, tuple[int, SingleExcitation]
    ]
    general_ansatz_signs_and_param_map_double_exc: dict[
        DoubleExcitation, tuple[int, DoubleExcitation]
    ]

    @classmethod
    def setUpClass(cls) -> None:
        cls.spin_symmetric_uccsd_ansatz = TrotterSingletUCCSD(
            n_spin_orbitals=8,
            n_fermions=4,
            spin_symmetric=True,
        )

        cls.general_uccsd_ansatz = TrotterSingletUCCSD(
            n_spin_orbitals=8,
            n_fermions=4,
        )
        cls.general_ansatz_signs_and_param_map_single_exc = {
            (0, 4): (1, (0, 4)),
            (0, 6): (1, (0, 6)),
            (1, 5): (1, (1, 5)),
            (1, 7): (1, (1, 7)),
            (2, 4): (1, (2, 4)),
            (2, 6): (1, (2, 6)),
            (3, 5): (1, (3, 5)),
            (3, 7): (1, (3, 7)),
        }
        cls.general_ansatz_signs_and_param_map_double_exc = {
            (0, 1, 4, 5): (-1, (0, 1, 5, 4)),
            (0, 1, 4, 7): (-1, (0, 1, 7, 4)),
            (0, 1, 5, 6): (+1, (0, 1, 5, 6)),
            (0, 1, 6, 7): (-1, (0, 1, 7, 6)),
            (0, 2, 4, 6): (-1, (0, 2, 6, 4)),
            (0, 3, 4, 5): (-1, (0, 3, 5, 4)),
            (0, 3, 4, 7): (-1, (0, 3, 7, 4)),
            (0, 3, 5, 6): (+1, (0, 3, 5, 6)),
            (0, 3, 6, 7): (-1, (0, 3, 7, 6)),
            (1, 2, 4, 5): (+1, (2, 1, 5, 4)),
            (1, 2, 4, 7): (+1, (2, 1, 7, 4)),
            (1, 2, 5, 6): (-1, (2, 1, 5, 6)),
            (1, 2, 6, 7): (+1, (2, 1, 7, 6)),
            (1, 3, 5, 7): (-1, (1, 3, 7, 5)),
            (2, 3, 4, 5): (-1, (2, 3, 5, 4)),
            (2, 3, 4, 7): (-1, (2, 3, 7, 4)),
            (2, 3, 5, 6): (+1, (2, 3, 5, 6)),
            (2, 3, 6, 7): (-1, (2, 3, 7, 6)),
        }

    def test_spin_symmetricity(self) -> None:
        (
            spatial_param_single_exc_dict,
            spatial_param_doubel_exc_dict,
            spin_symmetric_param_list,
        ) = get_spin_symmetric_spatial_param_dict_and_circuit_param_list(
            self.spin_symmetric_uccsd_ansatz
        )

        general_ansatz_parameter_list = get_general_ansatz_parameter(
            8,
            4,
            self.general_ansatz_signs_and_param_map_single_exc,
            self.general_ansatz_signs_and_param_map_double_exc,
            spatial_param_single_exc_dict,
            spatial_param_doubel_exc_dict,
        )

        spin_symmetric_generator = get_uccsd_generator(
            self.spin_symmetric_uccsd_ansatz, spin_symmetric_param_list
        )

        general_generator = get_uccsd_generator(
            self.general_uccsd_ansatz, general_ansatz_parameter_list
        )

        assert len(spin_symmetric_generator - general_generator) == 0
