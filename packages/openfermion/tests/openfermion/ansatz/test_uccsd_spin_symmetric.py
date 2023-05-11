from numpy.random import randint

from quri_parts.chem.utils.excitations import excitations, spin_symmetric_excitations
from quri_parts.openfermion.ansatz import TrotterSingletUCCSD

spin_symmetric_uccsd_ansatz = TrotterSingletUCCSD(
    n_spin_orbitals=8,
    n_fermions=4,
    spin_symmetric=True,
)

spin_assymmetric_uccsd_ansatz = TrotterSingletUCCSD(
    n_spin_orbitals=8,
    n_fermions=4,
)


spatial_param_dict = {}
spin_symmetric_param_list = []

for param in spin_symmetric_uccsd_ansatz.param_mapping.in_params:
    name_split = param.name.split("_")
    if name_split[1] == "s":
        _, _, i_str, j_str = name_split
        i, j = int(i_str), int(j_str)
        spatial_param_dict[i, j] = randint(-10, 10)
        spin_symmetric_param_list.append(spatial_param_dict[i, j])

    if name_split[1] == "d":
        _, _, i_str, j_str, k_str, l_str = name_split
        i, j, k, l = int(i_str), int(j_str), int(k_str), int(l_str)
        spatial_param_dict[i, j, k, l] = randint(-10, 10)
        spin_symmetric_param_list.append(spatial_param_dict[i, j, k, l])

spin_assymmetric_signs = {
    (0, 4): [1, (0, 4)],
    (0, 6): [1, (0, 6)],
    (1, 5): [1, (1, 5)],
    (1, 7): [1, (1, 7)],
    (2, 4): [1, (2, 4)],
    (2, 6): [1, (2, 6)],
    (3, 5): [1, (3, 5)],
    (3, 7): [1, (3, 7)],
    (0, 1, 4, 5): [-1, (0, 1, 5, 4)],
    (0, 1, 4, 7): [-1, (0, 1, 7, 4)],
    (0, 1, 5, 6): [+1, (0, 1, 5, 6)],
    (0, 1, 6, 7): [-1, (0, 1, 7, 6)],
    (0, 2, 4, 6): [-1, (0, 2, 6, 4)],
    (0, 3, 4, 5): [-1, (0, 3, 5, 4)],
    (0, 3, 4, 7): [-1, (0, 3, 7, 4)],
    (0, 3, 5, 6): [+1, (0, 3, 5, 6)],
    (0, 3, 6, 7): [-1, (0, 3, 7, 6)],
    (1, 2, 4, 5): [+1, (2, 1, 5, 4)],
    (1, 2, 4, 7): [+1, (2, 1, 7, 4)],
    (1, 2, 5, 6): [-1, (2, 1, 5, 6)],
    (1, 2, 6, 7): [+1, (2, 1, 7, 6)],
    (1, 3, 5, 7): [-1, (1, 3, 7, 5)],
    (2, 3, 4, 5): [-1, (2, 3, 5, 4)],
    (2, 3, 4, 7): [-1, (2, 3, 7, 4)],
    (2, 3, 5, 6): [+1, (2, 3, 5, 6)],
    (2, 3, 6, 7): [-1, (2, 3, 7, 6)],
}

spin_assymmetric_param_list = []

for exc in excitations(8, 4)[0]:
    sgn, symmetric_key = spin_assymmetric_signs[exc]
    if len(symmetric_key) == 2:
        i, j = symmetric_key
        spin_assymmetric_param_list.append(sgn * spatial_param_dict[i // 2, j // 2])

for exc in excitations(8, 4)[1]:
    sgn, symmetric_key = spin_assymmetric_signs[exc]
    if len(symmetric_key) == 4:
        i, j, k, l = symmetric_key
        spin_assymmetric_param_list.append(
            sgn * spatial_param_dict[i // 2, j // 2, k // 2, l // 2]
        )

from quri_parts.circuit import QuantumGate
from quri_parts.core.operator import Operator, SinglePauli, pauli_label


def convert_gate_exponent_to_operator(gate: QuantumGate):
    pauli_map = {
        SinglePauli.X: "X",
        SinglePauli.Y: "Y",
        SinglePauli.Z: "Z",
    }
    pauli_string = ""
    for p, i in zip(gate.pauli_ids, gate.target_indices):
        pauli_string += pauli_map[p] + str(i) + " "
    return Operator({pauli_label(pauli_string): gate.params[0]})


op_assym = Operator({})
for g in spin_assymmetric_uccsd_ansatz.bind_parameters(
    spin_assymmetric_param_list
).gates:
    op_assym += convert_gate_exponent_to_operator(g)

op_sym = Operator({})
for g in spin_symmetric_uccsd_ansatz.bind_parameters(spin_symmetric_param_list).gates:
    op_sym += convert_gate_exponent_to_operator(g)


def test():
    assert len(op_assym - op_sym) == 0
