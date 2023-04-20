# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

# The implementation is originally from qulacs-visualizer
# https://github.com/Qulacs-Osaka/qulacs-visualizer
from typing import Sequence

import numpy as np
import numpy.typing as npt

from quri_parts.circuit import NonParametricQuantumCircuit, QuantumGate, gate_names

_GATE_STR_MAP = {
    gate_names.X: " X ",
    gate_names.Y: " Y ",
    gate_names.Z: " Z ",
    gate_names.H: " H ",
    gate_names.S: " S ",
    gate_names.Sdag: "Sdg",
    gate_names.SqrtX: "sqX",
    gate_names.SqrtXdag: "sXd",
    gate_names.SqrtY: "sqY",
    gate_names.SqrtYdag: "sYd",
    gate_names.T: " T ",
    gate_names.Tdag: "Tdg",
    gate_names.RX: "RX ",
    gate_names.RY: "RY ",
    gate_names.RZ: "RZ ",
    gate_names.U1: "U1 ",
    gate_names.U2: "U2 ",
    gate_names.U3: "U3 ",
    gate_names.CNOT: "CX ",
    gate_names.CZ: "CZ ",
    gate_names.Pauli: " P ",
    gate_names.PauliRotation: "PR ",
    gate_names.UnitaryMatrix: "Mat",
}


def draw_circuit(circuit: NonParametricQuantumCircuit, line_length: int = 80) -> None:
    """Circuit drawer which outputs given circuit as an ASCII art to standart
    streams. Current implementation only supports
    :class:`NonParametricQuantumCircuit.`

    Args:
        circuit: Circuit to be output.
        line_length: Maximum output line length.
    """

    qubit_count = circuit.qubit_count
    depth = circuit.depth  # depth of the circuit picture (can be increased lator)

    # 2D Boolean array. True if [i,j] is empty.
    gate_map = np.full((qubit_count, depth), True)

    # 4 for each qubit.
    vertical_size = qubit_count * 4
    # 7 for each gate. (depth-1) is for "-" connecting between gates.
    # The last term is "-" put at the beginning and the end of the circuit.
    horizontal_size = (depth * 7) + (depth - 1) + 2

    # 2D str array to be printed.
    circuit_picture = np.full((vertical_size, horizontal_size), " ")

    # "-" at the beginning and end of the circuit.
    for i in range(qubit_count):
        row_idx = (i + 1) * 4 - 2
        circuit_picture[row_idx][0] = "-"
        circuit_picture[row_idx][-1] = "-"

    for idx, gate in enumerate(circuit.gates):
        gate_aa = _generate_gate_aa(gate, idx)
        upper_left_idx, depth, gate_map, circuit_picture = _place_check(
            gate, qubit_count, depth, gate_map, circuit_picture
        )
        circuit_picture = _write_gate_string(gate_aa, upper_left_idx, circuit_picture)
        circuit_picture = _connect_wire(qubit_count, depth, circuit_picture)

    # Divide and fold
    if circuit_picture.shape[1] > line_length:
        n_divs = circuit_picture.shape[1] // line_length
        output = circuit_picture[:, :line_length]
        for i in range(1, n_divs):
            output = np.concatenate(
                [
                    output,
                    np.array(
                        [
                            [""] * line_length,
                            [""] * line_length,
                            ["="] * line_length,
                            [""] * line_length,
                        ]
                    ),
                ],
                axis=0,
            )
            output = np.concatenate(
                [
                    output,
                    circuit_picture[:, line_length * i : line_length * (i + 1)],  # noqa
                ],
                axis=0,
            )
    else:
        output = circuit_picture

    for line in output:
        print("".join(line))


def _generate_gate_aa(gate: QuantumGate, gate_idx: int) -> Sequence[str]:
    if gate_idx > 999:
        warnings.warn(
            f"Gate index larger than 999 is not supported. Index {gate_idx}"
            " is modified to 999.",
            Warning,
        )
    t_idxs = sorted(gate.target_indices)
    c_idxs = sorted(gate.control_indices)
    gate_string: list[str] = []

    if gate.name == gate_names.SWAP:
        gate_string = _create_swap_string(gate.target_indices, gate_idx)
    else:
        # checks if control bit index < target bit index
        if len(c_idxs) != 0 and min(t_idxs) > min(c_idxs):
            is_target_top = False
            gate_string = _create_upper_control_part(gate_string, c_idxs, t_idxs)
        else:
            is_target_top = True

        # generate target part
        gate_size = max(t_idxs) - min(t_idxs) + 1
        if is_target_top:
            gate_head = "  ___  "
        else:
            gate_head = "  _|_  "
        if gate.name in _GATE_STR_MAP:
            g_str = _GATE_STR_MAP[gate.name].ljust(3)
            gate_name_part = f" |{g_str}| "
        else:
            gate_name_part = " |UDF| "  # undefined gate
        gate_body_with_wire = "-|   |-"
        gate_body = " |   | "
        gate_bottom = " |___| "

        gate_string.append(gate_head)
        gate_string.append(gate_name_part)
        g_idx_str = str(gate_idx).ljust(3)
        gate_string.append(f"-|{g_idx_str}|-")

        for i in range(1, gate_size * 4 - 3):
            q_idx = (i + 2) // 4 + min(t_idxs)
            if q_idx in t_idxs:
                if i % 4 == 0:
                    gate_string.append(gate_body_with_wire)
                elif i % 4 == 1:
                    if q_idx + 1 in t_idxs:
                        gate_string.append(gate_body)
                    else:
                        gate_string.append(" |_ _| ")
                elif i % 4 == 2:
                    if q_idx - 1 in t_idxs:
                        gate_string.append(gate_body)
                    else:
                        gate_string.append(" _| |_ ")
                else:
                    gate_string.append(gate_body)
            else:
                if i % 4 == 0:
                    gate_string.append("--| |--")
                else:
                    gate_string.append("  | |  ")
        gate_string.append(gate_bottom)

        if len(c_idxs) != 0 and max(t_idxs) < max(c_idxs):
            gate_string = _create_lower_control_part(gate_string, c_idxs, t_idxs)
    return gate_string


def _create_upper_control_part(
    gate_string: list[str], control_idxs: Sequence[int], target_idxs: Sequence[int]
) -> list[str]:
    ctrl_q_head = "       "
    ctrl_q_name = "       "
    ctrl_q_body = "   ●   "
    vertical_wire = "   |   "

    upper_ctrl_q_list: list[int] = [
        ctrl_idx for ctrl_idx in control_idxs if ctrl_idx < min(target_idxs)
    ]

    gate_string.append(ctrl_q_head)
    gate_string.append(ctrl_q_name)
    gate_string.append(ctrl_q_body)

    dist = min(target_idxs) - min(upper_ctrl_q_list)
    for _ in range(dist * 4 - 3):
        gate_string.append(vertical_wire)

    for i in upper_ctrl_q_list[1:]:
        dist = min(target_idxs) - i
        p = dist * 4 - 2
        gate_string[-p] = ctrl_q_body

    return gate_string


def _create_lower_control_part(
    gate_string: list[str], control_idxs: Sequence[int], target_idxs: Sequence[int]
) -> list[str]:
    control_q_body = "   ●   "
    vertical_wire = "   |   "

    lower_ctrl_q_list: list[int] = [
        idx for idx in control_idxs if idx > max(target_idxs)
    ]

    dist = max(lower_ctrl_q_list) - max(target_idxs)
    loop = dist * 4 - 1
    for _ in range(loop):
        gate_string.append(vertical_wire)

    for i in lower_ctrl_q_list:
        diff = i - max(target_idxs)
        p = diff * 4 - loop - 2
        gate_string[p] = control_q_body
    return gate_string


def _create_swap_string(target_indices: Sequence[int], gate_idx: int) -> list[str]:
    gate_string: list[str] = []
    gate_size = max(target_indices) - min(target_indices) + 1

    gate_head = "       "
    gate_index_str = str(gate_idx).ljust(3)
    gate_name_part = f"  {gate_index_str}  "
    target_part = "---x---"  # target qubits
    non_target_part = "---|---"  # non-target qubits
    gate_body = "   |   "
    gate_bottom = "       "

    gate_string.append(gate_head)
    gate_string.append(gate_name_part)
    gate_string.append(target_part)
    for i in range(1, gate_size * 4 - 4):
        if i % 4 == 0:
            gate_string.append(non_target_part)
        else:
            gate_string.append(gate_body)
    gate_string.append(target_part)
    gate_string.append(gate_bottom)
    return gate_string


def _place_check(
    gate: QuantumGate,
    qubit_count: int,
    depth: int,
    gate_map: npt.NDArray[np.bool_],
    circuit_picture: npt.NDArray[np.string_],
) -> tuple[tuple[int, int], int, npt.NDArray[np.bool_], npt.NDArray[np.string_]]:
    gate_indices = (*gate.control_indices, *gate.target_indices)
    min_idx = min(gate_indices)
    max_idx = max(gate_indices)

    row_idx = min_idx * 4
    for i in range(depth):
        if all(gate_map[min_idx : max_idx + 1, i]):  # noqa
            gate_map[min_idx : max_idx + 1, : i + 1] = False  # noqa
            return (row_idx, i * 8 + 1), depth, gate_map, circuit_picture

    # Expand picture and gate_map
    depth += 1
    additional_gate_map = np.full(qubit_count, True).reshape(qubit_count, 1)
    gate_map = np.concatenate([gate_map, additional_gate_map], axis=1)

    additional_c_pic = np.full((qubit_count * 4, 8), " ")
    circuit_picture = np.concatenate([circuit_picture, additional_c_pic], axis=1)

    for i in range(qubit_count):
        row = (i + 1) * 4 - 2
        circuit_picture[row][-1] = "-"

    gate_map[min_idx:max_idx, : depth + 1] = False
    return (row_idx, depth), depth, gate_map, circuit_picture


def _write_gate_string(
    gate_string: Sequence[str],
    upper_left_index: tuple[int, int],
    circuit_picture: npt.NDArray[np.string_],
) -> npt.NDArray[np.string_]:
    row_idx, col_idx = upper_left_index
    width = 7
    for i, line in enumerate(gate_string):
        circuit_picture[row_idx + i][col_idx : col_idx + width] = list(line)  # noqa
    return circuit_picture


def _connect_wire(
    qubit_count: int, depth: int, circuit_picture: npt.NDArray[np.string_]
) -> npt.NDArray[np.string_]:
    horizontal_size = (depth * 7) + (depth - 1) + 2
    for i in range(qubit_count):
        row = (i + 1) * 4 - 2
        p = 0
        while True:
            char_now = circuit_picture[row][p]
            if char_now == "●":
                circuit_picture[row][p + 1] = "-"
            elif char_now == "-":
                if circuit_picture[row][p + 1] == " ":
                    circuit_picture[row][p + 1] = "-"
                elif circuit_picture[row][p + 1] == "|":
                    if circuit_picture[row][p + 5] == "|":
                        p += 5
                    elif circuit_picture[row][p + 3] == "|":
                        p += 3
                    else:
                        circuit_picture[row][p + 2] = "-"
                        p += 1
            p += 1
            if p + 1 == horizontal_size:
                break
    return circuit_picture
