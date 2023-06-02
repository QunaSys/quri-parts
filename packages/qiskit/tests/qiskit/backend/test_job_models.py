# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numpy import pi
from qiskit import QuantumCircuit as QiskitQuantumCircuit

from quri_parts.qiskit.backend import (
    QiskitSavedDataSamplingJob,
    QiskitSavedDataSamplingResult,
)


def test_qiskit_saved_data_result() -> None:
    raw_data = {
        "0110": 10238,
        "1011": 10388,
        "1110": 10254,
        "0111": 10342,
        "1010": 10203,
        "0011": 10179,
        "0001": 115224,
        "1100": 114515,
        "0100": 114650,
        "1000": 114347,
        "1001": 114832,
        "0010": 10255,
        "0000": 115088,
        "0101": 114819,
        "1111": 10157,
        "1101": 114509,
    }
    expected_counter = {
        6: 10238,
        11: 10388,
        14: 10254,
        7: 10342,
        10: 10203,
        3: 10179,
        1: 115224,
        12: 114515,
        4: 114650,
        8: 114347,
        9: 114832,
        2: 10255,
        0: 115088,
        5: 114819,
        15: 10157,
        13: 114509,
    }

    saved_data_sampling_result = QiskitSavedDataSamplingResult(raw_data=raw_data)
    assert saved_data_sampling_result.counts == expected_counter


def test_qiskit_saved_data_job() -> None:
    circuit_qasm_str = """OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    creg meas[4];
    u2(0,-2.9115927) q[0];
    u3(0.58079633,0,-pi) q[1];
    u2(0,-2.2715927) q[2];
    u2(-0.16,-pi) q[3];
    barrier q[0],q[1],q[2],q[3];
    measure q[0] -> meas[0];
    measure q[1] -> meas[1];
    measure q[2] -> meas[2];
    measure q[3] -> meas[3];
    """
    n_shots = 1000000
    raw_data = {
        "0110": 10238,
        "1011": 10388,
        "1110": 10254,
        "0111": 10342,
        "1010": 10203,
        "0011": 10179,
        "0001": 115224,
        "1100": 114515,
        "0100": 114650,
        "1000": 114347,
        "1001": 114832,
        "0010": 10255,
        "0000": 115088,
        "0101": 114819,
        "1111": 10157,
        "1101": 114509,
    }
    test_qiskit_saved_data_result = QiskitSavedDataSamplingResult(raw_data)
    samplinf_data_sampling_job = QiskitSavedDataSamplingJob(
        circuit_qasm_str, n_shots, test_qiskit_saved_data_result
    )

    expected_qiskit_circuit = QiskitQuantumCircuit(4)
    expected_qiskit_circuit.u2(0, -2.9115927, 0)
    expected_qiskit_circuit.u3(0.58079633, 0, -pi, 1)
    expected_qiskit_circuit.u2(0, -2.2715927, 2)
    expected_qiskit_circuit.u2(-0.16, -pi, 3)
    expected_qiskit_circuit.measure_all()

    assert (
        QiskitQuantumCircuit.from_qasm_str(samplinf_data_sampling_job.circuit_str)
        == expected_qiskit_circuit
    )
    assert samplinf_data_sampling_job.n_shots == 1000000
    assert samplinf_data_sampling_job.saved_result.counts == {
        6: 10238,
        11: 10388,
        14: 10254,
        7: 10342,
        10: 10203,
        3: 10179,
        1: 115224,
        12: 114515,
        4: 114650,
        8: 114347,
        9: 114832,
        2: 10255,
        0: 115088,
        5: 114819,
        15: 10157,
        13: 114509,
    }
