# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from braket.aws import AwsDevice

from quri_parts.circuit.transpile import CZ2CNOTHTranspiler, SequentialTranspiler


class AwsDeviceTranspiler(SequentialTranspiler):
    """CircuitTranspiler to convert a circuit configuration suitable for the
    AwsDevice.

    Args:
        device: Target AwsDevice Object.

    Note:
        This transpiler decomposes also CZ gates only if the device does not support
        CZ gates.
    """

    def __init__(self, device: AwsDevice):
        name_action_type = {
            "LocalSimulator": "braket.ir.jaqcd.program",
            "SV1": "braket.ir.jaqcd.program",
            "TN1": "braket.ir.jaqcd.program",
            "dm1": "braket.ir.jaqcd.program",
            "Harmony": "braket.ir.jaqcd.program",
            "Aria 1": "braket.ir.openqasm.program",
            "Aria 2": "braket.ir.openqasm.program",
            "Garnet": "braket.ir.openqasm.program",
        }
        transpilers = []
        action_type = name_action_type[device.name]
        device_operation = device.properties.dict()["action"][
            action_type
        ]["supportedOperations"]

        if "cz" not in device_operation:
            transpilers.append(CZ2CNOTHTranspiler())
        super().__init__(transpilers)
