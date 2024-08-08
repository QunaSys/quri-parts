import sys

from quri_parts.circuit.quri_parts_circuit_rs import (  # type: ignore[import-untyped]
    circuit,
    circuit_parametric,
    gate,
    gates,
    parameter,
)

# The following is required to treat PyO3's submodules
# to be like Python module.
sys.modules["quri_parts.circuit.rust.gate"] = gate
sys.modules["quri_parts.circuit.rust.gates"] = gates
sys.modules["quri_parts.circuit.rust.circuit"] = circuit
sys.modules["quri_parts.circuit.rust.parameter"] = parameter
sys.modules["quri_parts.circuit.rust.circuit_parametric"] = circuit_parametric
