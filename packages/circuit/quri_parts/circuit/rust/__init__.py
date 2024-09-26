import sys
from typing import Any

from quri_parts.circuit.quri_parts_circuit_rs import (  # type: ignore[import-untyped]
    circuit,
    circuit_parametric,
    gate,
    gates,
    noise,
    parameter,
)

_old__iadd__ = circuit_parametric.ParametricQuantumCircuit.__iadd__


def _new__iadd__(lhs: circuit_parametric.ParametricQuantumCircuit, rhs: Any) -> Any:
    try:
        return _old__iadd__(lhs, rhs)
    except NotImplementedError:
        return NotImplemented


setattr(circuit_parametric.ParametricQuantumCircuit, "__iadd__", _new__iadd__)


# The following is required to treat PyO3's submodules
# to be like Python module.
sys.modules["quri_parts.circuit.rust.gate"] = gate
sys.modules["quri_parts.circuit.rust.gates"] = gates
sys.modules["quri_parts.circuit.rust.circuit"] = circuit
sys.modules["quri_parts.circuit.rust.parameter"] = parameter
sys.modules["quri_parts.circuit.rust.circuit_parametric"] = circuit_parametric
sys.modules["quri_parts.circuit.rust.noise"] = noise

__all__ = ["gate", "gates", "parameter", "circuit", "circuit_parametric", "noise"]
