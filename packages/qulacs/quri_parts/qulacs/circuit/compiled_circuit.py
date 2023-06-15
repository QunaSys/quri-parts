from copy import copy
from typing import Callable, Sequence, Union

from qulacs import ParametricQuantumCircuit as QulacsParametricQuantumCircuit
from qulacs import QuantumCircuit as QulacsQuantumCircuit

import quri_parts.qulacs.circuit as qlc
from quri_parts.circuit import (
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    ImmutableQuantumCircuit,
    ImmutableUnboundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuit,
    QuantumCircuit,
    UnboundParametricQuantumCircuit,
)


def compile_circuit(circuit: QuantumCircuit) -> "_QulacsCircuit":
    return _QulacsCircuit(circuit)


def compile_parametric_circuit(
    circuit: Union[
        UnboundParametricQuantumCircuit, LinearMappedUnboundParametricQuantumCircuit
    ]
) -> Union[
    "_QulacsUnboundParametricCircuit", "_QulacsLinearMappedUnboundParametricCircuit"
]:
    if isinstance(circuit, UnboundParametricQuantumCircuit):
        return _QulacsUnboundParametricCircuit(circuit)
    elif isinstance(circuit, LinearMappedUnboundParametricQuantumCircuit):
        return _QulacsLinearMappedUnboundParametricCircuit(circuit)
    else:
        raise ValueError(f"Unsupported parametric circuit type: {type(circuit)}")


class _QulacsCircuit(ImmutableQuantumCircuit):
    def __init__(self, circuit: QuantumCircuit):
        super().__init__(circuit)
        self._qulacs_circuit = qlc.convert_circuit(circuit)

    @property
    def qulacs_circuit(self) -> QulacsQuantumCircuit:
        return self._qulacs_circuit.copy()


class _QulacsUnboundParametricCircuit(ImmutableUnboundParametricQuantumCircuit):
    def __init__(self, circuit: UnboundParametricQuantumCircuit):
        super().__init__(circuit)
        self._qulacs_circuit, self._param_mapper = qlc.convert_parametric_circuit(
            circuit
        )

    @property
    def qulacs_circuit(self) -> QulacsParametricQuantumCircuit:
        return self._qulacs_circuit.copy()

    @property
    def param_mapper(self) -> Callable[[Sequence[float]], Sequence[float]]:
        return copy(self._param_mapper)


class _QulacsLinearMappedUnboundParametricCircuit(
    ImmutableLinearMappedUnboundParametricQuantumCircuit
):
    def __init__(self, circuit: LinearMappedUnboundParametricQuantumCircuit):
        super().__init__(circuit)
        (
            self._qulacs_circuit,
            self._qulacs_param_mapper,
        ) = qlc.convert_parametric_circuit(circuit)

    @property
    def qulacs_circuit(self) -> QulacsParametricQuantumCircuit:
        return self._qulacs_circuit.copy()

    @property
    def param_mapper(self) -> Callable[[Sequence[float]], Sequence[float]]:
        return copy(self._qulacs_param_mapper)