from copy import copy
from typing import Callable, Sequence, Union, overload

from qulacs import ParametricQuantumCircuit as QulacsParametricQuantumCircuit
from qulacs import QuantumCircuit as QulacsQuantumCircuit

import quri_parts.qulacs.circuit as qlc
from quri_parts.circuit import (
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    ImmutableQuantumCircuit,
    ImmutableUnboundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuitBase,
    NonParametricQuantumCircuit,
    UnboundParametricQuantumCircuit,
    UnboundParametricQuantumCircuitBase,
)


def compile_circuit(circuit: NonParametricQuantumCircuit) -> "_QulacsCircuit":
    return _QulacsCircuit(circuit)


@overload
def compile_parametric_circuit(
    circuit: UnboundParametricQuantumCircuit,
) -> "_QulacsUnboundParametricCircuit":
    ...


@overload
def compile_parametric_circuit(
    circuit: LinearMappedUnboundParametricQuantumCircuit,
) -> "_QulacsLinearMappedUnboundParametricCircuit":
    ...


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
    def __init__(self, circuit: NonParametricQuantumCircuit):
        super().__init__(circuit)
        self._qulacs_circuit = qlc.convert_circuit(circuit)

    @property
    def qulacs_circuit(self) -> QulacsQuantumCircuit:
        return self._qulacs_circuit.copy()


class _QulacsUnboundParametricCircuit(ImmutableUnboundParametricQuantumCircuit):
    def __init__(self, circuit: UnboundParametricQuantumCircuitBase):
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
    def __init__(self, circuit: LinearMappedUnboundParametricQuantumCircuitBase):
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
