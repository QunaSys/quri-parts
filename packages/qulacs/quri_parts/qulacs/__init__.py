from typing import Sequence, Union, cast

from numpy.typing import ArrayLike
from typing_extensions import TypeAlias, TypeVar

from quri_parts.core.state import (
    CircuitQuantumState,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
    QuantumStateVector,
)

#: A type alias for state classes supported by Qulacs estimators.
#: Qulacs estimators support both of circuit states and state vectors.
QulacsStateT: TypeAlias = Union[CircuitQuantumState, QuantumStateVector]

#: A type alias for parametric state classes supported by Qulacs estimators.
#: Qulacs estimators support both of circuit states and state vectors.
QulacsParametricStateT: TypeAlias = Union[
    ParametricCircuitQuantumState, ParametricQuantumStateVector
]

Numerics = TypeVar("Numerics", int, float, complex)


def cast_to_list(int_sequence: Union[Sequence[Numerics], ArrayLike]) -> list[Numerics]:
    return cast(list[Numerics], int_sequence)
