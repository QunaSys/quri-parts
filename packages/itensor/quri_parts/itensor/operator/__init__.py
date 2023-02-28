from collections.abc import Iterable
from typing import Union

import juliacall
from juliacall import Main as jl
from quri_parts.core.operator import Operator, PauliLabel, pauli_name


def convert_operator(
    operator: Union[Operator, PauliLabel], s: juliacall.VectorValue
) -> juliacall.AnyValue:
    """Convert an :class:`~Operator` or a :class:`~PauliLabel`
    to an ITensor MPO operator."""
    paulis: Iterable[tuple[PauliLabel, complex]]
    if isinstance(operator, Operator):
        paulis = operator.items()
    else:
        paulis = [(operator, 1)]
    os: juliacall.AnyValue = jl.OpSum()
    for pauli, coef in paulis:
        pauli_gates: juliacall.VectorValue = jl.gate_list()
        if len(pauli) == 0:
            os = jl.add_coef_identity(os, coef, s)
            continue
        for i, p in pauli:
            pauli_gates = jl.add_pauli(pauli_gates, pauli_name(p), i + 1)
        os = jl.add_coef_pauli(os, coef, pauli_gates)
    op: juliacall.AnyValue = jl.MPO(os, s)
    return op
