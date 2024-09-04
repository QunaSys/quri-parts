from dataclasses import dataclass
from typing import Optional

from quri_parts.circuit.gate_names import NonParametricGateNameType

from .units import FrequencyValue, TimeValue


@dataclass(frozen=True)
class QubitProperty:
    """Noise property of a qubit.

    Args:
        T1 (TimeValue, optional): t1 relaxation time
        T2 (TimeValue, optional): t2 relaxation time
        frequency (FrequencyValue,optional): qubit frequency
        prob_meas0_on_state1 (float, optional): readout error probability of
            measuring 0 when the state is 1
        prob_meas1_on_state0 (float, optional): readout error probability of
            measuring 1 when the state is 0
        readout_time (TimeValue, optional): time duration on measurement.
        name (str, optional): name of the qubit
    """

    T1: Optional[TimeValue] = None
    T2: Optional[TimeValue] = None
    frequency: Optional[FrequencyValue] = None
    prob_meas0_on_state1: Optional[float] = None
    prob_meas1_on_state0: Optional[float] = None
    readout_time: Optional[TimeValue] = None
    name: Optional[str] = None


@dataclass(frozen=True)
class GateProperty:
    """Noise property of a gate.

    Args:
        gate (GateNameType): gate name
        quibts (list[int]): target qubits for the gate. The order is control_index0,
            control_index1, ..., target_index0, ...
        gate_error (float, optional): 1 - fidelity of the gate operation
        gate_time (float, optional): time duration of the gate operation
        name (str, optional): name of the gate
    """

    gate: NonParametricGateNameType
    qubits: list[int]
    gate_error: Optional[float] = None
    gate_time: Optional[TimeValue] = None
    name: Optional[str] = None
