# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod, abstractproperty
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Optional, Protocol, Union, runtime_checkable

from quri_parts.circuit import gate_names

#: A mutable unbound parametric quantum circuit.
#:
#: This class is for parametric circuits where all parametric gates have independent
#: parameters, i.e., no two parametric gates share the same parameter. If you want to
#: make dependency between parameters, see
#: : class: `~LinearMappedUnboundParametricQuantumCircuit`.
#: An immutable unbound parametric quantum circuit.
#:
#: This class is for parametric circuits where all parametric gates have independent
#: parameters, i.e., no two parametric gates share the same parameter. If you want to
#: make dependency between parameters, see
#: : class: `~ImmutableLinearMappedUnboundParametricQuantumCircuit`.
from quri_parts.circuit.rust.circuit_parametric import (
    ImmutableParametricQuantumCircuit,
    ParametricQuantumCircuit,
)

from .circuit import (
    GateSequence,
    MutableQuantumCircuitProtocol,
    QuantumCircuit,
    QuantumCircuitProtocol,
)
from .gate import ParametricQuantumGate, QuantumGate
from .gates import RX, RY, RZ, PauliRotation
from .parameter import Parameter
from .parameter_mapping import ParameterMapping


@runtime_checkable
class UnboundParametricQuantumCircuitProtocol(QuantumCircuitProtocol, Protocol):
    """Interface protocol for a quantum circuit containing unbound (i.e. not
    assigned values) parameters.

    For circuit execution, the parameters need to be assigned concrete
    values by :meth:`~bind_parameters` method.
    """

    @abstractmethod
    def bind_parameters(
        self, params: Sequence[float]
    ) -> "ImmutableBoundParametricQuantumCircuit":
        """Returns a new circuit with the parameters assigned concrete values.

        This method does not modify self but returns a newly created
        circuit.
        """
        ...

    @abstractmethod
    def freeze(self) -> "UnboundParametricQuantumCircuitProtocol":
        """Returns a \"freezed\" version of itself.

        The \"freezed\" version is an immutable object and can be reused
        safely without copying.
        """
        ...

    @abstractmethod
    def get_mutable_copy(self) -> "MutableUnboundParametricQuantumCircuitProtocol":
        """Returns a copy of itself that can be modified.

        Use this method when you want to get a new circuit based on an
        existing circuit but don't want to modify it.
        """
        ...

    @abstractmethod
    def primitive_circuit(self) -> "ImmutableUnboundParametricQuantumCircuit":
        r"""Returns the parametric circuit where each gate has an independent
        parameter.

        Note that some parametric circuit,
        e.g. :class:`LinearMappedUnboundParametricQuantumCircuit`, can have non-trivial
        mapping of the parameters. In this "primitive circuit", however,
        gate parameters are treated as independent,
        even if those in the original circuit depend on the same parameters.
        For example, if the parametric circuit is defined as:

        .. math::
            \begin{align}
                U_1(f(\theta_1, \theta_2)) U_2(g(\theta_1, \theta_2))
            \end{align}

        the primitive circuit should be as the following:

        .. math::
            \begin{align}
                U_1(\psi_1) U_2(\psi_2)
            \end{align}

        where U1, U2 are rotation gates and f, g are parameter mappings.
        """
        ...

    @abstractproperty
    def parameter_count(self) -> int:
        ...

    @abstractproperty
    def gates(self) -> Sequence[Union[QuantumGate, ParametricQuantumGate]]:
        """Returns the gate sequence of the circuit."""
        ...

    @abstractproperty
    def has_trivial_parameter_mapping(self) -> bool:
        """Returns if the input parameters are used for parametric gates
        without any conversions.

        Note that some parametric circuit,
        e.g. :class:`LinearMappedUnboundParametricQuantumCircuit`, can have non-trivial
        mapping of the parameters.
        """
        ...

    @abstractproperty
    def param_mapping(self) -> ParameterMapping:
        """Returns the parameter mapping of the circuit."""
        ...

    def bind_parameters_by_dict(
        self, params_dict: dict[Parameter, float]
    ) -> "ImmutableBoundParametricQuantumCircuit":
        """Returns a new circuit with the parameters assigned concrete values.

        This method does not modify self but returns a newly created
        circuit.
        """
        param_list = []
        for param in self.param_mapping.in_params:
            param_list.append(params_dict[param])
        return self.bind_parameters(param_list)

    def __add__(
        self, gates: Union[GateSequence, "UnboundParametricQuantumCircuitProtocol"]
    ) -> "UnboundParametricQuantumCircuitProtocol":
        """Returns a new combined circuit with the given gates added."""
        ...

    def __radd__(
        self, gates: Union[GateSequence, "UnboundParametricQuantumCircuitProtocol"]
    ) -> "UnboundParametricQuantumCircuitProtocol":
        """Returns a new combined circuit with the given gates added."""
        ...


class MutableUnboundParametricQuantumCircuitProtocol(
    UnboundParametricQuantumCircuitProtocol, MutableQuantumCircuitProtocol, Protocol
):
    ...


#: A base class for quantum circuits having parametric gates with unbound
#: parameters.
#:
#: This class is for parametric circuits where all parametric gates have independent
#: parameters, i.e., no two parametric gates share the same parameter. If you want to
#: make dependency between parameters, see
#: :class:`~LinearMappedUnboundParametricQuantumCircuitBase`.
UnboundParametricQuantumCircuitBase = ImmutableParametricQuantumCircuit

UnboundParametricQuantumCircuit = ParametricQuantumCircuit
ImmutableUnboundParametricQuantumCircuit = ImmutableParametricQuantumCircuit


class ImmutableBoundParametricQuantumCircuit(QuantumCircuit):
    """An immutable \"bound\" parametric quantum circuit, i.e. a parametric
    circuit with its parameters assigned concrete values."""

    unbound_param_circuit: ImmutableParametricQuantumCircuit
    parameter_map: dict[Parameter, float]

    def __new__(
        cls,
        circuit: UnboundParametricQuantumCircuitBase,
        parameter_map: Mapping[Parameter, float],
    ) -> "ImmutableBoundParametricQuantumCircuit":
        # We cannot extend ImmutableQuantumCircuit, because it is frozen.
        # For workaround, we extend QuantumCircuit instead of ImmutableQuantumCircuit.
        ob: "ImmutableBoundParametricQuantumCircuit" = super().__new__(
            cls, circuit.qubit_count, circuit.cbit_count
        )  # type: ignore
        ob.unbound_param_circuit = circuit.freeze()
        ob.parameter_map = dict(parameter_map)
        return ob

    @cached_property
    def gates(self) -> Sequence[QuantumGate]:
        def map_param_gate(
            gate: Union[QuantumGate, ParametricQuantumGate], param: Optional[Parameter]
        ) -> QuantumGate:
            if isinstance(gate, QuantumGate):
                return gate
            elif isinstance(gate, ParametricQuantumGate) and param is not None:
                value = self.parameter_map[param]
                if gate.name == gate_names.ParametricRX:
                    return RX(gate.target_indices[0], value)
                elif gate.name == gate_names.ParametricRY:
                    return RY(gate.target_indices[0], value)
                elif gate.name == gate_names.ParametricRZ:
                    return RZ(gate.target_indices[0], value)
                elif gate.name == gate_names.ParametricPauliRotation:
                    return PauliRotation(gate.target_indices, gate.pauli_ids, value)
                else:
                    raise ValueError("Unknown gate")
            else:
                raise ValueError("ParametricQuantumGate must have a parameter.")

        return tuple(map_param_gate(*t) for t in self.unbound_param_circuit._gates)

    @cached_property
    def depth(self) -> int:
        max_layers_each_qubit = [0] * self.qubit_count
        for gate in self.gates:
            acting_ids = [*gate.control_indices, *gate.target_indices]
            max_layers = max(max_layers_each_qubit[index] for index in acting_ids)
            for index in acting_ids:
                max_layers_each_qubit[index] = max_layers + 1
        return max(max_layers_each_qubit)

    def freeze(self) -> "ImmutableBoundParametricQuantumCircuit":
        return self


__all__ = [
    "ImmutableParametricQuantumCircuit",
    "ParametricQuantumCircuit",
    "UnboundParametricQuantumCircuit",
    "UnboundParametricQuantumCircuitProtocol",
    "MutableUnboundParametricQuantumCircuitProtocol",
    "ImmutableBoundParametricQuantumCircuit",
]
