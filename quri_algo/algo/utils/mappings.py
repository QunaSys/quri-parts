from typing import Iterator, Mapping, Sequence, TypeVar

from quri_parts.circuit import ImmutableQuantumCircuit

U = TypeVar("U")


class CircuitMapping(Mapping[ImmutableQuantumCircuit, U]):
    """Map an immutable quantum circuit to a value."""

    _vals_dictionary: dict[int, U]
    _keys_dictionary: dict[int, ImmutableQuantumCircuit]

    def _circuit_hash(self, circuit: ImmutableQuantumCircuit) -> int:
        """Hash the circuit gates and qubit count."""
        return hash((circuit.qubit_count, circuit.gates))

    def __init__(
        self, circuit_vals: Sequence[tuple[ImmutableQuantumCircuit, U]]
    ) -> None:
        self._vals_dictionary = {}
        self._keys_dictionary = {}
        for circuit, val in circuit_vals:
            self._vals_dictionary[self._circuit_hash(circuit)] = val
            self._keys_dictionary[self._circuit_hash(circuit)] = circuit

    def __getitem__(self, circuit: ImmutableQuantumCircuit) -> U:
        return self._vals_dictionary.__getitem__(self._circuit_hash(circuit))

    def __setitem__(self, circuit: ImmutableQuantumCircuit, value: U) -> None:
        self._vals_dictionary.__setitem__(self._circuit_hash(circuit), value)
        self._keys_dictionary.__setitem__(self._circuit_hash(circuit), circuit)

    def __delitem__(self, circuit: ImmutableQuantumCircuit) -> None:
        self._vals_dictionary.__delitem__(self._circuit_hash(circuit))
        self._keys_dictionary.__delitem__(self._circuit_hash(circuit))

    def __iter__(self) -> Iterator[ImmutableQuantumCircuit]:
        return self._keys_dictionary.values().__iter__()

    def __len__(self) -> int:
        return len(self._vals_dictionary)

    def __contains__(self, key: object) -> bool:
        return True if key in self._keys_dictionary.values() else False
