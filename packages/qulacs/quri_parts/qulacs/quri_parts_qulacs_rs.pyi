import qulacs

from quri_parts.circuit import ImmutableQuantumCircuit

def convert_circuit(gate: ImmutableQuantumCircuit) -> qulacs.QuantumCircuit: ...
