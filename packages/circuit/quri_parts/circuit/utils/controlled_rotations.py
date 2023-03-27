from .. import LinearMappedUnboundParametricQuantumCircuit, LinearParameterFunction


def add_controlled_RY_gate(
    circuit: LinearMappedUnboundParametricQuantumCircuit,
    control_index: int,
    target_index: int,
    param_fn: LinearParameterFunction,
) -> LinearMappedUnboundParametricQuantumCircuit:
    inv_sign_param_fn = {param: -1.0 * val for param, val in param_fn.items()}
    circuit.add_ParametricRY_gate(target_index, param_fn)
    circuit.add_CNOT_gate(control_index, target_index)
    circuit.add_ParametricRY_gate(target_index, inv_sign_param_fn)
    circuit.add_CNOT_gate(control_index, target_index)

    return circuit
