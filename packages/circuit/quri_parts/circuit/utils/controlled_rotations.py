from .. import (
    LinearMappedUnboundParametricQuantumCircuit,
    Parameter,
    ParameterOrLinearFunction,
)


def add_controlled_RY_gate(
    circuit: LinearMappedUnboundParametricQuantumCircuit,
    control_index: int,
    target_index: int,
    param_fn: ParameterOrLinearFunction,
) -> LinearMappedUnboundParametricQuantumCircuit:
    if isinstance(param_fn, Parameter):
        p_fn = {param_fn: 0.5}
        inv_sign_p_fn = {param_fn: -0.5}
    else:
        p_fn = {param: 0.5 * val for param, val in param_fn.items()}
        inv_sign_p_fn = {param: -0.5 * val for param, val in param_fn.items()}
    circuit.add_ParametricRY_gate(target_index, p_fn)
    circuit.add_CNOT_gate(control_index, target_index)
    circuit.add_ParametricRY_gate(target_index, inv_sign_p_fn)
    circuit.add_CNOT_gate(control_index, target_index)

    return circuit
