using ITensors


function initState(s, qubits::Integer)
    psi = productMPS(s,["0" for i in 1:qubits])
    return psi
end

function expectation(psi, op)
    exp =  inner(psi', op, psi)
    return exp
end

function add_gate(gate_list, gate_name, target_index::Integer)
    push!(gate_list, (gate_name, target_index))
    return gate_list
end

function gate_list()
    return []
end

function add_pauli(pauli_gates, pauli_name, target_index::Integer)
    push!(pauli_gates, pauli_name)
    push!(pauli_gates, target_index)
    return pauli_gates
end

function add_coef_pauli(os, coefficient, pauli_gates)
    os += (coefficient, pauli_gates...)
    return os
end