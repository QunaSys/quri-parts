using ITensors


function initState(s, qubits::Integer)
    psi = productMPS(s, ["0" for i in 1:qubits])
    return psi
end

function expectation(psi, op)
    exp = inner(psi', op, psi)
    return exp
end

function add_gate(gate_list::Vector, gate_name::String, target_index::Integer)::Vector
    push!(gate_list, (gate_name, target_index))
    return gate_list
end

function add_gate(gate_list::Vector, gate_name::String, control_index::Integer, target_index::Integer)::Vector
    push!(gate_list, (gate_name, control_index, target_index))
    return gate_list
end

function add_gate(gate_list::Vector, gate_name::String, target_index::Integer, param::Number)::Vector
    if gate_name == "Rx" || gate_name == "Ry"
        push!(gate_list, (gate_name, target_index, (θ=param,)))
    elseif gate_name == "Rz"
        push!(gate_list, (gate_name, target_index, (ϕ=param,)))
    else
        raise("Invalid gate name")
    end
    return gate_list
end

function gate_list()
    return []
end

function add_pauli(pauli_gates::Vector, pauli_name::String, target_index::Integer)::Vector
    push!(pauli_gates, pauli_name)
    push!(pauli_gates, target_index)
    return pauli_gates
end

function add_coef_pauli(os, coefficient, pauli_gates)
    os += (coefficient, pauli_gates...)
    return os
end


function sampling(psi, shots)
    orthogonalize!(psi, 1)
    result = []
    for i in 1:shots
        sampling = sample(psi)
        count = 0
        for i in 1:length(sampling)
            count += (sampling[i] - 1) * 2^(i - 1)
        end
        push!(result, count)
    end
    return result
end
