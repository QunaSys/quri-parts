using ITensors
function initState(s)
    psi = productMPS(s, ["0", "0", "1", "0", "0", "0"])
    return psi
end
