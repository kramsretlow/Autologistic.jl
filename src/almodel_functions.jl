# A fcn to convert the Boolean responses in an ALmodel into coded values.
# Returns a 2D array of Float64
function makecoded(M::ALmodel)
    lo = M.coding[1]
    hi = M.coding[2]
    n, m = size(M.responses)
    out = Array{Float64,2}(undef, n, m)
    for j = 1:m
        for i = 1:n
            out[i,j] = M.responses[i,j] ? hi : lo
        end
    end
    return out
end


# === centering adjustment =====================================================
# centering_adjustment(M) returns a Vector{Float64} giving the centering  
#   adjustments for ALmodel M.
# centering_adjustment(M,kind) returns the centering adjustment that would be 
#   if centering were of type kind.
# TODO: consider performance implications of calculating this each time instead
# of storing the value.
function centering_adjustment(M::ALmodel; kind::Union{Nothing,CenteringKinds}=nothing) 
    k = kind==nothing ? M.centering : kind
    if k == none
        return fill(0.0, length(M.unary))
    elseif k == onehalf
        return fill(0.5, length(M.unary))
    elseif k == expectation
        lo = M.coding[1]
        hi = M.coding[2]
        α = M.unary
        num = lo*exp.(lo*α) + hi*exp.(hi*α)
        denom = exp.(lo*α) + exp.(hi*α)
        return num./denom
    else 
        error("centering kind not recognized")
    end
end


# === negpotential function ====================================================
# negpotential(M) returns an m-vector of Float64 negpotential values, where 
# m is the number of replicate observations found in M.responses.
function negpotential(M::ALmodel)
    m = size(M.responses)[2]
    out = Array{Float64}(undef, m)
    Y = makecoded(M)
    α = M.unary
    Λ = M.pairwise
    μ = centering_adjustment(M)
    for j = 1:m
        out[j] = Y[:,j]'*α - α'*Λ*μ  + α'*Λ*α/2
    end
    return out
end

