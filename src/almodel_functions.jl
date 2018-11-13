# A fcn to convert Boolean responses into coded values.
# Returns a 2D array of Float64.  If Y is not supplied, use the responses stored
# in the 1st argument.
function makecoded(M::ALmodel, Y=nothing)
    if Y==nothing
        Y = M.responses
    else
        Y = makebool(Y)
    end
    lo = M.coding[1]
    hi = M.coding[2]
    n, m = size(Y)
    out = Array{Float64,2}(undef, n, m)
    for j = 1:m
        for i = 1:n
            out[i,j] = Y[i,j] ? hi : lo
        end
    end
    return out
end


# === centering adjustment =====================================================
# centering_adjustment(M) returns an Array{Float64,2} of the same dimension as 
# M.unary, giving the centering adjustments for ALmodel M.
# centering_adjustment(M,kind) returns the centering adjustment that would be 
#   if centering were of type kind.
# TODO: consider performance implications of calculating this each time instead
# of storing the value.
function centering_adjustment(M::ALmodel, kind::Union{Nothing,CenteringKinds}=nothing) 
    k = kind==nothing ? M.centering : kind
    if k == none
        return fill(0.0, size(M.unary))
    elseif k == onehalf
        return fill(0.5, size(M.unary))
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


# === pseudolikelihood =========================================================
# pseudolikelihood(M) computes the negative log pseudolikelihood for the given 
# ALmodel with its responses.  Returns a Float64.
function pseudolikelihood(M::ALmodel)
    out = 0.0
    Y = makecoded(M)
    mu = centering_adjustment(M)
    lo, hi = M.coding

    # Loop through replicates
    for j = 1:size(Y)[2]
        y = Y[:,j];                     #-Current replicate's observations.
        α = M.unary[:,j]                #-Current replicate's unary parameters.
        μ = mu[:,j]                     #-Current replicate's centering terms.
        Λ = M.pairwise[:,:,j]           #-Current replicate's assoc. matrix.
        s = α + Λ*(y - μ)               #-(λ-weighted) neighbour sums + unary.
        logPL = sum(y.*s - log.(exp.(lo*s) + exp.(hi*s)))
        out = out - logPL               #-Subtract this rep's log PL from total.
    end

    return out

end


# === negpotential function ====================================================
# negpotential(M) returns an m-vector of Float64 negpotential values, where 
# m is the number of replicate observations found in M.responses.
function negpotential(M::ALmodel)
    Y = makecoded(M)
    m = size(Y,2)
    out = Array{Float64}(undef, m)
    α = M.unary
    Λ = M.pairwise
    μ = centering_adjustment(M)
    for j = 1:m
        out[j] = Y[:,j]'*α[:,j] - Y[:,j]'*Λ[:,:,j]*μ[:,j]  + Y[:,j]'*Λ[:,:,j]*Y[:,j]/2
    end
    return out
end


# === probabilitytable =========================================================
# probabilitytable(M) returns a 2^n by n+1 by m array of Float64.  Each page in 
# the 3D array is a probability table giving all possible configurations of the
# response in the rows, with the associated probabilities in the last column. 
# The pages correspond to the different replicates in the ALmodel M (in general,
# different replicates need to be tabulated separately, because they could have 
# different unary and pairwise terms) 
# If the probability table is only desired for certain replicates, use keyword 
# argument replicates to provide the indices of the desired ones. 
# If the number of responses is greater than 20, this function will throw an
# error.  Use keyword argument force to override this behavior.
function fullPMF(M::ALmodel; replicates=nothing, force::Bool=false)
    n, m = size(M.unary)
    nc = 2^n
    if n>20 && !force
        error("Attempting to tabulate a PMF with more than 2^20 configurations."
              * "\nIf you really want to do this, set force=true.")
    end
    if replicates == nothing
        replicates = 1:m
    elseif minimum(replicates)<1 || maximum(replicates)>m 
        error("replicate index out of bounds")
    end
    lo = M.coding[1]
    hi = M.coding[2]
    T = zeros(nc, n+1, length(replicates))
    configs = zeros(nc,n)
    partition = zeros(m)
    for i in 1:n
        inner = [repeat([lo],Int(nc/2^i)); repeat([hi],Int(nc/2^i))]
        configs[:,i] = repeat(inner , 2^(i-1) )
    end
    for i in 1:length(replicates)
        r = replicates[i]
        T[:,1:n,i] = configs
        α = M.unary[:,r]
        Λ = M.pairwise[:,:,r]
        μ = centering_adjustment(M)[:,r]
        unnormalized = mapslices(v -> exp.(v'*α - v'*Λ*μ + v'*Λ*v/2), configs, dims=2)
        partition[i] = sum(unnormalized)
        T[:,n+1,i] = unnormalized / partition[i]
    end

    return (table=T, partition=partition)
end

