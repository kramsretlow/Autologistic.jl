"""
    AbstractAutologisticModel

Abstract type representing autologistic models.  This type has methods defined for most
operations one will want to perform, so that concrete subtypes should not have to define
too many methods unless more specialized and efficient algorithms for the specific subtype.

All concrete subtypes should have the following fields:

*   `responses::Array{Bool,2}` -- The binary observations. Rows are for nodes in the 
    graph, and columns are for replicate observations.  It is a 2D array even if only one 
    set of responses is observed.
*   `unary<:AbstractUnaryParameter` -- Specifies the unary part of the model.
*   `pairwise<:AbstractPairwiseParameter`  -- Specifies the pairwise part of the model 
    (including the graph).
*   `centering<:CenteringKinds` -- Specifies the form of centering used, if any.
*   `coding::Tuple{T,T} where T<:Real` -- Gives the numeric coding of the responses.
*   `labels::Tuple{String,String}` -- Provides names for the high and low states.
*   `coordinates<:SpatialCoordinates` -- Provides 2D or 3D coordinates for each vertex in 
    the graph (or `nothing` if no coordinates).

The following functions are defined for the abstract type, and are considered part of the 
type's interface (in this list, `M` of type inheriting from `AbstractAutologisticModel`).

*   `getparameters(M)` and `setparameters!(M, newpars::Vector{Float64})`
*   `getunaryparameters(M)` and `setunaryparameters!(M, newpars::Vector{Float64})`
*   `getpairwiseparameters(M)` and `setpairwiseparameters!(M, newpars::Vector{Float64})`
*   `makecoded(M, Y)`
*   `centeringterms(M, kind::Union{Nothing,CenteringKinds})`
*   `pseudolikelihood(M)`
*   `negpotential(M)`
*   `fullPMF(M; replicates=nothing, force::Bool)`
*   `marginalprobabilities(M; replicates, force::Bool)`
*   `conditionalprobabilities(M; vertices, replicates)`
*   `sample(M, k::Int, method::SamplingMethods, replicate::Int, average::Bool, start, 
    burnin::Int, verbose::Bool)`

The `sample()` function is a wrapper for a variety of random sampling algorithms enumerated
in `SamplingMethods`.

# Examples
```jldoctest
julia> M = ALsimple(Graph(4,4));
julia> typeof(M)
ALsimple{CenteringKinds,Int64,Nothing}
julia> isa(M, AbstractAutologisticModel)
true
```
"""
abstract type AbstractAutologisticModel end

# === Getting/setting parameters ===============================================
getparameters(M::AbstractAutologisticModel) = [getparameters(M.unary); getparameters(M.pairwise)]
getunaryparameters(M::AbstractAutologisticModel) = getparameters(M.unary)
getpairwiseparameters(M::AbstractAutologisticModel) = getparameters(M.pairwise)
function setparameters!(M::AbstractAutologisticModel, newpars::Vector{Float64})
    p, q = (length(getunaryparameters(M)), length(getpairwiseparameters(M)))
    @assert length(newpars) == p + q "newpars has wrong length"
    setparameters!(M.unary, newpars[1:p])
    setparameters!(M.pairwise, newpars[p+1:p+q])
    return newpars
end
function setunaryparameters!(M::AbstractAutologisticModel, newpars::Vector{Float64})
    setparameters!(M.unary, newpars)
end
function setpairwiseparameters!(M::AbstractAutologisticModel, newpars::Vector{Float64})
    setparameters!(M.pairwise, newpars)
end


# A fcn to convert Boolean responses into coded values.
# Returns a 2D array of Float64.  If Y is not supplied, use the responses stored
# in the 1st argument.
function makecoded(M::AbstractAutologisticModel, Y=nothing)
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
# centeringterms(M) returns an Array{Float64,2} of the same dimension as 
# M.unary, giving the centering adjustments for AutologisticModel M.
# centeringterms(M,kind) returns the centering adjustment that would be 
#   if centering were of type kind.
# TODO: consider performance implications of calculating this each time instead
# of storing the value.
function centeringterms(M::AbstractAutologisticModel, kind::Union{Nothing,CenteringKinds}=nothing) 
    k = kind==nothing ? M.centering : kind
    if k == none
        return fill(0.0, size(M.unary))
    elseif k == onehalf
        return fill(0.5, size(M.unary))
    elseif k == expectation
        lo, hi = M.coding
        α = M.unary[:,:] 
        num = lo*exp.(lo*α) + hi*exp.(hi*α)
        denom = exp.(lo*α) + exp.(hi*α)
        return num./denom
    else 
        error("centering kind not recognized")
    end
end


# === pseudolikelihood =========================================================
# pseudolikelihood(M) computes the negative log pseudolikelihood for the given 
# AutologisticModel with its responses.  Returns a Float64.
function pseudolikelihood(M::AbstractAutologisticModel)
    out = 0.0
    Y = makecoded(M)
    mu = centeringterms(M)
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
# TODO: clean up for allocations/speed. Based on experience with sample(), might
#       want to loop explicitly.
function negpotential(M::AbstractAutologisticModel)
    Y = makecoded(M)
    m = size(Y,2)
    out = Array{Float64}(undef, m)
    α = M.unary[:,:]
    μ = centeringterms(M)
    for j = 1:m
        Λ = M.pairwise[:,:,j]
        out[j] = Y[:,j]'*α[:,j] - Y[:,j]'*Λ*μ[:,j]  + Y[:,j]'*Λ*Y[:,j]/2
    end
    return out
end


# === fullPMF ==================================================================
"""
    fullPMF(M::AbstractAutologisticModel; replicates=nothing, force::Bool=false)

Compute the PMF of an AbstractAutologisticModel, and return a `NamedTuple` `(:table, :partition)`.

For an AutologisticModel with ``n`` observations and ``m`` replicates, `:table` is a ``2^n×(n+1)×m`` 
array of `Float64`. Each page of the 3D array holds a probability table for a replicate.  
Each row of the table holds a specific configuration of the responses, with the 
corresponding probability in the last column.  In the ``m=1`` case,  `:table` is a 2D array.

Output `:partition` is a vector of normalizing constant (a.k.a. partition function) values.
In the ``m=1`` case, it is a scalar `Float64`.

# Arguments
- `M::AbstractAutologisticModel`: an autologistic model.
- `replicates=nothing`: indices of specific replicates from which to obtain the output. By 
  default, all replicates are used.
- `force::Bool=false`: calling the function with ``n>20`` will throw an error unless 
  `force=true`. 

# Examples
```jldoctest
julia> M = ALRsimple(Graph(3,0),ones(3,1));
julia> pmf = fullPMF(M);
julia> pmf.table
8×4 Array{Float64,2}:
 -1.0  -1.0  -1.0  0.125
 -1.0  -1.0   1.0  0.125
 -1.0   1.0  -1.0  0.125
 -1.0   1.0   1.0  0.125
  1.0  -1.0  -1.0  0.125
  1.0  -1.0   1.0  0.125
  1.0   1.0  -1.0  0.125
  1.0   1.0   1.0  0.125
julia> pmf.partition
 8.0
```
"""
function fullPMF(M::AbstractAutologisticModel; replicates=nothing, force::Bool=false)
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
        μ = centeringterms(M)[:,r]
        unnormalized = mapslices(v -> exp.(v'*α - v'*Λ*μ + v'*Λ*v/2), configs, dims=2)
        partition[i] = sum(unnormalized)
        T[:,n+1,i] = unnormalized / partition[i]
    end
    if length(replicates)==1
        T  = dropdims(T,dims=3)
        partition = partition[1]
    end
    return (table=T, partition=partition)
end


# ***TODO: documentation***
#Returns an n-by-m array (or an n-vector if  m==1). The [i,j]th element is the 
#marginal probability of the high state in the ith variable at the jth replciate.
function marginalprobabilities(M::AbstractAutologisticModel; replicates=nothing, force::Bool=false)
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
    hi = M.coding[2]
    out = zeros(n,length(replicates))

    tbl = fullPMF(M).table

    for j = 1:length(replicates)
        r = replicates[j]
        for i = 1:n
            out[i,j] = sum(mapslices(x -> x[i]==hi ? x[n+1] : 0.0, tbl[:,:,r], dims=2))
        end
    end
    if length(replicates) == 1
        return vec(out)
    end
    return out
end


# Compute the conditional probability that variables take the high state, given the
# current values of all of their neighbors. If vertices or replicates are provided,
# the results are only computed for the desired variables & replicates.  Otherwise
# results are computed for all variables and replicates.
# TODO: optimize for speed/efficiency
function conditionalprobabilities(M::AbstractAutologisticModel; vertices=nothing, replicates=nothing)
    n, m = size(M.unary)
    if vertices==nothing
        vertices = 1:n
    end
    if replicates==nothing
        replicates = 1:m
    end
    out = zeros(Float64, length(vertices), length(replicates))
    Y = makecoded(M)
    μ = centeringterms(M)
    lo, hi = M.coding
    adjlist = M.pairwise.G.fadjlist

    for j = 1:length(replicates)
        r = replicates[j]
        for i = 1:length(vertices)
            v = vertices[i]
            # get neighbor sum
            ns = 0.0
            for ix in adjlist[v]
                ns = ns + M.pairwise[v,ix,r] * (Y[ix,r] - μ[ix,r])
            end
            # get cond prob
            loval = exp(lo*(M.unary[v,r] + ns))
            hival = exp(hi*(M.unary[v,r] + ns))
            if hival == Inf
                out[i,r] = 1.0
            else
                out[i,r] = hival / (loval + hival)
            end
        end
    end
    return out
end


"""
    sample(
        M::AbstractAutologisticModel, 
        k::Int = 1;
        method::SamplingMethods = Gibbs,
        replicate::Int = 1, 
        average::Bool = false, 
        start = nothing, 
        burnin::Int = 0,
        verbose::Bool = false
    )

Draws `k` random samples from autologistic model `M`. For a model `M` with `n` 
vertices in its graph, the return value is an `Array{Float64,1}` of length `n` when 
`average=true`, and an `n×k` `Array{Float64,2}` when `average=false`.

# Keyword Arguments

**`method`** is a member of the enum [`SamplingMethods`](@ref), specifying which sampling
method will be used.  The default uses Gibbs sampling.  Where feasible, it is recommended 
to use one of the perfect sampling alternatives. See [`SamplingMethods`](@ref) for more.

**`replicate`** specifies which replicate of the model to use for sampling. 

**`average`** controls whether the return value is the proportion of "high" samples at each 
(when `average=true`) or the full set of samples (when `average=false`). Note that when the
coding is not (0,1), the the return value is the estimated probability of getting a "high"
outcome, **not** the arithmetic average of the samples.

**`start`** allows a starting configuration of the random variables to be provided. Only
used if `method=Gibbs`. Any vector with two unique values can be used as `start`.

**`burnin`** specifies the number of initial samples to discard from the results.  Only used
if `method=Gibbs`.

**`verbose`** controls output to the console.  If `true`, intermediate information about 
sampling progress is printed to the console. Otherwise no output is shown.

# Examples
```jldoctest
julia> M = ALsimple(Graph(4,4));
julia> M.coding = (-2,3);
julia> r = sample(M,10);
julia> size(r)
(4, 10)
julia> unique(r)
2-element Array{Float64,1}:
  3.0
 -2.0
```
"""
function sample(M::AbstractAutologisticModel, k::Int = 1; method::SamplingMethods = Gibbs, 
                replicate::Int = 1, average::Bool = false, start = nothing, burnin::Int = 0,
                verbose::Bool = false)
    if k < 1 
        error("k must be positive") 
    end
    if replicate < 1 || replicate > size(M.unary, 2) 
        error("replicate must be between 1 and the number of replicates")
    end
    if burnin < 0 
        error("burnin must be nonnegative") 
    end
    lo = Float64(M.coding[1])
    hi = Float64(M.coding[2])
    Y = vec(makecoded(M, M.responses[:,replicate]))  #TODO: be cetain M isn't mutated.
    Λ = M.pairwise[:,:,replicate]   
    α = M.unary[:,replicate]
    μ = centeringterms(M)[:,replicate]
    n = length(α)
    adjlist = M.pairwise.G.fadjlist
    if method == Gibbs
        if start == nothing
            start = rand([lo, hi], n)
        else
            start = vec(makecoded(M, makebool(start)))
        end
        return gibbssample(lo, hi, Y, Λ, adjlist, α, μ, n, k, average, start, burnin, verbose)
    elseif method == perfect_reuse_samples
        return cftp_reuse_samples(lo, hi, Λ, adjlist, α, μ, n, k, average, verbose)
    elseif method == perfect_reuse_seeds
        return cftp_reuse_seeds(lo, hi, Λ, adjlist, α, μ, n, k, average, verbose)
    elseif method == perfect_bounding_chain
        return cftp_bounding_chain(lo, hi, Λ, adjlist, α, μ, n, k, average, verbose)
    elseif method == perfect_read_once
        return cftp_read_once(lo, hi, Λ, adjlist, α, μ, n, k, average, verbose)
    end
end
