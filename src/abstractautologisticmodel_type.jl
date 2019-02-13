"""
    AbstractAutologisticModel

Abstract type representing autologistic models.  This type has methods defined for most
operations one will want to perform, so that concrete subtypes should not have to define
too many methods unless more specialized and efficient algorithms for the specific subtype.

All concrete subtypes should have the following fields:

*   `responses::Array{Bool,2}` -- The binary observations. Rows are for nodes in the 
    graph, and columns are for independent (vector) observations.  It is a 2D array even if 
    there is only one observation.
*   `unary<:AbstractUnaryParameter` -- Specifies the unary part of the model.
*   `pairwise<:AbstractPairwiseParameter`  -- Specifies the pairwise part of the model 
    (including the graph).
*   `centering<:CenteringKinds` -- Specifies the form of centering used, if any.
*   `coding::Tuple{T,T} where T<:Real` -- Gives the numeric coding of the responses.
*   `labels::Tuple{String,String}` -- Provides names for the high and low states.
*   `coordinates<:SpatialCoordinates` -- Provides 2D or 3D coordinates for each vertex in 
    the graph.

The following functions are defined for the abstract type, and are considered part of the 
type's interface (in this list, `M` of type inheriting from `AbstractAutologisticModel`).

*   `getparameters(M)` and `setparameters!(M, newpars::Vector{Float64})`
*   `getunaryparameters(M)` and `setunaryparameters!(M, newpars::Vector{Float64})`
*   `getpairwiseparameters(M)` and `setpairwiseparameters!(M, newpars::Vector{Float64})`
*   `makecoded(M, Y)`
*   `centeringterms(M, kind::Union{Nothing,CenteringKinds})`
*   `pseudolikelihood(M)`
*   `negpotential(M)`
*   `fullPMF(M; indices, force::Bool)`
*   `marginalprobabilities(M; indices, force::Bool)`
*   `conditionalprobabilities(M; vertices, indices)`
*   `sample(M, k::Int, method::SamplingMethods, indices::Int, average::Bool, start, 
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


# === show methods =============================================================

Base.show(io::IO, m::AbstractAutologisticModel) = print(io, "$(typeof(m))")

function Base.show(io::IO, ::MIME"text/plain", m::AbstractAutologisticModel)
    print(io, "Autologistic model of type $(typeof(m)), \n",
              "with $(size(m.unary,1)) vertices, $(size(m.unary, 2)) ",
              "$(size(m.unary,2)==1 ? "observation" : "observations") ", 
              "and fields:\n",
              showfields(m,2))
end

function showfields(m::AbstractAutologisticModel, leadspaces=0)
    return repeat(" ", leadspaces) * 
           "(**Autologistic.showfields not implemented for $(typeof(m))**)\n"
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

    # Loop through observations
    for j = 1:size(Y)[2]
        y = Y[:,j];                     #-Current observation's values.
        α = M.unary[:,j]                #-Current observation's unary parameters.
        μ = mu[:,j]                     #-Current observation's centering terms.
        Λ = M.pairwise[:,:,j]           #-Current observation's assoc. matrix.
        s = α + Λ*(y - μ)               #-(λ-weighted) neighbour sums + unary.
        logPL = sum(y.*s - log.(exp.(lo*s) + exp.(hi*s)))
        out = out - logPL               #-Subtract this rep's log PL from total.
    end

    return out

end


# === negpotential function ====================================================
# negpotential(M) returns an m-vector of Float64 negpotential values, where 
# m is the number of observations found in M.responses.
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
    fullPMF(M::AbstractAutologisticModel; indices=1:size(M.unary,2), force::Bool=false)

Compute the PMF of an AbstractAutologisticModel, and return a `NamedTuple` `(:table, :partition)`.

For an AutologisticModel with ``n`` variables and ``m`` observations, `:table` is a ``2^n×(n+1)×m`` 
array of `Float64`. Each page of the 3D array holds a probability table for an observation.  
Each row of the table holds a specific configuration of the responses, with the 
corresponding probability in the last column.  In the ``m=1`` case,  `:table` is a 2D array.

Output `:partition` is a vector of normalizing constant (a.k.a. partition function) values.
In the ``m=1`` case, it is a scalar `Float64`.

# Arguments
- `M`: an autologistic model.
- `indices`: indices of specific observations from which to obtain the output. By 
  default, all observations are used.
- `force`: calling the function with ``n>20`` will throw an error unless 
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
function fullPMF(M::AbstractAutologisticModel; indices=1:size(M.unary,2), 
                 force::Bool=false)
    n, m = size(M.unary)
    nc = 2^n
    if n>20 && !force
        error("Attempting to tabulate a PMF with more than 2^20 configurations."
              * "\nIf you really want to do this, set force=true.")
    end
    if minimum(indices)<1 || maximum(indices)>m 
        error("observation index out of bounds")
    end
    lo = M.coding[1]
    hi = M.coding[2]
    T = zeros(nc, n+1, length(indices))
    configs = zeros(nc,n)
    partition = zeros(m)
    for i in 1:n
        inner = [repeat([lo],Int(nc/2^i)); repeat([hi],Int(nc/2^i))]
        configs[:,i] = repeat(inner , 2^(i-1) )
    end
    for i in 1:length(indices)
        r = indices[i]
        T[:,1:n,i] = configs
        α = M.unary[:,r]
        Λ = M.pairwise[:,:,r]
        μ = centeringterms(M)[:,r]
        unnormalized = mapslices(v -> exp.(v'*α - v'*Λ*μ + v'*Λ*v/2), configs, dims=2)
        partition[i] = sum(unnormalized)
        T[:,n+1,i] = unnormalized / partition[i]
    end
    if length(indices)==1
        T  = dropdims(T,dims=3)
        partition = partition[1]
    end
    return (table=T, partition=partition)
end

# === maximum likelihood estimation ============================================
function loglikelihood(M::AbstractAutologisticModel; force::Bool=false)
    parts = fullPMF(M, force=force).partition
    return sum(negpotential(M) - log.(parts))
end

function negloglik!(θ::Vector{Float64}, M::AbstractAutologisticModel; force::Bool=false)
    setparameters!(M,θ)
    return -loglikelihood(M, force=force)
end

#TODO: documentation, tests
function fit_ml!(M::AbstractAutologisticModel; 
                 start=zeros(length(getparameters(M))), 
                 sigdigits=3,
                 force::Bool=false,
                 verbose::Bool=false,
                 g_tol=1e-8,
                 allow_f_increases=true,
                 iterations=1000,
                 time_limit=NaN)

    originalparameters = getparameters(M)
    npar = length(originalparameters)

    opts = Optim.Options(show_trace=verbose, allow_f_increases=allow_f_increases,
                         time_limit=time_limit, g_tol=g_tol)
    if verbose 
        println("Calling Optim.optimize with BFGS method...")
    end
    out = optimize(θ -> negloglik!(θ,M,force=force), start, BFGS(), opts)
    if !Optim.converged(out)
        setparameters!(M, originalparameters)
        @warn "Optim.optimize did not converge. Model parameters have not been changed."
        return (estimate="didn't converge", se="didn't converge", pvalues="didn't converge",
                CIs="didn't converge", Hinv="didn't converge", optimresults=out)
    end
    
    if verbose
        println("Approximating the Hessian at the MLE...")
    end
    H = hess(θ -> negloglik!(θ,M), out.minimizer)
    if verbose
        println("Getting standard errors...")
    end
    Hinv = inv(H)
    SE = round.(sqrt.(LinearAlgebra.diag(Hinv)), sigdigits=sigdigits)
    
    pvals = zeros(npar)
    CIs = [(0.0, 0.0) for i=1:npar]
    for i = 1:npar
        N = Normal(0,SE[i])
        pvals[i] = round(2*(1 - cdf(N, abs(out.minimizer[i]))), sigdigits=sigdigits)
        CIs[i] = round.(out.minimizer[i] .+ (quantile(N,0.025), quantile(N,0.975)), 
                        sigdigits=sigdigits)
    end

    setparameters!(M, out.minimizer)
    if verbose
        println("Completed successfully. Output is a named tuple " * 
                "(:estimate, :se, :pvalues, :CIs, :Hinv, :optimresults)")
    end
    return (estimate=round.(out.minimizer, sigdigits=sigdigits), se=SE, pvalues=pvals, 
            CIs=CIs, Hinv=Hinv, optimresults=out)
end


# ***TODO: documentation***
#Returns an n-by-m array (or an n-vector if  m==1). The [i,j]th element is the 
#marginal probability of the high state in the ith variable at the jth replciate.
function marginalprobabilities(M::AbstractAutologisticModel; indices=1:size(M.unary,2), 
                               force::Bool=false)
    n, m = size(M.unary)
    nc = 2^n
    if n>20 && !force
        error("Attempting to tabulate a PMF with more than 2^20 configurations."
              * "\nIf you really want to do this, set force=true.")
    end
    if minimum(indices)<1 || maximum(indices)>m 
        error("observation index out of bounds")
    end
    hi = M.coding[2]
    out = zeros(n,length(indices))

    tbl = fullPMF(M).table

    for j = 1:length(indices)
        r = indices[j]
        for i = 1:n
            out[i,j] = sum(mapslices(x -> x[i]==hi ? x[n+1] : 0.0, tbl[:,:,r], dims=2))
        end
    end
    if length(indices) == 1
        return vec(out)
    end
    return out
end


# Compute the conditional probability that variables take the high state, given the
# current values of all of their neighbors. If vertices or indices are provided,
# the results are only computed for the desired variables & observations.  Otherwise
# results are computed for all variables and observations.
# TODO: optimize for speed/efficiency
function conditionalprobabilities(M::AbstractAutologisticModel; vertices=1:size(M.unary)[1], 
                                  indices=1:size(M.unary,2))
    n, m = size(M.unary)
    if minimum(vertices)<1 || maximum(vertices)>n 
        error("vertices index out of bounds")
    end
    if minimum(indices)<1 || maximum(indices)>m 
        error("observation index out of bounds")
    end
    out = zeros(Float64, length(vertices), length(indices))
    Y = makecoded(M)
    μ = centeringterms(M)
    lo, hi = M.coding
    adjlist = M.pairwise.G.fadjlist

    for j = 1:length(indices)
        r = indices[j]
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
        indices = 1:size(M.unary,2), 
        average::Bool = false, 
        start = nothing, 
        burnin::Int = 0,
        verbose::Bool = false
    )

Draws `k` random samples from autologistic model `M`, and either returns the samples 
themselves, or the estimated probabilities of observing the "high" level at each vertex.

If the model has more than one observation, then `k` samples are drawn for each observation.
To restrict the samples to a subset of observations, use argument `indices`. 

For a model `M` with `n` vertices in its graph:

*   When `average=false`, the return value is `n` × `length(indices)` × `k`, with singleton
    dimensions dropped. 
*   When `average=true`, the return value is `n`  × `length(indices)`, with singleton
    dimensions dropped.

# Keyword Arguments

**`method`** is a member of the enum [`SamplingMethods`](@ref), specifying which sampling
method will be used.  The default uses Gibbs sampling.  Where feasible, it is recommended 
to use one of the perfect sampling alternatives. See [`SamplingMethods`](@ref) for more.

**`indices`** gives the indices of the observation to use for sampling. The default is all
indices, in which case each sample is of the same size as `M.responses`. 

**`average`** controls the form of the output. When `average=true`, the return value is the 
proportion of "high" samples at each vertex. (Note that this is **not** actually the
arithmetic average of the samples, unless the coding is (0,1). Rather, it is an estimate of 
the probability of getting a "high" outcome.)  When `average=false`, the full set of samples
is returned. 

**`start`** allows a starting configuration of the random variables to be provided. Only
used if `method=Gibbs`. Any vector with two unique values can be used as `start`. By default
a random configuration is used.

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
julia> sort(unique(r))
2-element Array{Float64,1}:
 -2.0
  3.0
```
"""
function sample(M::AbstractAutologisticModel, k::Int = 1; method::SamplingMethods = Gibbs, 
                indices=1:size(M.unary,2), average::Bool = false, start = nothing, 
                burnin::Int = 0, verbose::Bool = false)
    
    # Create storage object 
    n, m = size(M.unary)
    nidx = length(indices)
    if k < 1 
        error("k must be positive") 
    end
    if  minimum(indices) < 1 || maximum(indices) > m 
        error("indices must be between 1 and the number of observations")
    end
    if burnin < 0 
        error("burnin must be nonnegative") 
    end
    if average
        out = zeros(Float64, n, nidx)
    else
        out = zeros(Float64, n, nidx, k)
    end

    # Call the sampling function for each index. Give details if verbose=true.
    for i = 1:nidx
        if verbose
            println("== Sampling observation $(indices[i]) ==")
        end
        out[:,i,:] = sample_one_index(M, k, method=method, 
                                      index=indices[i], average=average,
                                      start=start, burnin=burnin, verbose=verbose)
    end

    # Return
    if nidx==1 && !average
        out = dropdims(out,dims=2)
    end
    if size(out,2) == size(out,3) == 1
        return out[:]
    else
        return out
    end

    # Update tests
end

function sample_one_index(M::AbstractAutologisticModel, k::Int = 1; 
                     method::SamplingMethods = Gibbs, index::Int = 1, average::Bool = false, 
                     start = nothing, burnin::Int = 0, verbose::Bool = false)
    lo = Float64(M.coding[1])
    hi = Float64(M.coding[2])
    Y = vec(makecoded(M, M.responses[:,index]))  #TODO: be cetain M isn't mutated.
    Λ = M.pairwise[:,:,index]   
    α = M.unary[:,index]
    μ = centeringterms(M)[:,index]
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


