# Take the strategy of using a single user-facing function sample() that has an
# argument method<:SamplingMethods (an enum).  Use the enum to do argument 
# checking and submit the work to the specialized functions (currently gibbssample()
# and perfectsample()).

function sample(M::ALmodel, k::Int = 1; method::SamplingMethods = Gibbs, replicate::Int = 1, 
                average::Bool = false, start = nothing, burnin::Int = 0, verbose::Bool = false)
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
    μ = centering_adjustment(M)[:,replicate]
    n = length(α)
    adjlist = M.pairwise.G.fadjlist  #**TODO: decide on what bits of G to pass**
    if method == Gibbs
        if start == nothing
            start = rand([lo, hi], n)
        else
            start = vec(makecoded(M, makebool(start)))
        end
        return gibbssample(lo, hi, Y, Λ, adjlist, α, μ, n, k, average, start, burnin, verbose)
    elseif method == perfect
        return perfectsample(lo, hi, Y, Λ, adjlist, α, μ, n, k, average, verbose)
    end
end

# Performance tip: writing the neighbor sum as a loop rather than an inner product
# saved lots of memory allocations.
function nbrsum(Λ, Y, μ, row, nbr)::Float64
    out = 0.0
    for ix in nbr
        out += Λ[row,ix] * (Y[ix] - μ[ix])
    end
    return out
end
function condprob(α, ns, lo, hi, ix)::Float64
    loval = exp(lo*(α[ix] + ns))
    hival = exp(hi*(α[ix] + ns))
    if hival==Inf
        return 1.0
    end
    return hival / (loval + hival)
end
function gibbsstep!(Y, lo, hi, Λ, adjlist, α, μ, n, rng=Random.GLOBAL_RNG)
    for i = 1:n
        ns = nbrsum(Λ, Y, μ, i, adjlist[i])
        p_i = condprob(α, ns, lo, hi, i)
        if rand(rng) < p_i
            Y[i] = hi
        else
            Y[i] = lo
        end
    end
end

function gibbssample(lo::Float64, hi::Float64, Y::Vector{Float64}, 
                     Λ::SparseMatrixCSC{Float64,Int}, adjlist::Array{Array{Int64,1},1},
                     α::Vector{Float64}, μ::Vector{Float64}, n::Int, k::Int, average::Bool, 
                     start::Vector{Float64}, burnin::Int, verbose::Bool)

    temp = average ? zeros(Float64, n) : zeros(Float64, n, k-burnin)
    ns = p_i = 0.0

    for j = 1:k
        gibbsstep!(Y, lo, hi, Λ, adjlist, α, μ, n)
        if average 
            # Performance tip: looping here saves memory allocations. (perhaps until we get
            # an operator like .+=)
            for i in 1:n
                temp[i] = temp[i] + Y[i]
            end
        elseif j > burnin
            for i in 1:n
                temp[i,j-burnin] = Y[i]
            end
        end
        if verbose 
            println("finished draw $(j) of $(k)") 
        end
    end
    if average
        return map(x -> (x - (k-burnin)*lo)/((k-burnin)*(hi-lo)), temp)
    else
        return temp
    end
end


function runepochs!(j, T, L, H, rngL, rngH, seeds, lo, hi, Λ, adjlist, α, μ, n)
    Random.seed!(rngL, seeds[j])
    Random.seed!(rngH, seeds[j])
    if j==1
        for t = -T:0
            gibbsstep!(L, lo, hi, Λ, adjlist, α, μ, n, rngL)
            gibbsstep!(H, lo, hi, Λ, adjlist, α, μ, n, rngH)
        end
    else
        for t = -T*2^(j-1) : -T*2^(j-2)-1
            gibbsstep!(L, lo, hi, Λ, adjlist, α, μ, n, rngL)
            gibbsstep!(H, lo, hi, Λ, adjlist, α, μ, n, rngH)
        end
        runepochs!(j-1, T, L, H, rngL, rngH, seeds, lo, hi, Λ, adjlist, α, μ, n)
    end
end

function perfectsample(lo::Float64, hi::Float64, Y::Vector{Float64}, 
                       Λ::SparseMatrixCSC{Float64,Int}, adjlist::Array{Array{Int64,1},1},
                       α::Vector{Float64}, μ::Vector{Float64}, n::Int, k::Int, 
                       average::Bool, verbose::Bool)

    temp = average ? zeros(Float64, n) : zeros(Float64, n, k)
    T = 2      #-Initial number of time steps to go back.
    maxepochs = 101  #TODO: magic constant
    seeds = ones(UInt32,maxepochs)
    rngL = MersenneTwister()
    rngH = MersenneTwister()
    L = H = zeros(n)

    for rep = 1:k
        seeds .= rand(UInt32, maxepochs)
        coalesce = false    

        # Keep track of the seeds used.  seeds(j) will hold the seed used to generate samples
        # from time -2^(j-1)T to -2^(j-2)+1.  We'll cap j at 100 as T*2^100 is a huge number
        # of time steps. We'll call j the "epoch index." It tells us how far back in time we
        # need to start (specifically, we start at time -T*2^(j-1)).
        j = 0
        while ~coalesce
            j = j + 1
            L .= lo
            H .= hi
            runepochs!(j, T, L, H, rngL, rngH, seeds, lo, hi, Λ, adjlist, α, μ, n)
            coalesce = L==H
            if verbose 
                println("Started from $(-T*2^(j-1)): $(sum(H != L)) elements different.") 
            end
        end
        if !coalesce
            warning("Sampler did not coalesce. Returning NaNs.")  #TODO: fix for average case
            L .= fill(NaN, n)
        end
        if average && coalesce
            for i in 1:n
                temp[i] = temp[i] + L[i]
            end
        else
            for i in 1:n
                temp[i,rep] = L[i]
            end
        end
        if verbose 
            println("finished draw $(j) of $(k)") 
        end
    end
    if average
        return map(x -> (x - k*lo)/(k*(hi-lo)), temp)
    else
        return temp
    end
end
