# Take the strategy of using a single user-facing function sample() that has an
# argument method<:SamplingMethods (an enum).  Use the enum to do argument 
# checking and submit the work to the specialized functions (currntly gibbssample()
# and perfectsample()).

function sample(M::ALmodel, k::Int=1; method::SamplingMethods=Gibbs, replicate::Int=1, 
                average::Bool=true, start=nothing, burnin::Int=0, verbose::Bool=false)
    if k < 1 
        error("k must be positive") 
    end
    if replicate<1 || replicate>size(M.unary,2) 
        error("replicate must be between 1 and the number of replicates")
    end
    if burnin<0 
        error("burnin must be nonnegative") 
    end
    lo = Float64(M.coding[1])
    hi = Float64(M.coding[2])
    Y = Vector{Float64}(vec(makecoded(M,M.responses[:,replicate])))
    Λ = M.pairwise[:,:,replicate]
    α = vec(M.unary[:,replicate])
    μ = vec(centering_adjustment(M)[:,replicate])
    n = length(α)
    if method==Gibbs
        if start==nothing
            start = Vector{Float64}(rand([M.coding[1] M.coding[2]], size(M.unary,1)))
        else
            start = Vector{Float64}(vec(makecoded(M,makebool(start))))
        end
        return gibbssample(lo, hi, Y, Λ, α, μ, n, k, average, start, burnin, verbose)
    elseif method==perfect
        return perfectsample(lo, hi, Y, Λ, α, μ, n, k, average, verbose)
    end
end

function gibbssample(lo::Float64, hi::Float64, Y::Vector{Float64}, Λ::T, 
                     α::Vector{Float64}, μ::Vector{Float64}, n::Int, k::Int, average::Bool, 
                     start::Vector{Float64}, burnin::Int, verbose::Bool) where T<:AbstractMatrix
    temp = zeros(n, k)
    for j = 1:k
        for i = 1:n
            nbrsum = Λ[i,:]'*(Y-μ)
            etalo = exp(lo*(α[i] + nbrsum))
            etahi = exp(hi*(α[i] + nbrsum))
            p_i = ifelse(etahi==Inf, 1.0, etahi/(etalo + etahi))   #-handle rare overflows.
            Y[i] = ifelse(rand()<p_i, hi, lo)
        end
        temp[:,j] = Y
        if verbose 
            println("finished draw $(j) of $(k)") 
        end
    end
    out = ifelse(burnin==0, temp, temp[burnin+1:end])
    if average
        return sum(out.==hi,dims=2)/(k-burnin)
    else
        return out
    end
end

#=
function perfectsample(lo::Float64, hi::Float64, Y::Vector{Float64}, Λ::T, 
                       α::Vector{Float64}, μ::Vector{Float64}, n::Int, k::Int, 
                       average::Bool, verbose::Bool) where T<:AbstractMatrix

end
=#