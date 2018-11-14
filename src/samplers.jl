# Take the strategy of using a single user-facing function sample() that has an
# argument method<:SamplingMethods (an enum).  Use the enum to do argument 
# checking and submit the work to the specialized functions (currntly gibbssample()
# and perfectsample()).

function sample(M::ALmodel, k::Int=1; method::SamplingMethods=Gibbs, replicate::Int=1, 
                average::Bool=true, start=nothing, burnin::Int=0, verbose::Bool=false)
    # Common checks here.
    if method==Gibbs
        #checks here... handle e.g. start==nothing
        return gibbssample(M, k, replicate, average, start, burnin, verbose)
    elseif method==perfect
        #checks here... 
        return perfectsample(M, k, replicate, average, verbose)
    end
end

function gibbssample(M::ALmodel, k::Int, replicate::Int, average::Bool, 
                     start::Vector{Float64}, burnin::Int, verbose::Bool)

end

function perfectsample(M::ALmodel, k::Int, replicate::Int, average::Bool, verbose::Bool)

end