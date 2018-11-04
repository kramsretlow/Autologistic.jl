#----- FullUnary ---------------------------------------------------------------
# A unary part with one parameter per variable per replicate.
struct FullUnary <: AbstractUnary
    α::Array{Float64,2}
end

# Constructors
function FullUnary(alpha::Array{Float64,1}) 
    return FullUnary( reshape(alpha, (length(alpha),1)) )
end
FullUnary(n::Int) = FullUnary(Array{Float64,2}(undef,n,1))
FullUnary(n::Int,m::Int) = FullUnary(Array{Float64,2}(undef,n,m))

# Methods required for AbstractArray interface
Base.size(u::FullUnary) = size(u.α)
Base.getindex(u::FullUnary, I::Vararg{Int,2}) = u.α[CartesianIndex(I)]
Base.values(u::FullUnary) = u.α  #TODO: determine if can delete this.
Base.setindex!(u::FullUnary, v::Real, I::Vararg{Int,2}) = (u.α[CartesianIndex(I)] = v)
# Methods required for AbstractUnary interface
getparameters(u::FullUnary) = dropdims(reshape(values(u), length(u), 1), dims=2)
function setparameters!(u::FullUnary, newpars::Vector{Float64})
    # Note, should check dimension match?...
    # Note, not implementing subsetting etc., can only replace the whole vector.
    u[:] = newpars  
end

