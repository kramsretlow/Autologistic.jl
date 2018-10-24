#----- FullUnary ---------------------------------------------------------------
# A unary part with one parameter per variable.
struct FullUnary <: AbstractUnary
    α::Vector{Float64}
end

# Constructors
FullUnary(n::Int) = FullUnary(Vector{Float64}(undef,n))

# Methods required for AbstractArray interface
Base.size(u::FullUnary) = (lastindex(u.α),)
Base.getindex(u::FullUnary, i::Int) = u.α[i]
Base.setindex!(u::FullUnary, v::Real, i::Int) = (u.α[i] = v)
# Methods required for AbstractUnary interface
getparameters(u::FullUnary) = values(u)
function setparameters!(u::FullUnary, newpars::Vector{Float64})
    # Note, should check dimension match...
    # Note, not implementing subsetting etc., can only replace the whole vector.
    u[:] = newpars
end
