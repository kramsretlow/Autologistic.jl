#----- FullUnary ---------------------------------------------------------------
# A unary part with one parameter per variable.
struct FullUnary{T<:Vector{<:Real}} <: AbstractUnary
	#TODO: constructors
    α::T
end
# Methods required for AbstractArray interface
Base.size(u::FullUnary) = (lastindex(u.α),)
Base.getindex(u::FullUnary, i::Int) = u.α[i]
Base.setindex!(u::FullUnary, v::Real, i::Int) = (u.α[i] = v)
# Methods required for AbstractUnary interface
getparameters(u::FullUnary) = values(u)
function setparameters!(u::FullUnary, newpars::Vector{<:Real})
    # Note, should check dimension match...
    # Note, not implementing subsetting etc., can only replace the whole vector.
    u[:] = newpars
end
