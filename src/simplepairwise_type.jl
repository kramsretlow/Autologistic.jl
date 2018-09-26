#----- SimplePairwise ----------------------------------------------------------
# Association matrix is a parameter times the adjacency matrix.
#    Note: made this mutable so I could set λ.  Not sure how best to handle.
mutable struct SimplePairwise{T<:Real} <: AbstractPairwise
	λ::T
	G::SimpleGraph{Int}
end
# Methods required for AbstractArray interface
Base.size(p::SimplePairwise) = (nv(p.G), nv(p.G))
Base.getindex(p::SimplePairwise, I::Vararg{Int,2}) = p.λ * adjacency_matrix(p.G)[I]
Base.values(p::SimplePairwise) = p.λ * adjacency_matrix(p.G)  #is it more efficient vs. not defining it?
Base.setindex!(p::SimplePairwise, v::Real, I::Vararg{Int,2}) = error("***TODO (not allowed msg)***")
# Methods required for AbstractPairwise interface
getparameters(p::SimplePairwise) = p.λ
function setparameters!(p::SimplePairwise, newpar::Real)
    p.λ = newpar
end
