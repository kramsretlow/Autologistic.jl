#----- SimplePairwise ----------------------------------------------------------
# Association matrix is a parameter times the adjacency matrix.

# ***TODO*** 
# [x] Make constructor with graph only (initialize λ to zero) 

# Type definition
#    Note: made this mutable so I could set λ.  Not sure how best to handle.
mutable struct SimplePairwise <: AbstractPairwise
	λ::Float64
	G::SimpleGraph{Int}
end

# Constructors
# - If provide only a graph, set λ = 0.
# - If provide only an integer, set λ = 0 and make a totally disconnected graph.
SimplePairwise(G::SimpleGraph) = SimplePairwise(0, G)
SimplePairwise(n::Int) = SimplePairwise(0, SimpleGraph(n))

# Methods required for AbstractArray interface (AbstractPairwise <: AbstractArray)
Base.size(p::SimplePairwise) = (nv(p.G), nv(p.G))
Base.getindex(p::SimplePairwise, I::Vararg{Int,2}) = p.λ * adjacency_matrix(p.G)[I]
Base.values(p::SimplePairwise) = p.λ * adjacency_matrix(p.G)  #is it more efficient vs. not defining it?
Base.setindex!(p::SimplePairwise, v::Real, I::Vararg{Int,2}) = error("***TODO (not allowed msg)***")

# Methods required for AbstractPairwise interface
getparameters(p::SimplePairwise) = p.λ
function setparameters!(p::SimplePairwise, newpar::Real)
    p.λ = newpar
end
