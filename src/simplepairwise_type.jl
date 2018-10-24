#----- SimplePairwise ----------------------------------------------------------
# Association matrix is a parameter times the adjacency matrix.

# ***TODO*** 
# [x] Make constructor with graph only (initialize λ to zero) 

# Type definition
#    Note: made this mutable so I could set λ.  Not sure how best to handle.
#    Note: decided that *all parameters should be Vector{Float64}* for type 
#          stability (even though here λ is scalar).
mutable struct SimplePairwise <: AbstractPairwise
	λ::Vector{Float64}
	G::SimpleGraph{Int}
	function SimplePairwise(lam, g)
		if length(lam) !== 1
			error("SimplePairwise: λ must have length 1")
		end
		new(lam, g)
	end
end

# Constructors
# - If provide only a graph, set λ = 0.
# - If provide only an integer, set λ = 0 and make a totally disconnected graph.
# - If provide a graph and a scalar, convert the scalar to a length-1 vector.
SimplePairwise(G::SimpleGraph) = SimplePairwise([0.0], G)
SimplePairwise(n::Int) = SimplePairwise(0, SimpleGraph(n))
SimplePairwise(λ::Real, G::SimpleGraph) = SimplePairwise([(Float64)(λ)], G)

# Methods required for AbstractArray interface (AbstractPairwise <: AbstractArray)
Base.size(p::SimplePairwise) = (nv(p.G), nv(p.G))
Base.getindex(p::SimplePairwise, I::Vararg{Int,2}) = p.λ[1] * adjacency_matrix(p.G)[I]
Base.values(p::SimplePairwise) = p.λ[1] * adjacency_matrix(p.G)  #is it more efficient vs. not defining it?
Base.setindex!(p::SimplePairwise, v::Real, I::Vararg{Int,2}) = error("***TODO (not allowed msg)***")

# Methods required for AbstractPairwise interface
getparameters(p::SimplePairwise) = p.λ
function setparameters!(p::SimplePairwise, newpar::Vector{Float64})
    p.λ = newpar
end
