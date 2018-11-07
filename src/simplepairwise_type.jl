#----- SimplePairwise ----------------------------------------------------------
# Association matrix is a parameter times the adjacency matrix.
# For this case, the association matrix Λ can not be different for different
# replicates.  So while we internally treat it like an n-by-n-by-m matrix, just
# return a 2D n-by-n matrix to the user. 

# ***TODO*** 
# [x] Make constructor with graph only (initialize λ to zero) 

# Type definition
#    Note: made this mutable so I could set λ.  Not sure how best to handle.
#    Note: decided that *all parameters should be Vector{Float64}* for type 
#          stability (even though here λ is scalar).
mutable struct SimplePairwise <: AbstractPairwise
	λ::Vector{Float64}
	G::SimpleGraph{Int}
	replicates::Int
	function SimplePairwise(lam, g, m)
		if length(lam) !== 1
			error("SimplePairwise: λ must have length 1")
		end
		if m < 1
			error("SimplePairwise: replicates must be positive")
		end
		new(lam, g, m)
	end
end

# Constructors
# - If provide only a graph, set λ = 0.
# - If provide only an integer, set λ = 0 and make a totally disconnected graph.
# - If provide a graph and a scalar, convert the scalar to a length-1 vector.
SimplePairwise(G::SimpleGraph) = SimplePairwise([0.0], G, 1)
SimplePairwise(G::SimpleGraph, m::Int) = SimplePairwise([0.0], G, m)
SimplePairwise(n::Int) = SimplePairwise(0, SimpleGraph(n), 1)
SimplePairwise(n::Int, m::Int) = SimplePairwise(0, SimpleGraph(n), m)
SimplePairwise(λ::Real, G::SimpleGraph) = SimplePairwise([(Float64)(λ)], G, 1)
SimplePairwise(λ::Real, G::SimpleGraph, m::Int) = SimplePairwise([(Float64)(λ)], G, m)

# Methods required for AbstractArray interface (AbstractPairwise <: AbstractArray)
# For this case, since the association matrix doesn't vary with replicates, just
# return a 2D matrix as the values.  Also allow indexing using 2 indices.  If 3 
# indices used, need to check for the 3rd one being out of bounds.
Base.size(p::SimplePairwise) = (nv(p.G), nv(p.G), p.replicates)
Base.values(p::SimplePairwise) = p.λ[1] * adjacency_matrix(p.G)  
function Base.getindex(p::SimplePairwise, I::Vararg{Int,3})
	if I[3] > p.replicates
		error("SimplePairwise getindex: 3rd index is larger than replicates")
	end
	return p.λ[1] * adjacency_matrix(p.G)[ I[[1, 2]] ]
end
Base.setindex!(p::SimplePairwise, v::Real, I::Vararg{Int,3}) = 
    error("Pairwise values cannot be set directly. Use setparameters! instead.")
Base.getindex(p::SimplePairwise, I::Vararg{Int,2}) = p.λ[1] * adjacency_matrix(p.G)[I]
Base.setindex!(p::SimplePairwise, v::Real, I::Vararg{Int,2}) = 
    error("Pairwise values cannot be set directly. Use setparameters! instead.")

# Methods required for AbstractPairwise interface
getparameters(p::SimplePairwise) = p.λ
function setparameters!(p::SimplePairwise, newpar::Vector{Float64})
    p.λ = newpar
end
