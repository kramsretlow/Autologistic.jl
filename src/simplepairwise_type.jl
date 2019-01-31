#----- SimplePairwise ----------------------------------------------------------
# Association matrix is a parameter times the adjacency matrix.
# For this case, the association matrix Λ can not be different for different
# observations.  So while we internally treat it like an n-by-n-by-m matrix, just
# return a 2D n-by-n matrix to the user. 

# ***TODO*** 
# [x] Make constructor with graph only (initialize λ to zero) 
# [ ] (here and elsewhere) make setparameters! functions return the modified object?

# Type definition
#    Note: made this mutable so I could set λ.  Not sure how best to handle.
#    Note: decided that *all parameters should be Vector{Float64}* for type 
#          stability (even though here λ is scalar).
mutable struct SimplePairwise <: AbstractPairwiseParameter
	λ::Vector{Float64}
	G::SimpleGraph{Int}
	count::Int
	A::SparseMatrixCSC{Float64,Int64}
	function SimplePairwise(lam, g, m)
		if length(lam) !== 1
			error("SimplePairwise: λ must have length 1")
		end
		if m < 1
			error("SimplePairwise: count must be positive")
		end
		new(lam, g, m, adjacency_matrix(g, Float64))
	end
end

# Constructors
# - If provide only a graph, set λ = 0.
# - If provide only an integer, set λ = 0 and make a totally disconnected graph.
# - If provide a graph and a scalar, convert the scalar to a length-1 vector.
SimplePairwise(G::SimpleGraph, count::Int=1) = SimplePairwise([0.0], G, count)
SimplePairwise(n::Int, count::Int=1) = SimplePairwise(0, SimpleGraph(n), count)
SimplePairwise(λ::Real, G::SimpleGraph) = SimplePairwise([(Float64)(λ)], G, 1)
SimplePairwise(λ::Real, G::SimpleGraph, count::Int) = SimplePairwise([(Float64)(λ)], G, count)

#---- AbstractArray methods ---- (following sparsematrix.jl)

# getindex - implementations 
Base.getindex(p::SimplePairwise, i::Int, j::Int) =	p.λ[1] * p.A[i, j]
Base.getindex(p::SimplePairwise, i::Int) = p.λ[1] * p.A[i]
Base.getindex(p::SimplePairwise, ::Colon, ::Colon) = p.λ[1] * p.A
Base.getindex(p::SimplePairwise, I::AbstractArray) = p.λ[1] * p.A[I]
Base.getindex(p::SimplePairwise, I::AbstractVector, J::AbstractVector) = p.λ[1] * p.A[I,J]

# getindex - translations
Base.getindex(p::SimplePairwise, I::Tuple{Integer, Integer}) = p[I[1], I[2]]
Base.getindex(p::SimplePairwise, I::Tuple{Integer, Integer, Integer}) = p[I[1], I[2]]
Base.getindex(p::SimplePairwise, i::Int, j::Int, r::Int) = p[i,j]
Base.getindex(p::SimplePairwise, ::Colon, ::Colon, ::Colon) = p[:,:]
Base.getindex(p::SimplePairwise, ::Colon, ::Colon, r::Int) = p[:,:]
Base.getindex(p::SimplePairwise, ::Colon, j) = p[1:size(p.A,1), j]
Base.getindex(p::SimplePairwise, i, ::Colon) = p[i, 1:size(p.A,2)]
Base.getindex(p::SimplePairwise, ::Colon, j, r) = p[:,j]
Base.getindex(p::SimplePairwise, i, ::Colon, r) = p[i,:]
Base.getindex(p::SimplePairwise, I::AbstractVector{Bool}, J::AbstractRange{<:Integer}) = p[findall(I),J]
Base.getindex(p::SimplePairwise, I::AbstractRange{<:Integer}, J::AbstractVector{Bool}) = p[I,findall(J)]
Base.getindex(p::SimplePairwise, I::Integer, J::AbstractVector{Bool}) = p[I,findall(J)]
Base.getindex(p::SimplePairwise, I::AbstractVector{Bool}, J::Integer) = p[findall(I),J]
Base.getindex(p::SimplePairwise, I::AbstractVector{Bool}, J::AbstractVector{Bool}) = p[findall(I),findall(J)]
Base.getindex(p::SimplePairwise, I::AbstractVector{<:Integer}, J::AbstractVector{Bool}) = p[I,findall(J)]
Base.getindex(p::SimplePairwise, I::AbstractVector{Bool}, J::AbstractVector{<:Integer}) = p[findall(I),J]

# setindex!
Base.setindex!(p::SimplePairwise, i::Int, j::Int) =
	error("Pairwise values cannot be set directly. Use setparameters! instead.")
Base.setindex!(p::SimplePairwise, i::Int, j::Int, k::Int) = 
	error("Pairwise values cannot be set directly. Use setparameters! instead.")
Base.setindex!(p::SimplePairwise, i::Int) =
	error("Pairwise values cannot be set directly. Use setparameters! instead.")


#---- AbstractPairwiseParameter interface methods ----
getparameters(p::SimplePairwise) = p.λ
function setparameters!(p::SimplePairwise, newpar::Vector{Float64})
    p.λ = newpar
end
