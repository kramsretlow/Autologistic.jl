#AbstractPairwise is the Λ (which could be parametrized)
# We make it a 3D AbstractArray for maximum flexibility, so that in the
# case of replicates with e.g. adaptive pairwise paramettrization, we can 
# have separate Λ values for each replicate.
# ***TODO***
# [x] Should this type be a subtype of a sparse array?  And a symmetric type? 
#    Or should we allow subtypes of AbstractPairwise to decide if they should be 
#    sparse or not? This is especially important if we make it a 3D array with 
#    replicates.
#      ==> Decided, keep it just a subtype of AbstractArray{Float64,3}, and let
#          concrete types decide how to handle their behavior.  Sparse arrays
#          and symmetric matrices are subtypes  of AbstractArray, so no problems
#          there.  
# [ ] Consider making neighborsums() part of the pairwise interface so they can
#     be efficiently calculated by each concrete type.

"""
    AbstractPairwise

Abstract type representing the pairwise part of an autologistic regression model.

All concrete subtypes should have the following fields:

*   `G::SimpleGraph{Int}` -- The graph for the model.
*   `replicates::Int`  -- The number of replicate observations.
*   `A::SparseMatrixCSC{Float64,Int64}`  -- The adjacency matrix of the graph.

In addition to `getindex()` and `setindex!()`, any concrete subtype `P<:AbstractPairwise` 
should also have the following methods defined:

*   `getparameters(P)`, returning a Vector{Float64}
*   `setparameters!(P, newpar::Vector{Float64})` for setting parameter values.

Note that indexing is performance-critical and should be implemented carefully in 
subtypes.  

The intention is that each subtype should implement a different way of parameterizing
the association matrix. The way parameters are stored and values computed is up to the
subtypes. 

This type inherits from `AbstractArray{Float64, 3}`.  The third index is to allow for 
replicate observations.  P[:,:,r] should return the association matrix of the rth
replicate in an appropriate subtype of AbstractMatrix.  It is not intended that the third 
index will be used for range or vector indexing like P[:,:,1:5] (though this may work 
due to AbstractArray fallbacks). 
"""
abstract type AbstractPairwise <: AbstractArray{Float64, 3} end

Base.IndexStyle(::Type{<:AbstractPairwise}) = IndexCartesian()

Base.summary(p::AbstractPairwise) = "**TODO**"

#---- fallback methods --------------
Base.size(p::AbstractPairwise) = (nv(p.G), nv(p.G), p.replicates)

function Base.getindex(p::AbstractPairwise, I::AbstractVector, J::AbstractVector)
    error("getindex not implemented for $(typeof(p))")
end

