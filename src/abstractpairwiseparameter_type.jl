#AbstractPairwiseParameter is the Λ (which could be parametrized)
# We make it a 3D AbstractArray for maximum flexibility, so that in the
# case of e.g. adaptive pairwise paramettrization with multiple observations, 
# we can have separate Λ values for each observation.
# ***TODO***
# [x] Should this type be a subtype of a sparse array?  And a symmetric type? 
#    Or should we allow subtypes of AbstractPairwiseParameter to decide if they should be 
#    sparse or not? This is especially important if we make it a 3D array with 
#    observations.
#      ==> Decided, keep it just a subtype of AbstractArray{Float64,3}, and let
#          concrete types decide how to handle their behavior.  Sparse arrays
#          and symmetric matrices are subtypes  of AbstractArray, so no problems
#          there.  
# [ ] Consider making neighborsums() part of the pairwise interface so they can
#     be efficiently calculated by each concrete type.

"""
    AbstractPairwiseParameter

Abstract type representing the pairwise part of an autologistic regression model.

All concrete subtypes should have the following fields:

*   `G::SimpleGraph{Int}` -- The graph for the model.
*   `count::Int`  -- The number of observations.
*   `A::SparseMatrixCSC{Float64,Int64}`  -- The adjacency matrix of the graph.

In addition to `getindex()` and `setindex!()`, any concrete subtype 
`P<:AbstractPairwiseParameter` should also have the following methods defined:

*   `getparameters(P)`, returning a Vector{Float64}
*   `setparameters!(P, newpar::Vector{Float64})` for setting parameter values.

Note that indexing is performance-critical and should be implemented carefully in 
subtypes.  

The intention is that each subtype should implement a different way of parameterizing
the association matrix. The way parameters are stored and values computed is up to the
subtypes. 

This type inherits from `AbstractArray{Float64, 3}`.  The third index is to allow for 
multiple observations. `P[:,:,r]` should return the association matrix of the rth
observation in an appropriate subtype of AbstractMatrix.  It is not intended that the third 
index will be used for range or vector indexing like `P[:,:,1:5]` (though this may work 
due to AbstractArray fallbacks). 

# Examples
```jldoctest
julia> M = ALsimple(Graph(4,4));
julia> typeof(M.pairwise)
SimplePairwise
julia> isa(M.pairwise, AbstractPairwiseParameter)
true
```
"""
abstract type AbstractPairwiseParameter <: AbstractArray{Float64, 3} end

Base.IndexStyle(::Type{<:AbstractPairwiseParameter}) = IndexCartesian()

#---- fallback methods --------------
Base.size(p::AbstractPairwiseParameter) = (nv(p.G), nv(p.G), p.count)

function Base.getindex(p::AbstractPairwiseParameter, I::AbstractVector, J::AbstractVector)
    error("getindex not implemented for $(typeof(p))")
end

function Base.show(io::IO, p::AbstractPairwiseParameter)
    r, c, m = size(p)
    str = "$(size2string(p)) $(typeof(p))"
    print(io, str)
end

#=
function Base.show(io::IO, ::MIME"text/plain", u::AbstractPairwiseParameter)
    r, c = size(u)
    if c==1
        str = "$(typeof(u)) with $(r) vertices " *
              "and average value $(round(mean(u), digits=3))"
    else
        str = "$(typeof(u)) with $(r) vertices and $(c) observations."  
    end
    print(io, str)
end
=#
