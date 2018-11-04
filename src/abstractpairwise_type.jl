#AbstractPairwise is the Λ (which could be parametrized)
# We make it a 3D AbstractArray for maximum flexibility, so that in the
# case of replicates with e.g. adaptive pairwise paramettrization, we can 
# have separate Λ values for each replicate.
abstract type AbstractPairwise <: AbstractArray{Real, 3} end
Base.IndexStyle(::Type{<:AbstractPairwise}) = IndexCartesian()
Base.summary(p::AbstractPairwise) = "**TODO**"
