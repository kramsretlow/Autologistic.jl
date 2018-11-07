#AbstractPairwise is the Λ (which could be parametrized)
# We make it a 3D AbstractArray for maximum flexibility, so that in the
# case of replicates with e.g. adaptive pairwise paramettrization, we can 
# have separate Λ values for each replicate.
# ***TODO***
# [] Should this type be a subtype of a sparse array?  Or should we allow
#    subtypes of AbstractPairwise to decide if they should be sparse or not?
#    This is especially important if we make it a 3D array with replicates.
abstract type AbstractPairwise <: AbstractArray{Real, 3} end
Base.IndexStyle(::Type{<:AbstractPairwise}) = IndexCartesian()
Base.summary(p::AbstractPairwise) = "**TODO**"
