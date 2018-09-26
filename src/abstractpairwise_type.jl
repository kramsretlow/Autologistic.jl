#AbstractPairwise is the Λ (which could be parametrized)
abstract type AbstractPairwise <: AbstractArray{Real, 2} end
Base.IndexStyle(::Type{<:AbstractPairwise}) = IndexCartesian()
Base.summary(p::AbstractPairwise) = "**TODO**"
