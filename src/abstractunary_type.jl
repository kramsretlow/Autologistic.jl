#AbstractUnary is the α (which could be parametrized, e.g. by β)
abstract type AbstractUnary <: AbstractArray{Real, 1} end
Base.IndexStyle(::Type{<:AbstractUnary}) = IndexLinear()
Base.summary(u::AbstractUnary) = "$(typeof(u)) with $(length(u)) elements " *
                    "and average value $(round(mean(values(u)), digits=3))"
