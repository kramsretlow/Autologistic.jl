#AbstractUnary is the α (which could be parametrized, e.g. by β)
# We make it a 2D AbstractArray so that we can handle replicates.
abstract type AbstractUnary <: AbstractArray{Real, 2} end
Base.IndexStyle(::Type{<:AbstractUnary}) = IndexCartesian()
function Base.summary(u::AbstractUnary)
    if size(u)[2] == 1
        return "$(typeof(u)) with $(length(u)) values " *
            "and average value $(round(mean(values(u)), digits=3))"
    else
        return "$(typeof(u)) with $(size(u)[1]) values and " *
            "$(size(u)[2]) replicates."  #TODO: make it better.
    end
end