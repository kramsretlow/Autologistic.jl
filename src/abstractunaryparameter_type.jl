#AbstractUnaryParameter is the α (which could be parametrized, e.g. by β)
# We make it a 2D AbstractArray so that we can handle replicates.
abstract type AbstractUnaryParameter <: AbstractArray{Float64, 2} end
Base.IndexStyle(::Type{<:AbstractUnaryParameter}) = IndexCartesian()
function Base.summary(u::AbstractUnaryParameter)
    if size(u)[2] == 1
        return "$(typeof(u)) with $(length(u)) values " *
            "and average value $(round(mean(values(u)), digits=3))"
    else
        return "$(typeof(u)) with $(size(u)[1]) values and " *
            "$(size(u)[2]) replicates."  #TODO: make it better.
    end
end