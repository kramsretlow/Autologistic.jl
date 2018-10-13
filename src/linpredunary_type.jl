#----- LinPredUnary ------------------------------------------------------------
# The unary part containing the regression linear predictor.
struct LinPredUnary{M<:Matrix{<:Real}, V<:Vector{<:Real}} <: AbstractUnary
    X::M
    β::V

    function LinPredUnary(x::M,beta::V) where {M, V}
        if size(x)[2] != length(beta)
            error("LinPredUnary: X and β dimensions are inconsistent")
        end
        new{M,V}(x,beta)
    end
end

# Methods required for AbstractArray interface
Base.size(u::LinPredUnary) = (size(u.X, 1),)
Base.getindex(u::LinPredUnary, i::Int) = u.X[i,:]' * u.β
Base.values(u::LinPredUnary) = u.X*u.β  #is this more efficient than fallback?
Base.setindex!(u::LinPredUnary, v::Real, i::Int) =
    error("Values of $(typeof(u)) must be set using setparameters!().")

# Methods required for AbstractUnary interface
getparameters(u::LinPredUnary) = u.β
function setparameters!(u::LinPredUnary, newpars::Vector{<:Real})
    u.β[:] = newpars
end
