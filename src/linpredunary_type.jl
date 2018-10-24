#----- LinPredUnary ------------------------------------------------------------
# The unary part containing the regression linear predictor.
struct LinPredUnary <: AbstractUnary
    X::Matrix{Float64}
    β::Vector{Float64}

    function LinPredUnary(x, beta) 
        if size(x)[2] != length(beta)
            error("LinPredUnary: X and β dimensions are inconsistent")
        end
        new(x,beta)
    end
end

# Constructors
function LinPredUnary(X::Matrix{Float64})
    (n,p) = size(X)
    return LinPredUnary(X, Vector{Float64}(undef,p))
end
function LinPredUnary(n::Int,p::Int)
    X = Matrix{Float64}(undef,n,p)
    return LinPredUnary(X, Vector{Float64}(undef,p))
end

# Methods required for AbstractArray interface
Base.size(u::LinPredUnary) = (size(u.X, 1),)
Base.getindex(u::LinPredUnary, i::Int) = u.X[i,:]' * u.β
Base.values(u::LinPredUnary) = u.X*u.β  #is this more efficient than fallback?
Base.setindex!(u::LinPredUnary, v::Real, i::Int) =
    error("Values of $(typeof(u)) must be set using setparameters!().")

# Methods required for AbstractUnary interface
getparameters(u::LinPredUnary) = u.β
function setparameters!(u::LinPredUnary, newpars::Vector{Float64})
    u.β[:] = newpars
end
