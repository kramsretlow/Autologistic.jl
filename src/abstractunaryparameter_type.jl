#AbstractUnaryParameter is the α (which could be parametrized, e.g. by β)
# We make it a 2D AbstractArray so that we can handle observations.
abstract type AbstractUnaryParameter <: AbstractArray{Float64, 2} end
Base.IndexStyle(::Type{<:AbstractUnaryParameter}) = IndexCartesian()

function Base.show(io::IO, u::AbstractUnaryParameter)
    r, c = size(u)
    str = "$(r)×$(c) $(typeof(u))"
    print(io, str)
end

function Base.show(io::IO, ::MIME"text/plain", u::AbstractUnaryParameter)
    r, c = size(u)
    if c==1
        str = "with $(r) vertices and average value $(round(mean(u), digits=3)).\n"
    else
        str = "with $(r) vertices and $(c) observations.\n"  
    end
    print(io, "Autologistic unary parameter of type $(typeof(u)),\n",
              str, 
              "Fields:\n",
              showfields(u,2),
              "Index into the variable (e.g. myunary[:,:]) to see values.")
end

function showfields(u::AbstractUnaryParameter, leadspaces=0)
    return "(**fields display not implemented**)\n"
end