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
        str = "\n$(size2string(u)) array with average value $(round(mean(u), digits=3)).\n"
    else
        str = " $(size2string(u)) array.\n"  
    end
    print(io, "Autologistic unary parameter α of type $(typeof(u)),",
              str, 
              "Fields:\n",
              showfields(u,2),
              "Use indexing (e.g. myunary[:,:]) to see α values.")
end

function showfields(u::AbstractUnaryParameter, leadspaces=0)
    return repeat(" ", leadspaces) * 
           "(**Autologistic.showfields not implemented for $(typeof(u))**)\n"
end
