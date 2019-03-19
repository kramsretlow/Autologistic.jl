#----- FullUnary ---------------------------------------------------------------
# A unary part with one parameter per variable per observation.
mutable struct FullUnary <: AbstractUnaryParameter
    α::Array{Float64,2}
end

# Constructors
function FullUnary(alpha::Array{Float64,1}) 
    return FullUnary( reshape(alpha, (length(alpha),1)) )
end
FullUnary(n::Int) = FullUnary(zeros(Float64,n,1))
FullUnary(n::Int,m::Int) = FullUnary(zeros(Float64,n,m))

#---- AbstractArray methods ----

Base.size(u::FullUnary) = size(u.α)
Base.length(u::FullUnary) = length(u.α)

# getindex - implementations
Base.getindex(u::FullUnary, I::AbstractArray) = u.α[I]
Base.getindex(u::FullUnary, i::Int, j::Int) = u.α[i,j]
Base.getindex(u::FullUnary, ::Colon, ::Colon) = u.α
Base.getindex(u::FullUnary, I::AbstractVector, J::AbstractVector) = u.α[I,J]

# getindex - translations
Base.getindex(u::FullUnary, I::Tuple{Integer, Integer}) = u[I[1], I[2]]
Base.getindex(u::FullUnary, ::Colon, j::Int) = u[1:size(u.α,1), j]
Base.getindex(u::FullUnary, i::Int, ::Colon) = u[i, 1:size(u.α,2)]
Base.getindex(u::FullUnary, I::AbstractRange{<:Integer}, J::AbstractVector{Bool}) = u[I,findall(J)]
Base.getindex(u::FullUnary, I::AbstractVector{Bool}, J::AbstractRange{<:Integer}) = u[findall(I),J]
Base.getindex(u::FullUnary, I::Integer, J::AbstractVector{Bool}) = u[I,findall(J)]
Base.getindex(u::FullUnary, I::AbstractVector{Bool}, J::Integer) = u[findall(I),J]
Base.getindex(u::FullUnary, I::AbstractVector{Bool}, J::AbstractVector{Bool}) = u[findall(I),findall(J)]
Base.getindex(u::FullUnary, I::AbstractVector{<:Integer}, J::AbstractVector{Bool}) = u[I,findall(J)]
Base.getindex(u::FullUnary, I::AbstractVector{Bool}, J::AbstractVector{<:Integer}) = u[findall(I),J]

# setindex!
Base.setindex!(u::FullUnary, v::Real, I::Vararg{Int,2}) = (u.α[CartesianIndex(I)] = v)

#---- AbstractUnaryParameter interface ----
getparameters(u::FullUnary) = dropdims(reshape(u, length(u), 1), dims=2)
function setparameters!(u::FullUnary, newpars::Vector{Float64})
    # Note, not implementing subsetting etc., can only replace the whole vector.
    if length(newpars) != length(u)
        error("incorrect parameter vector length")
    end
    u.α = reshape(newpars, size(u))
end

#---- to be used in show methods ----
function showfields(u::FullUnary, leadspaces=0)
    spc = repeat(" ", leadspaces)
    return spc * "α  $(size2string(u.α)) array (unary parameter values)\n"
end