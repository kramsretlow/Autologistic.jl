#----- FullUnary ---------------------------------------------------------------
# A unary part with one parameter per variable per replicate.
struct FullUnary <: AbstractUnary
    α::Array{Float64,2}
end

# Constructors
function FullUnary(alpha::Array{Float64,1}) 
    return FullUnary( reshape(alpha, (length(alpha),1)) )
end
FullUnary(n::Int) = FullUnary(Array{Float64,2}(undef,n,1))
FullUnary(n::Int,m::Int) = FullUnary(Array{Float64,2}(undef,n,m))

#---- AbstractArray methods ----

Base.size(u::FullUnary) = size(u.α)

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

#---- AbstractUnary interface ----
getparameters(u::FullUnary) = dropdims(reshape(values(u), length(u), 1), dims=2)
function setparameters!(u::FullUnary, newpars::Vector{Float64})
    # Note, should check dimension match?...
    # Note, not implementing subsetting etc., can only replace the whole vector.
    u[:] = newpars  
end

