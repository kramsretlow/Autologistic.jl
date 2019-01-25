# Is one type Autologistic all we need?
# We'd need many different constructors, or maybe some convenience functions
# for specific models.

# ***TODO*** 
# [x] Review constructors: 
#      - What (if any) inner constructor do we need?
#      - How best to check for dimension consistency, etc.?
# [x] Plan out constructors
# [x] Make various constructor functions
# [x] Make getparameters(), setparameters!() methods for AutologisticModel type
# [x] Decide if a "coords" field should be included.  This would be a 
#     type <: SpatialCoordinates, holding optional spatial coordinates of vertices. 
#     Should it be in AutologisticModel, or maybe make it part of the pairwise part?
#      ==> Go for it as part of AutologisticModel.  Need to adapt constructors and tests.
#          For common case where not supplied, just set all coords to (0,0).
#          (or, let it be possible to leave coords as nothing?)
# [ ] Review whether functions that take M::AutologisticModel as argument should actually
#     take M<:AbstractAutologisticModel instead.  Do we need the abstract type?

mutable struct AutologisticModel{U<:AbstractUnaryParameter, 
                       P<:AbstractPairwiseParameter, 
                       C<:CenteringKinds,
                       R<:Real,
                       S<:SpatialCoordinates} <: AbstractAutologisticModel
    responses::Array{Bool,2}                   
    unary::U
    pairwise::P
    centering::C
    coding::Tuple{R,R}           
    labels::Tuple{String,String}
    coordinates::S
    
    function AutologisticModel(y,u::U,p::P,c::C,cod::Tuple{R,R},lab,coords::S) where {U,P,C,R,S}
        if !(size(y) == size(u) == size(p)[[1,3]])
            error("AutologisticModel: inconsistent sizes of Y, unary, and pairwise")
        end
        if cod[1] >= cod[2]
            error("AutologisticModel: must have coding[1] < coding[2]")
        end
        if lab[1] == lab[2] 
            error("AutologisticModel: labels must be different")
        end
        new{U,P,C,R,S}(y,u,p,c,cod,lab,coords)
    end
end

# === Constructors =============================================================
function AutologisticModel(unary::U, pairwise::P; Y::Union{Nothing,<:VecOrMat}=nothing, 
                 centering::CenteringKinds=none, coding::Tuple{Real,Real}=(-1,1),
                 labels::Tuple{String,String}=("low","high"), 
                 coordinates::SpatialCoordinates=nothing
                ) where U<:AbstractUnaryParameter where P<:AbstractPairwiseParameter
    (n, m) = size(unary)
    if Y==nothing
        Y = Array{Bool,2}(undef, n, m)
    else 
        Y = makebool(Y)
    end
    return AutologisticModel(Y,unary,pairwise,centering,coding,labels,coordinates)
end
function makeALRsimple(graph::SimpleGraph{Int}, X::Float2D3D; 
                   Y::VecOrMat=Array{Bool,2}(undef,nv(graph),size(X,3)), 
                   β::Vector{Float64}=zeros(size(X,2)), 
                   λ::Float64=0.0, centering::CenteringKinds=none, 
                   coding::Tuple{Real,Real}=(-1,1),
                   labels::Tuple{String,String}=("low","high"),
                   coordinates::SpatialCoordinates=nothing)
    u = LinPredUnary(X, β)
    p = SimplePairwise(λ, graph, size(X,3))
    return AutologisticModel(makebool(Y),u,p,centering,coding,labels,coordinates)
end
#TODO: ALRadaptive() (requires an appropriate pairwise type)

# === Methods: getting/setting parameters ======================================
getparameters(M::AutologisticModel) = [getparameters(M.unary); getparameters(M.pairwise)]
getunaryparameters(M::AutologisticModel) = getparameters(M.unary)
getpairwiseparameters(M::AutologisticModel) = getparameters(M.pairwise)
function setparameters!(M::AutologisticModel, newpars::Vector{Float64})
    p, q = (length(getunaryparameters(M)), length(getpairwiseparameters(M)))
    @assert length(newpars) == p + q "newpars has wrong length"
    setparameters!(M.unary, newpars[1:p])
    setparameters!(M.pairwise, newpars[p+1:p+q])
    return newpars
end
function setunaryparameters!(M::AutologisticModel, newpars::Vector{Float64})
    setparameters!(M.unary, newpars)
end
function setpairwiseparameters!(M::AutologisticModel, newpars::Vector{Float64})
    setparameters!(M.pairwise, newpars)
end



