# Is one type Autologistic all we need?
# We'd need many different constructors, or maybe some convenience functions
# for specific models.

# ***TODO*** 
# [x] Review constructors: 
#      - What (if any) inner constructor do we need?
#      - How best to check for dimension consistency, etc.?
# [x] Plan out constructors
# [x] Make various constructor functions
# [x] Make getparameters(), setparameters!() methods for ALmodel type
# [ ] Decide if a "coords" field should be included.  This would be a 
#     type <: CoordType, holding optional spatial coordinates of vertices. 
#     Should it be in ALmodel, or maybe make it part of the pairwise part?
#      ==> Go for it as part of ALmodel.  Need to adapt constructors and tests.
#          For common case where not supplied, just set all coords to (0,0).
#          (or, let it be possible to leave coords as nothing?)

mutable struct ALmodel{U<:AbstractUnary, P<:AbstractPairwise, C<:CenteringKinds} <: AbstractAutologistic
    responses::Array{Bool,2}                   
    unary::U
    pairwise::P
    centering::C
    coding::Tuple{Real,Real}           
    labels::Tuple{String,String}
    
    function ALmodel(y,u::U,p::P,c::C,cod,lab) where {U,P,C}
        if !(size(y) == size(u) == size(p)[[1,3]])
            error("ALmodel: inconsistent sizes of Y, unary, and pairwise")
        end
        if cod[1] >= cod[2]
            error("ALmodel: must have coding[1] < coding[2]")
        end
        if lab[1] == lab[2] 
            error("ALmodel: labels must be different")
        end
        new{U,P,C}(y,u,p,c,cod,lab)
    end
end

# === Constructors =============================================================
function ALmodel(unary::U, pairwise::P; Y::Union{Nothing,<:VecOrMat}=nothing, 
                 centering::CenteringKinds=none, coding::Tuple{Real,Real}=(-1,1),
                 labels::Tuple{String,String}=("low","high")
                ) where U<:AbstractUnary where P<:AbstractPairwise
    n = length(unary)
    if Y==nothing
        Y = Array{Bool,2}(undef, n, 1)
    else 
        Y = makebool(Y)
    end
    return ALmodel(Y,unary,pairwise,centering,coding,labels)
end
function ALRsimple(graph::SimpleGraph{Int}, X::Float2D3D; 
                   Y::VecOrMat=Array{Bool,2}(undef,nv(graph),1), 
                   β::Vector{Float64}=Array{Float64,1}(undef,size(X)[2]), 
                   λ::Float64=0.0, centering::CenteringKinds=none, 
                   coding::Tuple{Real,Real}=(-1,1),
                   labels::Tuple{String,String}=("low","high"))
    u = LinPredUnary(X, β)
    p = SimplePairwise(λ, graph)
    return ALmodel(makebool(Y),u,p,centering,coding,labels)
end
#TODO: ALRadaptive() (requires an appropriate pairwise type)

# === Methods: getting/setting parameters ======================================
getparameters(M::ALmodel) = [getparameters(M.unary); getparameters(M.pairwise)]
getunaryparameters(M::ALmodel) = getparameters(M.unary)
getpairwiseparameters(M::ALmodel) = getparameters(M.pairwise)
function setparameters!(M::ALmodel, newpars::Vector{Float64})
    p, q = (length(getunaryparameters(M)), length(getpairwiseparameters(M)))
    @assert length(newpars) == p + q "newpars has wrong length"
    setparameters!(M.unary, newpars[1:p])
    setparameters!(M.pairwise, newpars[p+1:p+q])
    return newpars
end
function setunaryparameters!(M::ALmodel, newpars::Vector{Float64})
    setparameters!(M.unary, newpars)
end
function setpairwiseparameters!(M::ALmodel, newpars::Vector{Float64})
    setparameters!(M.pairwise, newpars)
end



