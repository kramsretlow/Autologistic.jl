# Is one type Autologistic all we need?
# We'd need many different constructors, or maybe some convenience functions
# for specific models.

# ***TODO*** 
# [x] Review constructors: 
#      - What (if any) inner constructor do we need?
#      - How best to check for dimension consistency, etc.?
# [] Plan out constructors
# [] Make various constructor functions
# [] Make getparameters(), setparameters!() methods for ALmodel type

mutable struct ALmodel{U<:AbstractUnary, P<:AbstractPairwise, C<:CenteringKinds} <: AbstractAutologistic
    responses::Array{Bool,2}                   
    unary::U
    pairwise::P
    centering::C
    coding::Tuple{Real,Real}           
    labels::Tuple{String,String}
    
    function ALmodel(y,u::U,p::P,c::C,cod,lab) where {U,P,C}
        if !(size(y)[1] == length(u) == size(p)[1])
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

# Constructors
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
function ALRsimple(graph::SimpleGraph{Int}, X::Matrix{<:Real}; 
                   Y::VecOrMat=Array{Bool,2}(undef,nv(graph),1), 
                   β::Vector{<:Real}=Array{Float64,1}(undef,size(X)[2]), 
                   λ::Float64=0.0, centering::CenteringKinds=none, 
                   coding::Tuple{Real,Real}=(-1,1),
                   labels::Tuple{String,String}=("low","high"))
    u = LinPredUnary(X, β)
    p = SimplePairwise(λ, graph)
    return ALmodel(makebool(Y),u,p,centering,coding,labels)
end
#ALRadaptive()

# TODO somewhere: make a lattice(n,m,k) function somewhere to produce a graph
# with n-by-m k-connected lattice.


# Methods to define
# getparameters, setparameters!
# getunaryparameters, setunaryparameters!
# getpairwiseparameters, setpairwiseparameters!