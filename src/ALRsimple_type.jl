mutable struct ALRsimple{C<:CenteringKinds,
                         R<:Real,
                         S<:SpatialCoordinates} <: AbstractAutologisticModel
    responses::Array{Bool,2}                   
    unary::LinPredUnary
    pairwise::SimplePairwise
    centering::C
    coding::Tuple{R,R}           
    labels::Tuple{String,String}
    coordinates::S

    function ALRsimple(y, u, p, c::C, cod::Tuple{R,R}, lab, coords::S) where {C,R,S}
        if !(size(y) == size(u) == size(p)[[1,3]])
            error("ALRsimple: inconsistent sizes of Y, unary, and pairwise")
        end
        if cod[1] >= cod[2]
            error("ALRsimple: must have coding[1] < coding[2]")
        end
        if lab[1] == lab[2] 
            error("ALRsimple: labels must be different")
        end
        new{C,R,S}(y,u,p,c,cod,lab,coords)
    end
end


# === Constructors =============================================================

function ALRsimple(unary::LinPredUnary, pairwise::SimplePairwise; 
                   Y::Union{Nothing,<:VecOrMat}=nothing, 
                   centering::CenteringKinds=none, 
                   coding::Tuple{Real,Real}=(-1,1),
                   labels::Tuple{String,String}=("low","high"), 
                   coordinates::SpatialCoordinates=nothing)
    (n, m) = size(unary)
    if Y==nothing
        Y = Array{Bool,2}(undef, n, m)
    else 
        Y = makebool(Y)
    end
    return ALRsimple(Y,unary,pairwise,centering,coding,labels,coordinates)
end

function ALRsimple(graph::SimpleGraph{Int}, X::Float2D3D; 
                   Y::VecOrMat=Array{Bool,2}(undef,nv(graph),size(X,3)), 
                   β::Vector{Float64}=zeros(size(X,2)), 
                   λ::Float64=0.0, centering::CenteringKinds=none, 
                   coding::Tuple{Real,Real}=(-1,1),
                   labels::Tuple{String,String}=("low","high"),
                   coordinates::SpatialCoordinates=nothing)
    u = LinPredUnary(X, β)
    p = SimplePairwise(λ, graph, size(X,3))
    return ALRsimple(makebool(Y),u,p,centering,coding,labels,coordinates)
end




