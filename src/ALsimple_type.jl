mutable struct ALsimple{C<:CenteringKinds, 
                        R<:Real, 
                        S<:SpatialCoordinates} <: AbstractAutologisticModel
    responses::Array{Bool,2}                   
    unary::FullUnary
    pairwise::SimplePairwise
    centering::C
    coding::Tuple{R,R}           
    labels::Tuple{String,String}
    coordinates::S

    function ALsimple(y, u, p, c::C, cod::Tuple{R,R}, lab, coords::S) where {C,R,S}
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

function ALsimple(unary::FullUnary, pairwise::SimplePairwise; 
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
    return ALsimple(Y,unary,pairwise,centering,coding,labels,coordinates)
end

function ALsimple(graph::SimpleGraph{Int}, alpha::Float1D2D; 
                  Y::VecOrMat=Array{Bool,2}(undef,nv(graph),size(alpha,2)), 
                  λ::Float64=0.0, 
                  centering::CenteringKinds=none, 
                  coding::Tuple{Real,Real}=(-1,1),
                  labels::Tuple{String,String}=("low","high"),
                  coordinates::SpatialCoordinates=nothing)
    u = FullUnary(alpha)
    p = SimplePairwise(λ, graph, size(alpha,2))
    return ALsimple(makebool(Y),u,p,centering,coding,labels,coordinates)
end

# include the observation count (can get n from the nv(graph))
function ALsimple(graph::SimpleGraph{Int}, count::Int=1; 
                  Y::VecOrMat=Array{Bool,2}(undef,nv(graph),count), 
                  λ::Float64=0.0, 
                  centering::CenteringKinds=none, 
                  coding::Tuple{Real,Real}=(-1,1),
                  labels::Tuple{String,String}=("low","high"),
                  coordinates::SpatialCoordinates=nothing)
    u = FullUnary(nv(graph),count)
    p = SimplePairwise(λ, graph, count)
    return ALsimple(makebool(Y),u,p,centering,coding,labels,coordinates)
end


# === show methods =============================================================

function Base.show(io::IO, ::MIME"text/plain", m::ALsimple)
    print(io, "Autologistic model of type ALsimple with parameter vector [α; λ].\n",
              "Fields:\n",
              showfields(m,2))
end

function showfields(m::ALsimple, leadspaces=0)
    spc = repeat(" ", leadspaces)
    return spc * "responses    $(size2string(m.responses)) Bool array\n" *
           spc * "unary        $(size2string(m.unary)) FullUnary with fields:\n" *
           showfields(m.unary, leadspaces+15) *
           spc * "pairwise     $(size2string(m.pairwise)) SimplePairwise with fields:\n" *
           showfields(m.pairwise, leadspaces+15) *
           spc * "centering    $(m.centering)\n" *
           spc * "coding       $(m.coding)\n" * 
           spc * "labels       $(m.labels)\n" *
           spc * "coordinates  $(typeof(m.coordinates))\n"
end

