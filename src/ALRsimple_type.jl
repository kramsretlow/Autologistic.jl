"""
    ALRsimple

An autologistic regression model with "simple smoothing":  the unary parameter is of type
`LinPredUnary`, and the pairwise parameter is of type `SimplePairwise`.
"""
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
"""
    ALRsimple(
        unary::LinPredUnary,
        pairwise::SimplePairwise;
        Y::Union{Nothing,<:VecOrMat}=nothing,
        centering::CenteringKinds=none, 
        coding::Tuple{Real,Real}=(-1,1),
        labels::Tuple{String,String}=("low","high"), 
        coordinates::SpatialCoordinates=[(0.0,0.0) for i=1:size(unary,1)]
    )

Constructs an `ALRsimple` autologistic regression model, with unary part `unary` and
pairwise part `pairwise`.

# Keyword arguments

**`Y`** is the array of dichotomous responses.  Any array with 2 unique values will work.
If the array has only one unique value, it must equal one of th coding values. The 
supplied object will be internally represented as a Boolean array.

**`centering`** controls what form of centering to use.

**`coding`** determines the numeric coding of the dichotomous responses. 

**`labels`** is a 2-tuple of text labels describing the meaning of `Y`. The first element
is the label corresponding to the lower coding value.

**`coordinates`** is an array of 2- or 3-tuples giving spatial coordinates of each vertex in
the graph. Default is to set all coordinates to zero.

# Examples
```jldoctest
julia> u = LinPredUnary(rand(10,3));
julia> p = SimplePairwise(Graph(10,20));
julia> model = ALRsimple(u, p, Y = rand([-2, 3], 10))
Autologistic regression model of type ALRsimple with parameter vector [β; λ].
Fields:
  responses    10×1 Bool array
  unary        10×1 LinPredUnary with fields:
                 X  10×3×1 array (covariates)
                 β  3-element vector (regression coefficients)
  pairwise     10×10×1 SimplePairwise with fields:
                 λ      [0.0] (association parameter)
                 G      the graph (10 vertices, 20 edges)
                 count  1 (the number of observations)
                 A      10×10 SparseMatrixCSC (the adjacency matrix)
  centering    none
  coding       (-1, 1)
  labels       ("low", "high")
  coordinates  10-element vector of Tuple{Float64,Float64}
```
"""
function ALRsimple(unary::LinPredUnary, pairwise::SimplePairwise; 
                   Y::Union{Nothing,<:VecOrMat}=nothing, 
                   centering::CenteringKinds=none, 
                   coding::Tuple{Real,Real}=(-1,1),
                   labels::Tuple{String,String}=("low","high"), 
                   coordinates::SpatialCoordinates=[(0.0,0.0) for i=1:size(unary,1)])
    (n, m) = size(unary)
    if Y==nothing
        Y = Array{Bool,2}(undef, n, m)
    else
        Y = makebool(Y, coding)
    end
    return ALRsimple(Y,unary,pairwise,centering,coding,labels,coordinates)
end

"""
    ALRsimple(
        graph::SimpleGraph{Int}, 
        X::Float2D3D; 
        Y::VecOrMat=Array{Bool,2}(undef,nv(graph),size(X,3)),
        β::Vector{Float64}=zeros(size(X,2)),
        λ::Float64=0.0, 
        centering::CenteringKinds=none, 
        coding::Tuple{Real,Real}=(-1,1),
        labels::Tuple{String,String}=("low","high"),
        coordinates::SpatialCoordinates=[(0.0,0.0) for i=1:nv(graph)]
    )

Constructs an `ALRsimple` autologistic regression model from a `graph` and an `X` array of
predictors. Use `Lightgraphs.jl` functions to make the graph. `X` is n×p×m, where n is the
number of vertices in the graph, p is the number of predictors, and m is the number of 
(multivariate) observations.  If m == 1, `X` can be 2D.

# Keyword arguments

**`Y`** is the array of dichotomous responses.  Any array with 2 unique values will work.
If the array has only one unique value, it must equal one of th coding values. The 
supplied object will be internally represented as a Boolean array.

**`β`** is the regression coefficients.

**`λ`** is the association parameter.

**`centering`** controls what form of centering to use.

**`coding`** determines the numeric coding of the dichotomous responses. 

**`labels`** is a 2-tuple of text labels describing the meaning of `Y`. The first element
is the label corresponding to the lower coding value.

**`coordinates`** is an array of 2- or 3-tuples giving spatial coordinates of each vertex in
the graph. Default is to set all coordinates to zero.

# Examples
```jldoctest
julia> model = ALRsimple(Graph(10,20), rand(10,3), β=[0.5, 0.75, 1.0], λ=1.1)
Autologistic regression model of type ALRsimple with parameter vector [β; λ].
    Fields:
      responses    10×1 Bool array
      unary        10×1 LinPredUnary with fields:
                     X  10×3×1 array (covariates)
                     β  3-element vector (regression coefficients)
      pairwise     10×10×1 SimplePairwise with fields:
                     λ      [1.1] (association parameter)
                     G      the graph (10 vertices, 20 edges)
                     count  1 (the number of observations)
                     A      10×10 SparseMatrixCSC (the adjacency matrix)
      centering    none
      coding       (-1, 1)
      labels       ("low", "high")
      coordinates  10-element vector of Tuple{Float64,Float64}
```
"""
function ALRsimple(graph::SimpleGraph{Int}, X::Float2D3D; 
                   Y::VecOrMat=Array{Bool,2}(undef,nv(graph),size(X,3)), 
                   β::Vector{Float64}=zeros(size(X,2)), 
                   λ::Float64=0.0, centering::CenteringKinds=none, 
                   coding::Tuple{Real,Real}=(-1,1),
                   labels::Tuple{String,String}=("low","high"),
                   coordinates::SpatialCoordinates=[(0.0,0.0) for i=1:nv(graph)])
    u = LinPredUnary(X, β)
    p = SimplePairwise(λ, graph, size(X,3))
    return ALRsimple(makebool(Y,coding),u,p,centering,coding,labels,coordinates)
end


# === show methods =============================================================
function Base.show(io::IO, ::MIME"text/plain", m::ALRsimple)
    print(io, "Autologistic regression model of type ALRsimple with parameter vector [β; λ].\n",
              "Fields:\n",
              showfields(m,2))
end

function showfields(m::ALRsimple, leadspaces=0)
    spc = repeat(" ", leadspaces)
    return spc * "responses    $(size2string(m.responses)) Bool array\n" *
           spc * "unary        $(size2string(m.unary)) LinPredUnary with fields:\n" *
           showfields(m.unary, leadspaces+15) *
           spc * "pairwise     $(size2string(m.pairwise)) SimplePairwise with fields:\n" *
           showfields(m.pairwise, leadspaces+15) *
           spc * "centering    $(m.centering)\n" *
           spc * "coding       $(m.coding)\n" * 
           spc * "labels       $(m.labels)\n" *
           spc * "coordinates  $(size2string(m.coordinates)) vector of $(eltype(m.coordinates))\n"
end



