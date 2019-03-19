"""
    ALsimple

An autologistic model with a `FullUnary` unary parameter type and a `SimplePairwise`
pairwise parameter type.   This model has the maximum number of unary parameters 
(one parameter per variable per observation), and a single association parameter.
"""
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
"""
    ALsimple(
        unary::FullUnary, 
        pairwise::SimplePairwise; 
        Y::Union{Nothing,<:VecOrMat}=nothing, 
        centering::CenteringKinds=none, 
        coding::Tuple{Real,Real}=(-1,1),
        labels::Tuple{String,String}=("low","high"), 
        coordinates::SpatialCoordinates=[(0.0,0.0) for i=1:size(unary,1)]
    )

Constructs an `ALsimple` autologistic model with unary part `unary` and pairwise part
`pairwise`.

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
julia> u = FullUnary(rand(10));
julia> p = SimplePairwise(Graph(10,20));
julia> model = ALsimple(u, p)
Autologistic model of type ALsimple with parameter vector [α; λ].
Fields:
  responses    10×1 Bool array
  unary        10×1 FullUnary with fields:
                 α  10×1 array (unary parameter values)
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
function ALsimple(unary::FullUnary, pairwise::SimplePairwise; 
                  Y::Union{Nothing,<:VecOrMat}=nothing, 
                  centering::CenteringKinds=none, 
                  coding::Tuple{Real,Real}=(-1,1),
                  labels::Tuple{String,String}=("low","high"), 
                  coordinates::SpatialCoordinates=[(0.0,0.0) for i=1:size(unary,1)])
    (n, m) = size(unary)
    if Y==nothing
        Y = Array{Bool,2}(undef, n, m)
    else 
        Y = makebool(Y,coding) 
    end
    return ALsimple(Y,unary,pairwise,centering,coding,labels,coordinates)
end

"""
    ALsimple(
        graph::SimpleGraph{Int}, 
        alpha::Float1D2D; 
        Y::VecOrMat=Array{Bool,2}(undef,nv(graph),size(alpha,2)), 
        λ::Float64=0.0, 
        centering::CenteringKinds=none, 
        coding::Tuple{Real,Real}=(-1,1),
        labels::Tuple{String,String}=("low","high"),
        coordinates::SpatialCoordinates=[(0.0,0.0) for i=1:nv(graph)]
    )

Constructs an `ALsimple` autologistic model from a `graph` and unary parameter values
`alpha`.

# Keyword arguments

**`Y`** is the array of dichotomous responses.  Any array with 2 unique values will work.
If the array has only one unique value, it must equal one of th coding values. The 
supplied object will be internally represented as a Boolean array.

**`λ`** is the association parameter.

**`centering`** controls what form of centering to use.

**`coding`** determines the numeric coding of the dichotomous responses. 

**`labels`** is a 2-tuple of text labels describing the meaning of `Y`. The first element
is the label corresponding to the lower coding value.

**`coordinates`** is an array of 2- or 3-tuples giving spatial coordinates of each vertex in
the graph. Default is to set all coordinates to zero.

# Examples
```jldoctest
julia> model = ALsimple(Graph(10,20), rand(10))
Autologistic model of type ALsimple with parameter vector [α; λ].
Fields:
  responses    10×1 Bool array
  unary        10×1 FullUnary with fields:
                 α  10×1 array (unary parameter values)
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
function ALsimple(graph::SimpleGraph{Int}, alpha::Float1D2D; 
                  Y::VecOrMat=Array{Bool,2}(undef,nv(graph),size(alpha,2)), 
                  λ::Float64=0.0, 
                  centering::CenteringKinds=none, 
                  coding::Tuple{Real,Real}=(-1,1),
                  labels::Tuple{String,String}=("low","high"),
                  coordinates::SpatialCoordinates=[(0.0,0.0) for i=1:nv(graph)])
    u = FullUnary(alpha)
    p = SimplePairwise(λ, graph, size(alpha,2))
    return ALsimple(makebool(Y,coding),u,p,centering,coding,labels,coordinates)
end

"""
    ALsimple(graph::SimpleGraph{Int}, count::Int=1; ...)

Construct an `ALsimple` autologistic model from a `graph`, with `count` undefined
observations.  Keyword arguments are the same as the 
`ALsimple(graph::SimpleGraph{Int}, alpha::Float1D2D; ...)` constructor.

# Examples
```
julia> g = Graph(10,20);
julia> model1 = ALsimple(g);
julia> model2 = ALsimple(g, 4);
julia> size(model1.responses)
(10, 1)

julia> size(model2.responses)
(10, 4)
```
"""
function ALsimple(graph::SimpleGraph{Int}, count::Int=1; 
                  Y::VecOrMat=Array{Bool,2}(undef,nv(graph),count), 
                  λ::Float64=0.0, 
                  centering::CenteringKinds=none, 
                  coding::Tuple{Real,Real}=(-1,1),
                  labels::Tuple{String,String}=("low","high"),
                  coordinates::SpatialCoordinates=[(0.0,0.0) for i=1:nv(graph)])
    u = FullUnary(nv(graph),count)
    p = SimplePairwise(λ, graph, count)
    return ALsimple(makebool(Y,coding),u,p,centering,coding,labels,coordinates)
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
           spc * "coordinates  $(size2string(m.coordinates)) vector of $(eltype(m.coordinates))\n"
end

