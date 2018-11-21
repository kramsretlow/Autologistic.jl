# Type Aliases
const VecOrMat = Union{Array{T,1}, Array{T,2}} where T
const Float1D2D = Union{Array{Float64,1},Array{Float64,2}}
const Float2D3D = Union{Array{Float64,2},Array{Float64,3}}
const CoordType = Union{Nothing, Array{NTuple{2,T},1},Array{NTuple{3,T},1}} where T<:Real

# Enumerations
@enum CenteringKinds none expectation onehalf
@enum SamplingMethods Gibbs perfect

# A function to make a 2D array of Booleans out of a 1- or 2-D input.
function makebool(v::V) where V<:VecOrMat
    if ndims(v)==1
        v = v[:,:]    #**convet to 2D, not sure the logic behind [:,:] index
    end
    if typeof(v) == Array{Bool,2} 
        return v 
    end
    vals = unique(v)
    if length(vals) != 2
        error("Need exactly 2 unique values to make a Bool array")
    end
    lower = minimum(vals)
    higher = maximum(vals)
    (nrow, ncol) = size(v)
    out = Array{Bool}(undef, nrow, ncol)
    for i in 1:nrow
        for j in 1:ncol
            v[i,j]==lower ? out[i,j] = false : out[i,j] = true
        end
    end
    return out
end



# A function to produce a graph with a 4-connected 2D grid structure, having r 
# rows and c columns.  Returns a tuple containing the graph, and an array of 
# vertex spatial coordinates.
# NB: LightGraphs has a function Grid() for this case.
# TODO: write tests
function grid4(r::Int, c::Int, xlim::Tuple{Real,Real}=(0.0,1.0), 
               ylim::Tuple{Real,Real}=(0.0,1.0))

    # Create graph with r*c vertices, no edges
    G = Graph(r*c)

    # loop through vertices. Number vertices columnwise.
    for i in 1:r*c
        if mod(i,r) !== 1       # N neighbor
            add_edge!(G,i,i-1) 
        end
        if i <= (c-1)*r         # E neighbor
            add_edge!(G,i,i+r)
        end 
        if mod(i,r) !== 0       # S neighbor
          add_edge!(G,i,i+1)
        end
        if i > r                # W neighbor
            add_edge!(G,i,i-r)
        end
    end

    rngx = range(xlim[1], stop=xlim[2], length=c)
    rngy = range(ylim[1], stop=ylim[2], length=r)
    locs = [(rngx[i], rngy[j]) for i in 1:c for j in 1:r]

    return (G=G, locs=locs)
end


# A function to produce a graph with an 8-connected 2D grid structure, having r 
# rows and c columns.  Returns a tuple containing the graph, and an array of 
# vertex spatial coordinates.
# TODO: write tests
function grid8(r::Int, c::Int, xlim::Tuple{Real,Real}=(0.0,1.0), 
               ylim::Tuple{Real,Real}=(0.0,1.0))

    # Create the 4-connected graph
    G, locs = grid4(r, c, xlim, ylim)

    # loop through vertices and add the diagonal edges.
    for i in 1:r*c
        if (mod(i,r) !== 1) && (i<=(c-1)*r)    # NE neighbor
            add_edge!(G,i,i+r-1) 
        end
        if (mod(i,r) !== 0) && (i <= (c-1)*r)  # SE neighbor
            add_edge!(G,i,i+r+1)
        end 
        if (mod(i,r) !== 0) && (i > r)         # SW neighbor
          add_edge!(G,i,i-r+1)
        end
        if (mod(i,r) !== 1) && (i > r)         # NW neighbor
            add_edge!(G,i,i-r-1)
        end
    end

    return (G=G, locs=locs)
end


# A function to generate a graph from points with given coordinates, by
# creating edges between all points within a certain Euclidean distance of 
# one another.
# TODO: write tests
function spatialgraph(coords::C, δ::Real) where C<:CoordType
    @assert coords !== nothing 
    n = length(coords)
    G = Graph(n)
    for i in 1:n
        for j in i+1:n 
            if norm(coords[i] .- coords[j]) <= δ
                add_edge!(G,i,j)
            end
        end
    end
    return (G=G, locs=coords)
end
