# Type Aliases
const VecOrMat = Union{Array{T,1}, Array{T,2}} where T
const Float1D2D = Union{Array{Float64,1},Array{Float64,2}}
const Float2D3D = Union{Array{Float64,2},Array{Float64,3}}
const SpatialCoordinates = Union{Array{NTuple{2,T},1},Array{NTuple{3,T},1}} where T<:Real

# Enumerations
"""
    CenteringKinds

An enumeration to facilitate choosing a form of centering for the model.  Available
choices are: 

*   `none`: no centering (centering adjustment equals zero).
*   `expectation`: the centering adjustment is the expected value of the response under the
    assumption of independence (this is what has been used in the "centered autologistic 
    model").
*   `onehalf`: a constant value of centering adjustment equal to 0.5.
"""
@enum CenteringKinds none expectation onehalf

"""
    SamplingMethods

An enumeration to facilitate choosing a method for sampling. Available choices are:

*   `Gibbs`  TODO
*   `perfect_bounding_chain`  TODO
*   `perfect_reuse_samples`  TODO 
*   `perfect_reuse_seeds`  TODO
*   `perfect_read_once`  TODO 
"""
@enum SamplingMethods Gibbs perfect_reuse_samples perfect_reuse_seeds perfect_read_once perfect_bounding_chain

# A function to make a 2D array of Booleans out of a 1- or 2-D input.
# 2nd argument `values` optionally can be a 2-tuple (low, high) specifying the two values
# we want to convert to boolean (useful for the case where all elements of `v` take one 
# value or the other.)
# - If v has more than 3 unique values, throw an error
# - If v has exactly 2 unique values, use those to set the coding (ignore vals)
# - If v has 1 unique value, use vals to determine if it's the high or low (throw an error
#   if v's value isn't in vals)
function makebool(v::VecOrMat, vals=nothing)
    if ndims(v)==1
        v = v[:,:]    #**convet to 2D, not sure the logic behind [:,:] index
    end
    if typeof(v) == Array{Bool,2} 
        return v 
    end
    (nrow, ncol) = size(v)
    out = Array{Bool}(undef, nrow, ncol)
    nv = length(unique(v))
    if nv > 2
        error("The input has more than two values.")
    elseif nv == 2
        lower = minimum(v)
    elseif typeof(vals) <: NTuple{2} && v[1] in vals
        lower = vals[1]
    else
        error("One unique value. Could not assign true or false.")
    end
    for i in 1:nrow
        for j in 1:ncol
            v[i,j]==lower ? out[i,j] = false : out[i,j] = true
        end
    end
    return out
end


# A fcn to convert Boolean responses into coded values.  1st argument is boolean
# Returns a 2D array of Float64.  If Y is not supplied, use the responses stored
# in the 1st argument.
function makecoded(b::VecOrMat, coding::Tuple{Real,Real})
    lo = Float64(coding[1])
    hi = Float64(coding[2])
    if ndims(b)==1
        b = b[:,:]
    end
    n, m = size(b)
    out = Array{Float64,2}(undef, n, m)
    for j = 1:m
        for i = 1:n
            out[i,j] = b[i,j] ? hi : lo
        end
    end
    return out
end



# A function to produce a graph with a 4-connected 2D grid structure, having r 
# rows and c columns.  Returns a tuple containing the graph, and an array of 
# vertex spatial coordinates.
# NB: LightGraphs has a function Grid() for this case.
# TODO: write tests
function makegrid4(r::Int, c::Int, xlim::Tuple{Real,Real}=(0.0,1.0), 
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
function makegrid8(r::Int, c::Int, xlim::Tuple{Real,Real}=(0.0,1.0), 
               ylim::Tuple{Real,Real}=(0.0,1.0))

    # Create the 4-connected graph
    G, locs = makegrid4(r, c, xlim, ylim)

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
function makespatialgraph(coords::C, δ::Real) where C<:SpatialCoordinates
    #Replace coords by an equivalent tuple of Float64, for consistency
    n = length(coords)
    locs = [Float64.(coords[i]) for i = 1:n]
    #Make the graph and add edges
    G = Graph(n)
    for i in 1:n
        for j in i+1:n 
            if norm(locs[i] .- locs[j]) <= δ
                add_edge!(G,i,j)
            end
        end
    end
    return (G=G, locs=locs)
end


# Helper function to compute an inner product between two vectors.
# Doing this way seems to reduce memory allocations vs. a' * b way.
# Assume dimensions are correct.
# TODO: later, try @inline, @inbounds, @boundscheck, etc. for speed.
# TODO: later, verify if this function is needed!!!
function innerprod(a, b)::Float64
    out = 0.0
    for i in eachindex(a)
        out += a[i] * b[i]
    end
    return out
end


# Open data sets 
function datasets(name::String)
    if name=="pigmentosa"
        dfpath = joinpath(dirname(pathof(Autologistic)), "..", "assets", "pigmentosa.csv")
        return CSV.read(dfpath)
    elseif name=="hydrocotyle"
        dfpath = joinpath(dirname(pathof(Autologistic)), "..", "assets", "hydrocotyle.csv")
        return CSV.read(dfpath)
    else
        error("Name is not one of the available options.")
    end
end


# Make size into strings like 10×5×2 (for use in show methods)
function size2string(x::T) where T<:AbstractArray
    d = size(x)
    n = length(d)
    if n ==1 
        return "$(d[1])-element"
    else
        str = "$(d[1])" 
        for i = 2:n
            str *= "×"
            str *= "$(d[i])"
        end
        return str
    end
end


# Approximate the Hessian of fcn at the point x, using a step width h.
# Uses the O(h^2) central difference approximation.
# Intended for obtaining standard errors from ML fitting.
# TODO: tests
function hess(fcn, x, h=1e-6)  
    n = length(x)
    H = zeros(n,n)
    hI = h*Matrix(1.0I,n,n)  #ith column of hI has h in ith position, 0 elsewhere.
    
    # Fill up the top half of the matrix
    for i = 1:n        
        for j = i:n
            h1 = hI[:,i]
            h2 = hI[:,j];
            H[i,j] = (fcn(x+h1+h2)-fcn(x+h1-h2)-fcn(x-h1+h2)+fcn(x-h1-h2)) / (4*h^2)
        end
    end
    
    # Fill the bottom half of H (use symmetry), and return
    return H + LinearAlgebra.triu(H,1)'
end

# Takes a named tuple (arising from keyword argument list) and produces two named tuples:
# one with the arguments for optimise(), and one for arguments to sample()
# Usage: optimargs, sampleargs = splitkw(keyword_tuple)
# (tests done)
splitkw = function(kwargs)
    optimnames = fieldnames(typeof(Optim.Options()))
    samplenames = (:method, :indices, :average, :config, :burnin, :verbose)
    optimargs = Dict{Symbol,Any}()
    sampleargs = Dict{Symbol,Any}()
    for (symb, val) in pairs(kwargs)
        if symb in optimnames
            push!(optimargs, symb => val)
        end
        if symb in samplenames
            push!(sampleargs, symb => val)
        end
    end
    return (;optimargs...), (;sampleargs...)
end
