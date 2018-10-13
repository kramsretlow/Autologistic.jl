const VecOrMat = Union{Vector{N},Matrix{N}} where N<:Number

function makebool(v::V) where V<:VecOrMat
    if ndims(v)==1
        v = v[:,:]    #**convet to 2D, not sure the logic behind [:,:] index
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
