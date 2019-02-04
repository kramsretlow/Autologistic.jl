module Autologistic

# TODO: import only needed functions?
using LightGraphs, LinearAlgebra, SparseArrays, Statistics, Random
using DataFrames, CSV

# ***TODO*** 
# [x] Make and export concrete Centering types (or let the types be an enumeration)
# [x] Consider how to include spatial/spatiotemporal locations of the vertices
#     in our types, so they can be used, e.g., for plotting.
# [ ] Be aware of rmul!, lmul!, mul!, and accumulate! methods in LinearAlgebra, for
#     doing matrix/vector operations and storing results in-place.  Look for speedups.

# ***Question: how to decide which abstract types to export?***
export
    #----- types -----
    AbstractAutologisticModel,
    AbstractPairwiseParameter,
    AbstractUnaryParameter,
    AutologisticModel,
    ALRsimple,
    ALsimple,
    SpatialCoordinates,
    FullUnary,
    LinPredUnary,
    SimplePairwise,
    #----- enums -----
    CenteringKinds, none, expectation, onehalf,
    SamplingMethods, Gibbs, perfect_reuse_samples, perfect_reuse_seeds, perfect_read_once, perfect_bounding_chain,
    #----- functions -----
    centeringterms,
    conditionalprobabilities,
    fullPMF,
    getparameters,
    getpairwiseparameters,
    getunaryparameters,
    makegrid4,
    makegrid8,
    makebool,
    makecoded,
    marginalprobabilities,
    negpotential,
    pseudolikelihood,
    sample,
    setparameters!,
    setunaryparameters!,
    setpairwiseparameters!,
    makespatialgraph

include("common.jl")
include("abstractautologisticmodel_type.jl")
include("abstractunaryparameter_type.jl")
include("abstractpairwiseparameter_type.jl")
include("fullunary_type.jl")
include("linpredunary_type.jl")
include("simplepairwise_type.jl")
include("ALsimple_type.jl")
include("ALRsimple_type.jl")
include("samplers.jl")


end # module
