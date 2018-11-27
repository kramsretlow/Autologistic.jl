module Autologistic

# TODO: import only needed functions?
using Statistics, LightGraphs, LinearAlgebra, SparseArrays, Random

# ***TODO*** 
# [x] Make and export concrete Centering types (or let the types be an enumeration)
# [ ] Consider how to include spatial/spatiotemporal locations of the vertices
#     in our types, so they can be used, e.g., for plotting.

# ***Question: how to decide which abstract types to export?***
export
    #----- types -----
    AbstractAutologistic,
    AbstractPairwise,
    AbstractUnary,
    ALmodel,
    CenteringKinds, none, expectation, onehalf,
    CoordType,
    FullUnary,
    LinPredUnary,
    SamplingMethods, Gibbs, perfect,
    SimplePairwise,
    #----- functions -----
    ALRsimple,
    centering_adjustment,
    fullPMF,
    getparameters,
    getpairwiseparameters,
    getunaryparameters,
    grid4,
    grid8,
    makebool,
    makecoded,
    negpotential,
    pseudolikelihood,
    sample,
    setparameters!,
    setunaryparameters!,
    setpairwiseparameters!,
    spatialgraph

include("abstractautologistic_type.jl")
include("abstractunary_type.jl")
include("abstractpairwise_type.jl")
include("common.jl")
include("almodel_type.jl")
include("almodel_functions.jl")
include("fullunary_type.jl")
include("linpredunary_type.jl")
include("simplepairwise_type.jl")
include("samplers.jl")


end # module
