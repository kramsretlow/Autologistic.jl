module Autologistic

using LightGraphs, LinearAlgebra, SparseArrays, Statistics, Random, CSV, Optim, 
      Distributions, Distributed, SharedArrays

export
    #----- types -----
    AbstractAutologisticModel,
    AbstractPairwiseParameter,
    AbstractUnaryParameter,
    ALfit,
    ALRsimple,
    ALsimple,
    FullPairwise,
    FullUnary,
    SpatialCoordinates,
    LinPredUnary,
    SimplePairwise,
    #----- enums -----
    CenteringKinds, none, expectation, onehalf,
    SamplingMethods, Gibbs, perfect_reuse_samples, perfect_reuse_seeds, perfect_read_once, perfect_bounding_chain,
    #----- functions -----
    addboot!,
    centeringterms,
    conditionalprobabilities,
    fit_ml!,
    fit_pl!,
    fullPMF,
    getparameters,
    getpairwiseparameters,
    getunaryparameters,
    loglikelihood,
    makegrid4,
    makegrid8,
    makebool,
    makecoded,
    makespatialgraph,
    marginalprobabilities,
    negpotential,
    oneboot,
    pseudolikelihood,
    sample,
    setparameters!,
    setunaryparameters!,
    setpairwiseparameters!

include("common.jl")
include("ALfit_type.jl")
include("abstractautologisticmodel_type.jl")
include("abstractunaryparameter_type.jl")
include("abstractpairwiseparameter_type.jl")
include("fullpairwise_type.jl")
include("fullunary_type.jl")
include("linpredunary_type.jl")
include("simplepairwise_type.jl")
include("ALsimple_type.jl")
include("ALRsimple_type.jl")
include("samplers.jl")


end # module
