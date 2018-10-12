module Autologistic

using Statistics, LightGraphs

# ***TODO*** 
# [] Make and export concrete Centering types (or let the types be an enumeration)

# ***Question: how to decide which abstract types to export?***
export
    #----- types -----
    AbstractAutologistic,
    AbstractUnary,
    AbstractPairwise,
    ALmodel,
    Centering, none, expectation, onehalf,
    FullUnary,
    LinPredUnary,
    SimplePairwise,
    #----- functions -----
    getparameters,
    setparameters!

include("abstractautologistic_type.jl")
include("abstractunary_type.jl")
include("abstractpairwise_type.jl")
include("centering.jl")
include("almodel_type.jl")
include("fullunary_type.jl")
include("linpredunary_type.jl")
include("simplepairwise_type.jl")

include("common.jl")

end # module
