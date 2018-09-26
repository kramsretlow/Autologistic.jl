module Autologistic

using Statistics, LightGraphs

# ***Question: how to decide which abstract types to export?***
export
    #----- types -----
    AbstractAutologistic,
    AbstractUnary,
    AbstractPairwise,
    AbstractCentering,
    ALmodel,
    FullUnary,
    LinPredUnary,
    SimplePairwise,
    #----- functions -----
    greet,
    getparameters,
    setparameters!

include("abstractautologistic_type.jl")
include("abstractunary_type.jl")
include("abstractpairwise_type.jl")
include("abstractcentering_type.jl")
include("coding_type.jl")
include("almodel_type.jl")
include("fullunary_type.jl")
include("linpredunary_type.jl")
include("simplepairwise_type.jl")






include("common.jl")

greet() = print("Hello World!")

end # module
