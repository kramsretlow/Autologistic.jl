module Autologistic

using Statistics, LightGraphs

# Types
export Coding

# Functions
export greet

include("abstractautologistic_type.jl")
include("abstractunary_type.jl")
include("abstractpairwise_type.jl")
include("abstractcentering_type.jl")
include("coding_type.jl")
include("almodel_type.jl")






include("common.jl")

greet() = print("Hello World!")

end # module
