# Is one type Autologistic all we need?
# We'd need many different constructors, or maybe some convenience functions
# for specific models.

# ***TODO*** 
# [] Review constructors: 
#      - What (if any) inner constructor do we need?
#      - How best to check for dimension consistency, etc.?
# [] Plan out constructors
# [] Make various constructor functions
# [] Make getparameters(), setparameters!() methods for ALmodel type

struct ALmodel{U<:AbstractUnary, P<:AbstractPairwise, C<:AbstractCentering} <: AbstractAutologistic
    Y::Array{Bool,2}                   #The responses
    unary::U
    pairwise::P
    centering::C                       #could this just be a function?
    coding::Tuple{Real,Real}           #***Q: make this a type parameter?***
    labels::Tuple{String,String}
end

# Constructors
