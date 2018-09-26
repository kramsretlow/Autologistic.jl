# Is one type Autologistic all we need?
# We'd need many different constructors, or maybe some convenience functions
# for specific models.

# ***TODO*** re-read about constructors

struct ALmodel{U<:AbstractUnary, P<:AbstractPairwise, C<:AbstractCentering} <: AbstractAutologistic
    Y::Array{Bool,2}  #The responses
    coding::Coding
    unary::U
    pairwise::P
    centering::C #could this just be a function?
end
