# Is one type Autologistic all we need?
# We'd need many different constructors, or maybe some convenience functions
# for specific models.

# ***TODO*** 
# [x] Review constructors: 
#      - What (if any) inner constructor do we need?
#      - How best to check for dimension consistency, etc.?
# [] Plan out constructors
# [] Make various constructor functions
# [] Make getparameters(), setparameters!() methods for ALmodel type

struct ALmodel{U<:AbstractUnary, P<:AbstractPairwise, C<:Centering} <: AbstractAutologistic
    Y::Array{Bool,2}                   #The responses
    unary::U
    pairwise::P
    centering::C
    coding::Tuple{Real,Real}           #***Q: make this a type parameter?***
    labels::Tuple{String,String}
    
    function ALmodel(y,u::U,p::P,c::C,cod,lab) where {U,P,C}
        if !(size(y)[1] == length(u) == size(p)[1])
            error("Inconsistent sizes of Y, unary, and pairwise")
        end
        if cod[1] >= cod[2]
            error("must have coding[1] < coding[2]")
        end
        if lab[1] == lab[2] 
            error("labels must be different")
        end
        new{U,P,C}(y,u,p,c,cod,lab)
    end
end

# Constructors
 