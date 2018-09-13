# common utilities and types

# Define a type to function as an enumeration "States", to hold the string names
# of the two states in the dichotomous variable.  By default this will be "low"
# and "high".
# - Question: should I arrange it so that the two strings can be user-specified
#   at construction time?  This would make things look nice for the user, but
#   would make comparison of Dichotomous variables (e.g. for equivalence of
#   configuration) harder.
# - Question: how to ensure that the value is one of the coding values?
# - Question: be sure about making it a "mutable struct" or just a "struct".
#   E.g. if fields are arrays (mutable types) then their values can still be
#   changed inside a regular immutable struct.

# Define a type "Dichotomous" that holds a state, and a coding.
# Inner constructor with "new" used to ensure value matches coding. BUT, since
#   it's mutable, can still change value later on.
mutable struct Dichotomous
    labels::Tuple{String,String}  #two strings giving the labels, e.g. "low" and "high"
    coding::Tuple{Real,Real}  #two numbers giving the numerical coding of the states
    value::Real  #numeric value of this particular Dichotomous
    Dichotomous(labels, coding, value) = value in coding ?
        new(labels, coding, value) : error("value doesn't match coding")
end

#Non-default "outer" constructor
Dichotomous() = Dichotomous(("low","high"), (-1,1), -1)


#NB for future: can use incomplete initialization if fields can't be assigned at
#   construction time.
