# common utilities and types

# Define a type to function as an enumeration "States", to hold the string names
# of the two states in the dichotomous variable.  By default this will be "low"
# and "high".
# - Question: should I arrange it so that the two strings can be user-specified
#   at construction time?  This would make things look nice for the user, but
#   would make comparison of Dichotomous variables (e.g. for equivalence of
#   configuration) harder.
struct Dichotomous
    labels  #two strings giving the labels, e.g. "low" and "high"
    coding  #two numbers giving the numerical coding of the states
    value  #numeric value of this particular Dichotomous
end

# Define a type "Dichotomous" that holds a state, and a coding.
