# The Coding type is for holding a coding specification.
# The actual values of the dichotomous variables will be input as Booleans to
# minimize ambiguities

struct Coding
    labels::Tuple{String,String}  #two strings giving the labels, e.g. "low" and "high"
    coding::Tuple{Real,Real}  #two numbers giving the numerical coding of the states
end

#Non-default "outer" constructor
Coding() = Coding(("low","high"), (-1,1))
