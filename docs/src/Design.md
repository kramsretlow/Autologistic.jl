# Design of the Package

In the [Background](@ref) section, it was strongly encouraged to use the symmetric
autologistic model.  Still, the package allows the user to construct AL/ALR models with
different choices of centering and coding, to compare the merits of different choices for
themselves. The package was also built to allow different ****TODO****

In this package, the responses are always stored as arrays of type `Bool`, to separate the
configuration of low/high responses from the choice of coding. If `M` is an AL or ALR model
type, the field `M.coding` holds the numeric coding as a 2-tuple.

TODO - list type design and how it's planned to be used.