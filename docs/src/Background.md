# Background

Here we will provide a brief overview of the autologistic model, to establish some
conventions and terminology that will help you to moake appropriate use of
`Autologistic.jl`.

While we refer to *binary* data, more generally and accurately we should call it
*dichotomous* data: categorical observations with two possible values (low or high, alive
or dead, present or absent, etc.).  It is commonplace to encode such data as 0 or 1, but
other coding choices could be made, and in `Autologistic.jl` the default coding is -1 and
+1.  The coding choice is not trivial: two ALR models with different numeric coding will
not, in general, be equivalent.  Furthermore, the ``(-1, 1)`` coding has distinct advantages

## The Autologistic (AL) Model

TODO

## The Autologistic Regression (ALR) Model

TODO

## Design of the Package

TODO - list type design and how it's planned to be used.