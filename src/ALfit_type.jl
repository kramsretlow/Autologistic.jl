"""
	ALfit

A type to hold estimation output for autologistic models.  Fitting functions return an 
object of this type.

Depending on the fitting method, some fields might not be set.  Fields that are not used
are set to `nothing` or to zero-dimensional arrays.  The fields are:

* `estimate`: A vector of parameter estimates
* `se`: A vector of standard errors for the estimates
* `pvalues`: A vector of p-values for testing the null hypothesis that the parameters equal
  zero (one-at-a time hypothesis tests).
* `CIs`: A vector of 95% confidence intervals for the parameters (a vector of 2-tuples).
* `optim`: the output of the call to `optimize` used to get the estimates.
* `Hinv` (used by `fit_ml!`): The inverse of the Hessian matrix of the objective function, 
  evaluated at the estimate.
* `nboot` - (`fit_pl!`) number of bootstrap samples to use for error estimation
* `kwargs` - (`fit_pl!`) string representation of extra keyword arguments passed in the call
  (a record of which arguments were passed to `sample`)
* `bootsamples` - (`fit_pl!`) the bootstrap samples
* `bootestimates` - (`fit_pl!`) the bootstrap parameter estimates
* `convergence` - either a Boolean indicating optimization convergence ( for `fit_ml!`), or
  a vector of such values for the optimizations done to estimate bootstrap replicates.

The empty constructor `ALfit()` will initialize an object with all fields empty, so the
needed fields can be filled afterwards.
"""
mutable struct ALfit
    estimate::Vector{Float64}
    se::Vector{Float64}
    pvalues::Vector{Float64}
    CIs::Vector{Tuple{Float64,Float64}}
    optim
    Hinv::Array{Float64,2}
    nboot::Int
    kwargs::String
    bootsamples
    bootestimates
    convergence
end

# Constructor with no arguments - for object creation. Initialize everything to empty or 
# nothing.
ALfit() = ALfit(zeros(Float64,0),
                zeros(Float64,0),
                zeros(Float64,0),
                Vector{Tuple{Float64,Float64}}(undef,0),
                nothing,
                zeros(Float64,0,0),
                0,
                "",
                nothing,
                nothing,
                nothing)


# === show methods =============================================================
Base.show(io::IO, f::ALfit) = print(io, "ALfit")

function Base.show(io::IO, ::MIME"text/plain", f::ALfit)
    print(io, "Autologistic model fitting results. Its non-empty fields are:\n", 
          showfields(f,2), "Use summary(fit; [parnames, sigdigits]) to see a table of estimates.\n",
          "Use do_boot(model, fit, ...) to add bootstrap variance estimates after estimation")
end
                
function showfields(f::ALfit, leadspaces=0)
    spc = repeat(" ", leadspaces)
    out = ""
    if length(f.estimate) > 0
        out *= spc * "estimate       " * 
               "$(size2string(f.estimate)) vector of parameter estimates\n"
    end
    if length(f.se) > 0
        out *= spc * "se             " * 
               "$(size2string(f.se)) vector of standard errors\n"
    end
    if length(f.pvalues) > 0
        out *= spc * "pvalues        " * 
               "$(size2string(f.pvalues)) vector of 2-sided p-values\n"
    end
    if length(f.CIs) > 0
        out *= spc * "CIs            " * 
               "$(size2string(f.CIs)) vector of 95% confidence intervals (as tuples)\n"
    end
    if f.optim !== nothing
        out *= spc * "optim          " * 
               "the output of the call to optimize()\n"
    end
    if length(f.Hinv) > 0
        out *= spc * "Hinv           " * 
               "the inverse of the Hessian, evaluated at the optimum\n"
    end
    if f.nboot > 0
        out *= spc * "nboot          " * 
               "the number of bootstrap replicates drawn\n"
    end
    if f.kwargs !== ""
        out *= spc * "kwargs         " * 
               "extra keyword arguments passed to sample()\n"
    end
    if f.bootsamples !== nothing
        out *= spc * "bootsamples    " * 
               "$(size2string(f.bootsamples)) array of bootstrap replicates\n"
    end
    if f.bootestimates !== nothing
        out *= spc * "bootestimates  " * 
               "$(size2string(f.bootestimates)) array of bootstrap estimates\n"
    end
    if f.convergence !== nothing
        if length(f.convergence) == 1
            out *= spc * "convergence    " * 
                   "$(f.convergence)\n"
        else
            out *= spc * "convergence    " * 
                   "$(size2string(f.convergence)) vector of bootstrap convergence flags\n"
        end
    end
    if out == ""
        out = spc * "(all fields empty)\n"
    end
    return out
end

# Line up all strings in rows 2:end of a column of String matrix S, so that a certain
# character (e.g. decimal point or comma) aligns. Do this by prepending spaces.
# After processing, text will line up but strings still might not be all the same length.
function align!(S, col, char)
    nrow = size(S,1)
    locs = findfirst.(isequal(char), S[2:nrow,col])

    # If no char found - make all strings same length
    if all(locs .== nothing)  
        lengths = length.(S[2:nrow,col])
        maxlength = maximum(lengths)
        for i = 2:nrow
            S[i,col] = repeat(" ", maxlength - lengths[i-1]) * S[i,col]
        end
        return
    end

    # Otherwise, align the characters
    maxloc = any(locs .== nothing) ? maximum(length.(S[2:nrow,col])) : maximum(locs)
    for i = 2:nrow
        if locs[i-1] == nothing 
            continue
        end
        S[i,col] = repeat(" ", maxloc - locs[i-1]) * S[i,col]
    end
end

function Base.summary(io::IO, f::ALfit; parnames=nothing, sigdigits=3)
    npar = length(f.estimate)
    if npar==0
        println(io, "No estimates to tabulate")
        return
    end
    if parnames != nothing && length(parnames) !== npar
        error("parnames vector is not the correct length")
    end
    out = Matrix{String}(undef, npar+1, 5)

    out[1,:] = ["name", "est", "se", "p-value", "95% CI"]
    for i = 2:npar+1
        out[i,1] = parnames==nothing ? "parameter $(i-1)" : parnames[i-1]
    end        
    out[2:npar+1, 2] = string.(round.(f.estimate,sigdigits=sigdigits))
    out[2:npar+1, 3] = string.(round.(f.se,sigdigits=sigdigits))
    out[2:npar+1, 4] = string.(round.(f.pvalues,sigdigits=sigdigits))
    out[2:npar+1, 5] = [string(round.((f.CIs[i][1], f.CIs[i][2]),sigdigits=sigdigits)) for i=1:npar]

    align!(out, 2, '.')
    align!(out, 3, '.')
    align!(out, 4, '.')
    align!(out, 5, ',')

    colwidths = [maximum(length.(out[:,i])) for i=1:5]
    for i = 1:size(out,1)
        for j = 1:size(out,2)
            print(io, out[i,j], repeat(" ", colwidths[j]-length(out[i,j])))
            if j < 5
                print(io, "   ")
            else 
                print(io, "\n")
            end
        end
    end
end

function Base.summary(f::ALfit; parnames=nothing, sigdigits=3)
    Base.summary(stdout, f; parnames=parnames, sigdigits=sigdigits)
end

# TODO:
# - Update pigmentosa example and commit/push.  ***then stop: paper!***
# - tests?
# - Write fit_pl! and do_boot
# - Figure out how to parallelize fit_pl!/do_boot
