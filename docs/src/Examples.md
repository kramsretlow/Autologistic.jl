# Examples

These examples demonstrate most of the functionality of the package, its typical usage, and
how to make some plots you might want to use.

The examples:

* [An Ising Model](@ref) shows how to use the package to explore the autologistic
  probability distribution, without concern about covariates or parameter estimation.
* [Clustered Binary Data (Small ``n``)](@ref) shows how to use the package for regression
  analysis when the graph is small enough to permit computation of the normalizing constant.
  In this case standard maximum likelihood methods of inference can be used.
* [Spatial Binary Regression](@ref) shows how to use the package for autologistic regression
  analysis for larger, spatially-referenced graphs. In this case pseudolikelihood is used
  for estimation, and a (possibly parallelized) parametric bootstrap is used for inference.

## An Ising Model

The term "Ising model" is usually used to refer to a Markov random field of dichotomous
random variables on a regular lattice.  The graph is such that each variable shares an
edge only with its nearest neighbors in each dimension.  It's
a traditional model for magnetic spins, where the coding ``(-1,1)`` is usually used.
There's one parameter per vertex (a "local magnetic field") that increases or
decreases the chance of getting a ``+1`` state at that vertex; and there's a single pairwise
parameter that controls the strength of interaction between neighbor states.

In our terminology it's just an autologistic model with the appropriate graph.
Specifically, it's an `ARsimple` model: one with `FullUnary` type unary parameter, and
`SimplePairwise` type pairwise parameter.

We can create such a model once we have the graph.  For example, let's create two 30-by-30
lattices: one without any special handling of the boundary, and one with periodic boundary
conditions. This can be done with
[LightGraphs.jl](https://github.com/JuliaGraphs/LightGraphs.jl)'s `Grid` function.

```@example Ising
n = 30  # numer of rows and columns
using LightGraphs
G1 = Grid([n, n], periodic=false)
G2 = Grid([n, n], periodic=true)
nothing # hide
```

Now create an AL model for each case. Initialize the unary parameters to Gaussian white
noise. By default the pairwise parameter is set to zero, which implies independence of the
variables. Use the same parameters for the two models, so the only difference betwen them
is the graph.

```@example Ising
using Random, Autologistic
Random.seed!(8888)
α = randn(n^2)
M1 = ALsimple(G1, α)
M2 = ALsimple(G2, α)
```

The REPL output shows information about the model.  It's an `ALsimple` type with one
observation of length 900.

The `conditionalprobabilities` function returns the probablity of observing a ``+1`` state
at each vertex, conditional on the vertex's neighbor values. These can be visualized
as an image, using a `heatmap` (from [Plots.jl](https://github.com/JuliaPlots/Plots.jl)):

```@example Ising
using Plots
condprobs = conditionalprobabilities(M1)
hm = heatmap(reshape(condprobs, n, n), c=:grays, aspect_ratio=1,
             title="probability of +1 under independence")
plot(hm)
```

Since the association parameter is zero, there are no neighborhood effects.  The above
conditional probabilities are equal to the marginal probabilities.

Next, set the association parameters to 0.75, a fairly strong association level, to
introduce a neighbor effect.

```@example Ising
setpairwiseparameters!(M1, [0.75])
setpairwiseparameters!(M2, [0.75])
nothing # hide
```

A quick way to see the effect of this parameter is to observe random samples from the
models. The `sample` function can be used to do this. For this example, use perfect
sampling using a bounding chain algorithm (the enumeration
[`SamplingMethods`](@ref) lists the available sampling options).

```@example Ising
s1 = sample(M1, method=perfect_bounding_chain)
s2 = sample(M2, method=perfect_bounding_chain)
nothing #hide
```

The samples can also be visualized using `heatmap`:

```@example Ising
pl1 = heatmap(reshape(s1, n, n), c=:grays, colorbar=false, title="regular boundary");
pl2 = heatmap(reshape(s2, n, n), c=:grays, colorbar=false, title="periodic boundary");
plot(pl1, pl2, size=(800,400), aspect_ratio=1)
```

In these plots, black indicates the low state, and white the high state.  A lot of local
clustering is occurring in the samples due to the neighbor effects.

To see the long-run differences between the two models, we can look at the marginal
probabilities. They can be estimated by drawing many samples and averaging them
(note that running this code chunk can take a couple of minutes):

```julia
marg1 = sample(M1, 500, method=perfect_bounding_chain, verbose=true, average=true)
marg2 = sample(M2, 500, method=perfect_bounding_chain, verbose=true, average=true)
pl3 = heatmap(reshape(marg1, n, n), c=:grays, colorbar=false, title="regular boundary");
pl4 = heatmap(reshape(marg2, n, n), c=:grays, colorbar=false, title="periodic boundary");
plot(pl3, pl4, size=(800,400), aspect_ratio=1)
savefig("marginal-probs.png")
```

The figure `marginal-probs.png` looks like this:

![marginal-probs.png](../assets/marginal-probs.png)

Although the differences between the two marginal distributions are not striking, the
extra edges connecting top to bottom and left to right do have some influence on the
probabilities at the periphery of the square.

As a final demonstration, perform Gibbs sampling for model `M2`, starting from
a random state.  Display a gif animation of the progress.

```julia
nframes = 150
gibbs_steps = sample(M2, nframes, method=Gibbs)
anim = @animate for i =  1:nframes
    heatmap(reshape(gibbs_steps[:,i], n, n), c=:grays, colorbar=false, 
            aspect_ratio=1, title="Gibbs sampling: step $(i)")
end
gif(anim, "ising_gif.gif", fps=10)
```

![ising_gif.gif](../assets/ising_gif.gif)

## Clustered Binary Data (Small ``n``)

The *retinitis pigmentosa* data set
[obtained here](https://sites.google.com/a/channing.harvard.edu/bernardrosner/channing/regression-method-when-the-eye-is-the-unit-of-analysis)
is an opthalmology data set.  The data comes from 444 patients that had both eyes
examined.  The data can be loaded with `Autologistic.datasets`:

```@repl pigmentosa
using Autologistic, DataFrames, LightGraphs
df = Autologistic.datasets("pigmentosa");
first(df, 6)
describe(df)
```

The response for each eye is **va**, an indicator of poor visual acuity (coded 0 = no,
1 = yes in the loaded data set). Seven covariates were also recorded for each eye:

* **aut_dom**: autosomal dominant (0=no, 1=yes)
* **aut_rec**: autosomal recessive (0=no, 1=yes)
* **sex_link**: sex-linked (0=no, 1=yes)
* **age**: age (years, range 6-80)
* **sex**: gender (0=female, 1=male)
* **psc**: posterior subscapsular cataract (0=no, 1=yes)
* **eye**: which eye is it? (0=left, 1=right)

The last four factors are relevant clinical observations, and the first three are genetic
factors. The data set also includes an **ID** column with an ID number specific to each
patient.  Eyes with the same ID come from the same person.

The natural unit of analysis is the eye, but pairs of observations from the same
patient are "clustered" because the occurrence of acuity loss in the left and right eye
is probably correlated. We can model each person's two eyes' **va** outcomes as two
dichotomous random variables with a 2-vertex, 1-edge graph.

```@example pigmentosa
G = Graph(2,1)
```

Each of the 444 observations has this graph, and each has its own set of covariates.

If we include all seven predictors, plus intercept, in our model, we have 2 variables per
observation, 8 predictors, and 444 obsrevations. 

Before creating the model we need to re-structure the covariates. The data in `df` has one
row per eye, with the variable `ID` indicating which eyes belong to the same patient.  We
need to rearrange the responses (`Y`) and the predictors (`X`) into arrays suitable for our
autologistic models, namely:

* `Y` is ``2 \times 444`` with one observation per column.
* `X` is ``2 \times 8 \times 444`` with one ``2 \times 8`` matrix of predictors for each
  observation.  The first column of each predictor matrix is an intercept column, and  
  columns 2 through 8 are for `aut_dom`, `aut_rec`, `sex_link`, `age`, `sex`, `psc`, and
  `eye`, respectively.

```@example pigmentosa
X = Array{Float64,3}(undef, 2, 8, 444);
Y = Array{Float64,2}(undef, 2, 444);
for i in 1:2:888
    subject = Int((i+1)/2)
    X[1,:,subject] = [1 permutedims(Vector(df[i,2:8]))]
    X[2,:,subject] = [1 permutedims(Vector(df[i+1,2:8]))]
    Y[:,subject] = convert(Array, df[i:i+1, 9])
end
```

For example, patient 100 had responses

```@example pigmentosa
Y[:,100]
```

Indicating visual acuity loss in the left eye, but not in the right. The predictors for
this individual are

```@example pigmentosa
X[:,:,100]
```

Now we can create our autologistic regression model.

```@example pigmentosa
model = ALRsimple(G, X, Y=Y)
```

This creates a model with the "simple pairwise" structure, using a single association
parameter. The default is to use no centering adjustment, and to use coding ``(-1,1)`` for
the responses.  This "symmetric" version of the model is recommended for
[a variety of reasons](https://doi.org/10.3389/fams.2017.00024).  Using different coding
or centering choices is only recommended if you have a thorough understanding of what
you are doing; but if you wish to use different choices, this can easily be done using
keyword arguments. For example, `ALRsimple(G, X, Y=Y, coding=(0,1), centering=expectation)`
creates the "centered autologistic model" that has appeared in the literature, e.g.
[here](https://link.springer.com/article/10.1198/jabes.2009.07032) and
[here](https://doi.org/10.1002/env.1102)

The model has nine parameters (eight regression coefficients plus the association
parameter).  All parameters are initialized to zero:

```@example pigmentosa
getparameters(model)
```

When we call `getparameters`, the vector returned always has the unary parameters first,
with the pairwise parameter(s) appended at the end.

Because there are only two vertices in the graph, we can use the full likelihood
(`fit_ml!` function) to do parameter estimation.  This function returns a structure with
the estimates as well as standard errors, p-values, and 95% confidence intervals for the 
parameter estimates.

```@example pigmentosa
out = fit_ml!(model)
```

To view the estimation results, use `summary`:

```@example pigmentosa
summary(out, parnames = ["icept", "aut_dom", "aut_rec", "sex_link", "age", "sex", 
        "psc", "eye", "λ"])
```

From this we see that the association parameter is fairly large (0.818), supporting the
idea that the left and right eyes are associated.  It is also highly statistically
significant.  Among the covariates, `sex_link`, `age`, and `psc` are all statistically
significant.

## Spatial Binary Regression

Autologistic regression is a natural candidate for analysis of spatial binary data, where
locations in the same neighborhood are more likely to have the same outcome than sites that
are far apart.
The [hydrocotyle data](https://doi.org/10.1016/j.ecolmodel.2007.04.024) provide a typical
example.  The response in this data set is the presence/absence of a certain plant species
in a grid of 2995 regions covering Germany. The data set is included in Autologistic.jl:

```@repl hydro
using Autologistic, DataFrames, LightGraphs
df = Autologistic.datasets("hydrocotyle")
```

In the data frame, the variables `X` and `Y` give the spatial coordinates of each region (in
dimensionless integer units), `obs` gives the presence/absence data (1 = presence), and
`altitude` and `temperature` are covariates.

We will use an `ALRsimple` model for these data.  The graph can be formed using
[`makespatialgraph`](@ref):

```@example hydro
locations = [(df.X[i], df.Y[i]) for i in 1:size(df,1)]
g = makespatialgraph(locations, 1.0)
nothing # hide
```

`makespatialgraph` takes an array of tuples giving the spatial coodinates, and a distance
threshold. It returns a named tuple with the graph and the array of coordinates.  The graph
has one vertex for each spatial location, and edge for every pair of locations that are
within the threshold distance of each other. For these data arranged on a grid, a threshold
of 1.0 will make a 4-nearest-neighbors lattice. Letting the threshold be `sqrt(2)` would
make an 8-nearest-neighbors lattice.

We can visualize the graph, the responses, and the predictors using
[GraphRecipes.jl](https://github.com/JuliaPlots/GraphRecipes.jl) (there are
[several other](http://juliagraphs.github.io/LightGraphs.jl/latest/plotting.html)
options for plotting graphs as well).

```@example hydro
using GraphRecipes, Plots

# Function to convert a value to a gray shade
makegray(x, lo, hi) = RGB([(x-lo)/(hi-lo) for i=1:3]...)  

# Function to plot the graph with node shading determined by v.
# Plot each node as a square and don't show the edges.
function myplot(v, lohi=nothing)  
    if lohi==nothing
        colors = makegray.(v, minimum(v), maximum(v))
    else
        colors = makegray.(v, lohi[1], lohi[2])
    end
    return graphplot(g.G, x=df.X, y=df.Y, background_color = :lightblue,
                marker = :square, markersize=2, markerstrokewidth=0,
                markercolor = colors, yflip = true, linecolor=nothing)
end

# Make the plot
plot(myplot(df.obs), myplot(df.altitude), myplot(df.temperature),
     layout=(1,3), size=(800,300), titlefontsize=8,
     title=hcat("Species Presence (white = yes)", "Altitude (lighter = higher)",
                "Temperature (lighter = higher)"))
```

### Constructing the model

We can see that the species primarily is found at low-altidude locations. To model the
effect of altitude and temperature on species presence, construct an `ALRsimple` model.

```@example hydro
# Autologistic.jl requres predictors to be a matrix of Float64
Xmatrix = Array{Float64}([ones(2995) df.altitude df.temperature])

# Create the model
hydro = ALRsimple(g.G, Xmatrix, Y=df.obs)

```

The model `hydro` has four parameters: three regression coefficients (interceept, altitude,
and temperature) plus an association parameter.  It is a "symmetric" autologistic model,
because it has a coding symmetric around zero and no centering term.

### Fitting the model by pseudolikelihood

With 2995 nodes in the graph, the likelihood is intractable for this case.  Use `fit_pl!` to
do parameter estimation by pseudolikelihood instead.  The fitting function uses the BFGS
algorithm via [`Optim.jl`](http://julianlsolvers.github.io/Optim.jl/stable/).  Any of
Optim's [general options](http://julianlsolvers.github.io/Optim.jl/stable/#user/config/)
can be passed to `fit_pl!` to control the optimization.  We have found that
`allow_f_increases` often aids convergence; it is used here:

```@repl hydro
fit1 = fit_pl!(hydro, allow_f_increases=true)
parnames = ["intercept", "altitude", "temperature", "association"];
summary(fit1, parnames=parnames)
```

`fit_pl!` sets the parameters of the model object to the optimal values, and also returns
an object of type `ALfit`, with fields holding useful information.  Calling `summary(fit1)`
produces a summary table of the estimates.  For now there are no standard errors.  This
will be addressed below.

To quickly visualize the quality of the fitted model, we can use sampling to get the
marginal probabilities:

```@example hydro
# Average 500 samples to estimate marginal probability of species presence
marginal1 = sample(hydro, 500, method=perfect_bounding_chain, average=true)

# Draw 2 random samples for visualizing generated data.
draws = sample(hydro, 2, method=perfect_bounding_chain)

# Plot them
plot(myplot(marginal1, (0,1)), myplot(draws[:,1]), myplot(draws[:,2]),
     layout=(1,3), size=(800,300), titlefontsize=8,
     title=["Marginal Probability" "Random sample 1" "Random Sample 2"])
```

In the above code, perfect sampling was used to draw samples from the fitted distribution.
The marginal plot shows consistency with the observed data, and the two generated data
sets show a level of spatial clustering similar to the observed data.

### Error estimation 1: bootstrap after the fact

A parametric bootstrap can be used to get an estimate of the precision of the estimates
returned by `fit_pl!`.  The function [`oneboot`](@ref) has been included in the package to
facilitate this.  Each call of `oneboot` draws a random sample from the fitted distribution,
then re-fits the model using this sample as the responses. It returns a named tuple giving
the sample, the parameter estimates, and a convergence flag.  Extra keyword arguments are
passed to `sample` or `optimize` as appropriate to control the process:

```@repl hydro
oneboot(hydro, allow_f_increases=true, method=perfect_bounding_chain)
```

An array of the tuples produced by `oneboot` can be fed to [`addboot!`](@ref) to update
the fitting summary with precision estimates:

```julia
nboot = 2000
boots = [oneboot(hydro, allow_f_increases=true, method=perfect_bounding_chain) for i=1:nboot]
addboot!(fit1, boots)
```

On the author's workstation, this took about 5.7 minutes.  After adding the bootstrap
information, the fitting results look like this:

```
julia> summary(fit1,parnames=parnames)
name          est       se       95% CI
intercept     -0.192    0.319     (-0.858, 0.4)
altitude      -0.0573   0.015    (-0.0887, -0.0296)
temperature    0.0498   0.0361   (-0.0163, 0.126)
association    0.361    0.018      (0.326, 0.397)
```

Confidence intervals for altitude and the association parameter both exclude zero, so we
conclude that they are statistically significant.


### Error estimation 2: (parallel) bootstrap when fitting

Alternatively, the bootstrap inference procedure can be done at the same time as fitting by
providing the keyword argument `nboot` (which specifies the number of bootstrap samles to
generate) when calling `fit_pl!`. If you do this, *and* you have more than one worker
process available, then the bootstrap will be done in parallel across the workers (using an
`@distributed for` loop).  This makes it easy to achieve speed gains from parallelism on
multicore workstations.

```julia
using Distributed                  # needed for parallel computing
addprocs(6)                        # create 6 worker processes
@everywhere using Autologistic     # workers need the package loaded
fit2 = fit_pl!(hydro, nboot=2000, 
               allow_f_increases=true, method=perfect_bounding_chain)
```

In this case the 2000 bootstrap replications took about 1.1 minutes on the same 6-core
workstation. The output object `fit2` already includes the confidence intervals:

```
julia> summary(fit2, parnames=parnames)
name          est       se       95% CI
intercept     -0.192    0.33        (-0.9, 0.407)
altitude      -0.0573   0.0157   (-0.0897, -0.0297)
temperature    0.0498   0.0372   (-0.0169, 0.13)
association    0.361    0.0179     (0.327, 0.396)
```

For parallel computing of the bootstrap in other settings (eg. on a cluster), it should be
fairly simple implement in a script, using the `oneboot`/`addboot!` approach of the previous
section.

### Comparison to the centered model

(blah) (local minimia, need to provide starting values)

```julia
centered_hydro = deepcopy(hydro)
centered_hydro.coding = (0,1)
centered_hydro.centering = expectation
fit3 = fit_pl!(centered_hydro, nboot=2000, start=[-1.7, -0.17, 0.0, 1.5],
               allow_f_increases=true, method=perfect_bounding_chain)
```

```
julia> summary(fit3, parnames=parnames)
name          est       se       95% CI
intercept     -2.29     1.07       (-4.6, -0.345)
altitude      -0.16     0.0429   (-0.258, -0.088)
temperature    0.0634   0.115    (-0.138, 0.32)
association    1.51     0.0505     (1.42, 1.61)
```


(fit the model and show predictions for assoc=0)

(**** TODO: need marginals plot to color scale always from 0 to 1****)

```julia
# Make modified parameter vectors
centered_pars = getparameters(centered_hydro)
centered_pars[4] = 0.0
symmetric_pars = getparameters(hydro)
symmetric_pars[4] = 0.0

# Change models to have the new parameters
setparameters!(centered_hydro, centered_pars)
setparameters!(hydro, symmetric_pars)

# Sample to estimate marginal probabilities
centered_marg = sample(centered_hydro, 500, method=perfect_bounding_chain, average=true)
symmetric_marg = sample(hydro, 500, method=perfect_bounding_chain, average=true)

# Plot to compare
plot(myplot(centered_marg, (0,1)), myplot(symmetric_marg, (0,1)),
     layout=(1,2), size=(500,300), titlefontsize=8,
     title=["Centered Model" "Symmetric Model"])
```


### Comparison to logistic regression

(include a tip about getting logistic-comparable coefficients: either use symmetric model
and transform after, or use zero-one/model with centering=1/2)



