# Examples

These examples demonstrate most of the functionality of the package, its typical usage, and
how to make some plots you might want to use.

The examples:

* [An Ising Model](@ref) shows how to use the package to explore the autologistic
  probability distribution, without concern about covariates or parameter estimation.
* [Clustered Binary Data (Small ``n``)](@ref) shows how to use the package for regression
  analysis of correlated binary responses when the graph is small enough to permit
  computation of the normalizing constant.
* [Spatial Binary Regression](@ref) shows how to use the package for autologistic regression
  analysis for larger, spatially-referenced graphs.

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

![marginal-probs.png](/assets/marginal-probs.png)

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

![ising_gif.gif](/assets/ising_gif.gif)

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

TODO.  Create the graph using spatialgraph and plot the endogenous probabilities (gplot);
fit the ALRsimple model and do inference with parametric bootstrap; show how alternative
models (e.g. centered ALR model) can be made and compared.  Show the estimated fitted
probabilities of symmetric and centered models (and/or predicted probs for altitude=0?)

