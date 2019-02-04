# Examples

TODO

We will use the notation ``n`` for the number of variables, ``p`` for the number of
predictors (including intercept), ``m`` for the number of (vector) observations.

## An Ising Model

TODO

## Clustered Binary Data (Small ``n``)

The *retinitis pigmentosa* data set [obtained here](https://sites.google.com/a/channing.harvard.edu/bernardrosner/channing/regression-method-when-the-eye-is-the-unit-of-analysis) is an opthalmology data set.  Both eyes of 444 patients were examined.  
The response for each eye is **va**, an indicator of poor visual acuity (coded 0 = no,
1 = yes in the original source). Seven covariates were also recorded for each eye:

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
is probably correlated. We can model the dichotomous responses as a simple graph with two
vertices and one edge, representing a single person.  We have 444 observations with this
graph, each having its own set of covariates.

If we include all seven predictors, plus intercept, in our model, we have
``(n,p,m) = (2,8,444)``. Because individual observations have only two correlated variables,
we can compute the full likelihood and use standard maximum likelihood methods to do
estimation and inference.

```@repl pigmentosa
using Autologistic, DataFrames, CSV, LightGraphs
df = Autologistic.datasets("pigmentosa");
describe(df)
```

```@example pigmentosa; continued = true
first(df,6)
```

Which produces output

```@example pigmentosa
```

```@example pigmentosa
X = Array{Float64,3}(undef, 2, 8, 444)
Y = Array{Float64,2}(undef, 2, 444)
for i in 1:2:888
    subject = Int((i+1)/2)
    X[1,:,subject] = [1 permutedims(Vector(df[i,2:8]))]
    X[2,:,subject] = [1 permutedims(Vector(df[i+1,2:8]))]
    Y[:,subject] = convert(Array, df[i:i+1, 9])
end
G = Graph(2,1)
LR = ALRsimple(G, X, Y=Y, coding=(0,1))  #-For logistic regression use
SZO = ALRsimple(G, X, Y=Y, coding=(0,1))
CZO = ALRsimple(G, X, Y=Y, coding=(0,1), centering=expectation)
SPM = ALRsimple(G, X, Y=Y, coding=(-1,1))
```

## Spatial Binary Regression

TODO

