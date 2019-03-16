# Autologistic

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://kramsretlow.github.io/Autologistic.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://kramsretlow.github.io/Autologistic.jl/dev)
[![Build Status](https://travis-ci.com/kramsretlow/Autologistic.jl.svg?branch=master)](https://travis-ci.com/kramsretlow/Autologistic.jl)
[![codecov](https://codecov.io/gh/kramsretlow/Autologistic.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kramsretlow/Autologistic.jl)

A Julia package for computing with the autologistic (Ising) probability model
and performing autologistic regression.

Autologistic regression is like an extension of logistic regression that allows the binary
responses to be correlated.  An undirected graph is used to encode the association structure
among the responses.

The package follows the treatment of this model given in the paper
[Better Autologistic Regression](https://doi.org/10.3389/fams.2017.00024).  As described in
that paper, different variants of "the" autologistic regression model are actually different
probability models. One reason this package was created was to allow researchers to compare
the performance of the different model variants.  You can create different variants of
the model easily and fit them using either maximum likelhood (for small-n cases) or maximum
pseudolikelihood (for large-n cases).

Much more detail is provided in the [documentation](https://kramsretlow.github.io/Autologistic.jl/stable).

```julia
# To get a feeling for the package facilities.
# The package uses LightGraphs.jl for graphs.
using Autologistic, LightGraphs
g = Graph(100, 400)            #-Create a random graph (100 vertices, 400 edges)
X = [ones(100) rand(100,3)]    #-Random matrix of predictors.
Y = rand([0, 1], 100)          #-Random binary responses.
model = ALRsimple(g, X, Y=Y)   #-Create autologistic regression model

# Estimate parametersr pseudolikelihood. Do parametric bootstrap for error
# estimation.  Draw bootstrap samples using perfect sampling.
fit = fit_pl!(model, nboot=2000, method=perfect_read_once)

# Draw samples from the fitted model and get the average to estimate
# the marginal probability distribution. Use a different perfect sampling
# algorithm.
marginal = sample(model, 1000, method=perfect_bounding_chain, average=true)
```