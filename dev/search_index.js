var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Introduction",
    "title": "Introduction",
    "category": "page",
    "text": ""
},

{
    "location": "#Introduction-1",
    "page": "Introduction",
    "title": "Introduction",
    "category": "section",
    "text": "The Autologistic.jl package provides tools for analyzing correlated binary data using autologistic (AL) or autologistic regression (ALR) models.  The AL model is best thought of as a probability distribution for correlated binary random variables, like an analogue of the multivariate normal distribution for binary responses. The ALR models incorporate covariate effects into this distribution and are therefore more useful for data analysis.The ALR model is potentially useful for any situation involving correlated binary responses. It can be described in a few ways.  It is:An extension of logistic regression to handle non-independent responses.\nA Markov random field model for dichotomous random variables, with covariates.\nAn extension of the Ising model to handle different graph structures and to include covariate effects.\nThe quadratic exponential binary (QEB) distribution, incorporating covariate effects.This package follows the treatment of this model given in the paper Better Autologistic Regression.  Please refer to that article for in-depth discussion of the model, and please cite it if you use this package in your research.  The Background section in this manual also provides an overview of the model.For quick navigation, here is a table of contents, followed by an index.Pages = [\"index.md\", \"Background.md\", \"Examples.md\", \"api.md\"]\r\nDepth = 2"
},

{
    "location": "Background/#",
    "page": "Background",
    "title": "Background",
    "category": "page",
    "text": ""
},

{
    "location": "Background/#Background-1",
    "page": "Background",
    "title": "Background",
    "category": "section",
    "text": "TODO"
},

{
    "location": "Background/#The-Autologistic-(AL)-Model-1",
    "page": "Background",
    "title": "The Autologistic (AL) Model",
    "category": "section",
    "text": "TODO"
},

{
    "location": "Background/#The-Autologistic-Regression-(ALR)-Model-1",
    "page": "Background",
    "title": "The Autologistic Regression (ALR) Model",
    "category": "section",
    "text": "TODO"
},

{
    "location": "Background/#Design-of-the-Package-1",
    "page": "Background",
    "title": "Design of the Package",
    "category": "section",
    "text": "TODO - list type design and how it\'s planned to be used."
},

{
    "location": "Examples/#",
    "page": "Examples",
    "title": "Examples",
    "category": "page",
    "text": ""
},

{
    "location": "Examples/#Examples-1",
    "page": "Examples",
    "title": "Examples",
    "category": "section",
    "text": "TODO"
},

{
    "location": "Examples/#An-Ising-Model-1",
    "page": "Examples",
    "title": "An Ising Model",
    "category": "section",
    "text": "TODO"
},

{
    "location": "Examples/#Low-Dimensional-Correlated-Binary-Data-1",
    "page": "Examples",
    "title": "Low-Dimensional Correlated Binary Data",
    "category": "section",
    "text": "TODO"
},

{
    "location": "Examples/#High-Dimensional-Spatial-Binary-Data-1",
    "page": "Examples",
    "title": "High-Dimensional Spatial Binary Data",
    "category": "section",
    "text": "TODO"
},

{
    "location": "api/#",
    "page": "API Reference",
    "title": "API Reference",
    "category": "page",
    "text": ""
},

{
    "location": "api/#API-Reference-1",
    "page": "API Reference",
    "title": "API Reference",
    "category": "section",
    "text": "CurrentModule = Autologistic"
},

{
    "location": "api/#Autologistic.AbstractAutologisticModel",
    "page": "API Reference",
    "title": "Autologistic.AbstractAutologisticModel",
    "category": "type",
    "text": "AbstractAutologisticModel\n\nAbstract type representing autologistic models.  This type has methods defined for most operations one will want to perform, so that concrete subtypes should not have to define too many methods unless more specialized and efficient algorithms for the specific subtype.\n\nAll concrete subtypes should have the following fields:\n\nresponses::Array{Bool,2} – The binary observations. Rows are for nodes in the    graph, and columns are for independent (vector) observations.  It is a 2D array even if    there is only one observation.\nunary<:AbstractUnaryParameter – Specifies the unary part of the model.\npairwise<:AbstractPairwiseParameter  – Specifies the pairwise part of the model    (including the graph).\ncentering<:CenteringKinds – Specifies the form of centering used, if any.\ncoding::Tuple{T,T} where T<:Real – Gives the numeric coding of the responses.\nlabels::Tuple{String,String} – Provides names for the high and low states.\ncoordinates<:SpatialCoordinates – Provides 2D or 3D coordinates for each vertex in    the graph (or nothing if no coordinates).\n\nThe following functions are defined for the abstract type, and are considered part of the  type\'s interface (in this list, M of type inheriting from AbstractAutologisticModel).\n\ngetparameters(M) and setparameters!(M, newpars::Vector{Float64})\ngetunaryparameters(M) and setunaryparameters!(M, newpars::Vector{Float64})\ngetpairwiseparameters(M) and setpairwiseparameters!(M, newpars::Vector{Float64})\nmakecoded(M, Y)\ncenteringterms(M, kind::Union{Nothing,CenteringKinds})\npseudolikelihood(M)\nnegpotential(M)\nfullPMF(M; indices, force::Bool)\nmarginalprobabilities(M; indices, force::Bool)\nconditionalprobabilities(M; vertices, indices)\nsample(M, k::Int, method::SamplingMethods, indices::Int, average::Bool, start,    burnin::Int, verbose::Bool)\n\nThe sample() function is a wrapper for a variety of random sampling algorithms enumerated in SamplingMethods.\n\nExamples\n\njulia> M = ALsimple(Graph(4,4));\njulia> typeof(M)\nALsimple{CenteringKinds,Int64,Nothing}\njulia> isa(M, AbstractAutologisticModel)\ntrue\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.SamplingMethods",
    "page": "API Reference",
    "title": "Autologistic.SamplingMethods",
    "category": "type",
    "text": "SamplingMethods\n\nAn enumeration to facilitate choosing a method for sampling. Available choices are:\n\nGibbs  TODO\nperfect_bounding_chain  TODO\nperfect_reuse_samples  TODO \nperfect_reuse_seeds  TODO\nperfect_read_once  TODO\n\n\n\n\n\n"
},

{
    "location": "api/#Types-and-Constructors-1",
    "page": "API Reference",
    "title": "Types and Constructors",
    "category": "section",
    "text": "AbstractAutologisticModel\r\nSamplingMethods"
},

{
    "location": "api/#Autologistic.fullPMF",
    "page": "API Reference",
    "title": "Autologistic.fullPMF",
    "category": "function",
    "text": "fullPMF(M::AbstractAutologisticModel; indices=1:size(M.unary,2), force::Bool=false)\n\nCompute the PMF of an AbstractAutologisticModel, and return a NamedTuple (:table, :partition).\n\nFor an AutologisticModel with n variables and m observations, :table is a 2^n(n+1)m  array of Float64. Each page of the 3D array holds a probability table for an observation.   Each row of the table holds a specific configuration of the responses, with the  corresponding probability in the last column.  In the m=1 case,  :table is a 2D array.\n\nOutput :partition is a vector of normalizing constant (a.k.a. partition function) values. In the m=1 case, it is a scalar Float64.\n\nArguments\n\nM: an autologistic model.\nindices: indices of specific observations from which to obtain the output. By  default, all observations are used.\nforce: calling the function with n20 will throw an error unless  force=true. \n\nExamples\n\njulia> M = ALRsimple(Graph(3,0),ones(3,1));\njulia> pmf = fullPMF(M);\njulia> pmf.table\n8×4 Array{Float64,2}:\n -1.0  -1.0  -1.0  0.125\n -1.0  -1.0   1.0  0.125\n -1.0   1.0  -1.0  0.125\n -1.0   1.0   1.0  0.125\n  1.0  -1.0  -1.0  0.125\n  1.0  -1.0   1.0  0.125\n  1.0   1.0  -1.0  0.125\n  1.0   1.0   1.0  0.125\njulia> pmf.partition\n 8.0\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.sample",
    "page": "API Reference",
    "title": "Autologistic.sample",
    "category": "function",
    "text": "sample(\n    M::AbstractAutologisticModel, \n    k::Int = 1;\n    method::SamplingMethods = Gibbs,\n    indices = 1:size(M.unary,2), \n    average::Bool = false, \n    start = nothing, \n    burnin::Int = 0,\n    verbose::Bool = false\n)\n\nDraws k random samples from autologistic model M, and either returns the samples  themselves, or the estimated probabilities of observing the \"high\" level at each vertex.\n\nIf the model has more than one observation, then k samples are drawn for each observation. To restrict the samples to a subset of observations, use argument indices. \n\nFor a model M with n vertices in its graph:\n\nWhen average=false, the return value is n × length(indices) × k, with singleton   dimensions dropped. \nWhen average=true, the return value is n  × length(indices), with singleton   dimensions dropped.\n\nKeyword Arguments\n\nmethod is a member of the enum SamplingMethods, specifying which sampling method will be used.  The default uses Gibbs sampling.  Where feasible, it is recommended  to use one of the perfect sampling alternatives. See SamplingMethods for more.\n\nindices gives the indices of the observation to use for sampling. The default is all indices, in which case each sample is of the same size as M.responses. \n\naverage controls the form of the output. When average=true, the return value is the  proportion of \"high\" samples at each vertex. (Note that this is not actually the arithmetic average of the samples, unless the coding is (0,1). Rather, it is an estimate of  the probability of getting a \"high\" outcome.)  When average=false, the full set of samples is returned. \n\nstart allows a starting configuration of the random variables to be provided. Only used if method=Gibbs. Any vector with two unique values can be used as start. By default a random configuration is used.\n\nburnin specifies the number of initial samples to discard from the results.  Only used if method=Gibbs.\n\nverbose controls output to the console.  If true, intermediate information about  sampling progress is printed to the console. Otherwise no output is shown.\n\nExamples\n\njulia> M = ALsimple(Graph(4,4));\njulia> M.coding = (-2,3);\njulia> r = sample(M,10);\njulia> size(r)\n(4, 10)\njulia> sort(unique(r))\n2-element Array{Float64,1}:\n -2.0\n  3.0\n\n\n\n\n\n"
},

{
    "location": "api/#Methods-1",
    "page": "API Reference",
    "title": "Methods",
    "category": "section",
    "text": "fullPMF\r\nsample"
},

]}
