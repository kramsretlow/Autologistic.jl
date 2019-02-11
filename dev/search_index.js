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
    "text": "The Autologistic.jl package provides tools for analyzing correlated binary data using autologistic (AL) or autologistic regression (ALR) models.  The AL model is a multivariate probability distribution, like an analogue of the multivariate normal distribution, except for dichotomous (two-valued) categorical responses. The ALR models incorporate covariate effects into this distribution and are therefore more useful for data analysis.The ALR model is potentially useful for any situation involving correlated binary responses. It can be described in a few ways.  It is:An extension of logistic regression to handle non-independent responses.\nA Markov random field model for dichotomous random variables, with covariates.\nAn extension of the Ising model to handle different graph structures and to include covariate effects.\nThe quadratic exponential binary (QEB) distribution, incorporating covariate effects.This package follows the treatment of this model given in the paper Better Autologistic Regression.  Please refer to that article for in-depth discussion of the model, and please cite it if you use this package in your research.  The Background section in this manual also provides an overview of the model.For quick navigation, here is a table of contents, followed by an index.Pages = [\"index.md\", \"Background.md\", \"Examples.md\", \"api.md\"]\r\nDepth = 2"
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
    "text": "Here we will provide a brief overview of the autologistic model, to establish some conventions and terminology that will help you to moake appropriate use of Autologistic.jl.While we refer to binary data, more generally and accurately we should call it dichotomous data: categorical observations with two possible values (low or high, alive or dead, present or absent, etc.).  It is commonplace to encode such data as 0 or 1, but other coding choices could be made, and in Autologistic.jl the default coding is -1 and +1.  The coding choice is not trivial: two ALR models with different numeric coding will not, in general, be equivalent.  Furthermore, the (-1 1) coding has distinct advantages"
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
    "text": "TODOWe will use the notation n for the number of variables, p for the number of predictors (including intercept), m for the number of (vector) observations."
},

{
    "location": "Examples/#An-Ising-Model-1",
    "page": "Examples",
    "title": "An Ising Model",
    "category": "section",
    "text": "TODO (here show how ALsimple can be used as a probability model for dichotomous RVs)(Also show how the model can be mutated after construction)"
},

{
    "location": "Examples/#Clustered-Binary-Data-(Small-n)-1",
    "page": "Examples",
    "title": "Clustered Binary Data (Small n)",
    "category": "section",
    "text": "The retinitis pigmentosa data set obtained here is an opthalmology data set.  Both eyes of 444 patients were examined.   The response for each eye is va, an indicator of poor visual acuity (coded 0 = no, 1 = yes in the original source). Seven covariates were also recorded for each eye:aut_dom: autosomal dominant (0=no, 1=yes)\naut_rec: autosomal recessive (0=no, 1=yes)\nsex_link: sex-linked (0=no, 1=yes)\nage: age (years, range 6-80)\nsex: gender (0=female, 1=male)\npsc: posterior subscapsular cataract (0=no, 1=yes)\neye: which eye is it? (0=left, 1=right)The last four factors are relevant clinical observations, and the first three are genetic factors. The data set also includes an ID column with an ID number specific to each patient.  Eyes with the same ID come from the same person.The natural unit of analysis is the eye, but pairs of observations from the same patient are \"clustered\" because the occurrence of acuity loss in the left and right eye is probably correlated. We can model the dichotomous responses as a simple graph with two vertices and one edge, representing a single person.  We have 444 observations with this graph, each having its own set of covariates.If we include all seven predictors, plus intercept, in our model, we have (npm) = (28444). Because individual observations have only two correlated variables, we can compute the full likelihood and use standard maximum likelihood methods to do estimation and inference.using Autologistic, DataFrames, CSV, LightGraphs\r\ndf = Autologistic.datasets(\"pigmentosa\");\r\ndescribe(df)first(df,6)Which produces outputX = Array{Float64,3}(undef, 2, 8, 444)\r\nY = Array{Float64,2}(undef, 2, 444)\r\nfor i in 1:2:888\r\n    subject = Int((i+1)/2)\r\n    X[1,:,subject] = [1 permutedims(Vector(df[i,2:8]))]\r\n    X[2,:,subject] = [1 permutedims(Vector(df[i+1,2:8]))]\r\n    Y[:,subject] = convert(Array, df[i:i+1, 9])\r\nend\r\nG = Graph(2,1)\r\nLR = ALRsimple(G, X, Y=Y, coding=(0,1))  #-For logistic regression use\r\nSZO = ALRsimple(G, X, Y=Y, coding=(0,1))\r\nCZO = ALRsimple(G, X, Y=Y, coding=(0,1), centering=expectation)\r\nSPM = ALRsimple(G, X, Y=Y, coding=(-1,1))"
},

{
    "location": "Examples/#Spatial-Binary-Regression-1",
    "page": "Examples",
    "title": "Spatial Binary Regression",
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
    "location": "api/#Autologistic.AbstractPairwiseParameter",
    "page": "API Reference",
    "title": "Autologistic.AbstractPairwiseParameter",
    "category": "type",
    "text": "AbstractPairwiseParameter\n\nAbstract type representing the pairwise part of an autologistic regression model.\n\nAll concrete subtypes should have the following fields:\n\nG::SimpleGraph{Int} – The graph for the model.\ncount::Int  – The number of observations.\nA::SparseMatrixCSC{Float64,Int64}  – The adjacency matrix of the graph.\n\nIn addition to getindex() and setindex!(), any concrete subtype  P<:AbstractPairwiseParameter should also have the following methods defined:\n\ngetparameters(P), returning a Vector{Float64}\nsetparameters!(P, newpar::Vector{Float64}) for setting parameter values.\n\nNote that indexing is performance-critical and should be implemented carefully in  subtypes.  \n\nThe intention is that each subtype should implement a different way of parameterizing the association matrix. The way parameters are stored and values computed is up to the subtypes. \n\nThis type inherits from AbstractArray{Float64, 3}.  The third index is to allow for  multiple observations. P[:,:,r] should return the association matrix of the rth observation in an appropriate subtype of AbstractMatrix.  It is not intended that the third  index will be used for range or vector indexing like P[:,:,1:5] (though this may work  due to AbstractArray fallbacks). \n\nExamples\n\njulia> M = ALsimple(Graph(4,4));\njulia> typeof(M.pairwise)\nSimplePairwise\njulia> isa(M.pairwise, AbstractPairwiseParameter)\ntrue\n\n\n\n\n\n"
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
    "text": "AbstractPairwiseParameter\r\nAbstractAutologisticModel\r\nSamplingMethods"
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
