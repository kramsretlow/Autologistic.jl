var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Autologistic Regression Modelling in Julia",
    "title": "Autologistic Regression Modelling in Julia",
    "category": "page",
    "text": ""
},

{
    "location": "#Autologistic-Regression-Modelling-in-Julia-1",
    "page": "Autologistic Regression Modelling in Julia",
    "title": "Autologistic Regression Modelling in Julia",
    "category": "section",
    "text": "Pages = [\"index.md\", \"api.md\"]\r\nDepth = 2This is sample documentation.  Here is a link to the fullPMF method, just to practice linking."
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
    "location": "api/#Types-and-Constructors-1",
    "page": "API Reference",
    "title": "Types and Constructors",
    "category": "section",
    "text": "TODO"
},

{
    "location": "api/#Autologistic.fullPMF",
    "page": "API Reference",
    "title": "Autologistic.fullPMF",
    "category": "function",
    "text": "fullPMF(M::AbstractAutologisticModel; replicates=nothing, force::Bool=false)\n\nCompute the PMF of an AbstractAutologisticModel, and return a NamedTuple (:table, :partition).\n\nFor an AutologisticModel with n observations and m replicates, :table is a 2^n(n+1)m  array of Float64. Each page of the 3D array holds a probability table for a replicate.   Each row of the table holds a specific configuration of the responses, with the  corresponding probability in the last column.  In the m=1 case,  :table is a 2D array.\n\nOutput :partition is a vector of normalizing constant (a.k.a. partition function) values. In the m=1 case, it is a scalar Float64.\n\nArguments\n\nM::AbstractAutologisticModel: an autologistic model.\nreplicates=nothing: indices of specific replicates from which to obtain the output. By  default, all replicates are used.\nforce::Bool=false: calling the function with n20 will throw an error unless  force=true. \n\nExamples\n\njulia> M = ALRsimple(Graph(3,0),ones(3,1));\njulia> pmf = fullPMF(M);\njulia> pmf.table\n8×4 Array{Float64,2}:\n -1.0  -1.0  -1.0  0.125\n -1.0  -1.0   1.0  0.125\n -1.0   1.0  -1.0  0.125\n -1.0   1.0   1.0  0.125\n  1.0  -1.0  -1.0  0.125\n  1.0  -1.0   1.0  0.125\n  1.0   1.0  -1.0  0.125\n  1.0   1.0   1.0  0.125\njulia> pmf.partition\n 8.0\n\n\n\n\n\n"
},

{
    "location": "api/#Methods-1",
    "page": "API Reference",
    "title": "Methods",
    "category": "section",
    "text": "fullPMF"
},

]}
