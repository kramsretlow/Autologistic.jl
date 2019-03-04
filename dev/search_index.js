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
    "text": "The Autologistic.jl package provides tools for analyzing correlated binary data using autologistic (AL) or autologistic regression (ALR) models.  The AL model is a multivariate probability distribution for dichotomous (two-valued) categorical responses. The ALR models incorporate covariate effects into this distribution and are therefore more useful for data analysis.The ALR model is potentially useful for any situation involving correlated binary responses. It can be described in a few ways.  It is:An extension of logistic regression to handle non-independent responses.\nA Markov random field model for dichotomous random variables, with covariates.\nAn extension of the Ising model to handle different graph structures and to include covariate effects.\nThe quadratic exponential binary (QEB) distribution, incorporating covariate effects.This package follows the treatment of this model given in the paper Better Autologistic Regression.  Please refer to that article for in-depth discussion of the model, and please cite it if you use this package in your research.  The Background section in this manual also provides an overview of the model."
},

{
    "location": "#Contents-1",
    "page": "Introduction",
    "title": "Contents",
    "category": "section",
    "text": "Pages = [\"index.md\", \"Background.md\", \"Design.md\", \"BasicUsage.md\", \"Examples.md\", \"api.md\"]\r\nDepth = 2"
},

{
    "location": "#Reference-Index-1",
    "page": "Introduction",
    "title": "Reference Index",
    "category": "section",
    "text": "The following topics are documented in the Reference section:"
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
    "text": "This section provides a brief overview of the autologistic model, to establish some conventions and terminology that will help you to make appropriate use of Autologistic.jl.The package is concerned with the analysis of dichotomous data: categorical observations that can take two possible values (low or high, alive or dead, present or absent, etc.).   It is common to refer to such data as binary, and to use numeric values 0 and 1 to represent the two states, but the two numbers we choose to represent the two states is an arbitrary choice. This might seem like a small detail, but for autologistic regression models, the choice of numeric coding is very important. The pair of  values used to represent the two states is called the coding.note: Important Fact 1\nFor ALR models, two otherwise-identical models that differ only in their coding will generally not be equivalent probability models.  Changing the coding fundamentally changes the model. For a variety of reasons, the (-11) coding is strongly recommended, and is used by default.Logistic regression is the most common model for independent binary responses.   Autologistic models are one way to model correlated binary/dichotomous responses."
},

{
    "location": "Background/#The-Autologistic-(AL)-Model-1",
    "page": "Background",
    "title": "The Autologistic (AL) Model",
    "category": "section",
    "text": "Let mathbfY be a vector of n dichotomous random variables, expressed using any chosen coding.  The AL model is a probability model for the joint probabiliyt mass function (PMF) of the random vector:Pr(mathbfY=mathbfy) propto expleft(mathbfy^Tboldsymbolalpha -\r\nmathbfy^TboldsymbolLambdaboldsymbolmu +\r\nfrac12mathbfy^TboldsymbolLambdamathbfyright)The model is only specified up to a proportionality constant.  The proportionality constant (sometimes called the \"partition function\") is intractable for even moderately large n: evaluating it requires computing the right hand side of the above equation for 2^n possible configurations of the dichotomous responses.Inside the exponential of the PMF there are three terms:The first term is the unary term, and mathbfalpha is called the unary parameter.  It summarizes each variable\'s endogenous tendency to take the \"high\" state (larger positive alpha_i values make random variable Y_i more likely to take the \"high\" value).  Note that in practical models, mathbfalpha could be expressed in terms of some other parameters.\nThe second term is an optional centering term, and the value mu_i is called the centering adjustment for variable i.  The package includes different options for centering, in the CenteringKinds enumeration.  Setting centering to none will set the centering adjustment to zero; setting centering to expectation will use the centering adjustment of the \"centered autologistic model\" that has appeared in the literature (e.g. here and here).note: Important Fact 2\nJust as with coding, changing an un-centered model to a centered one is not a minor change.  It produces a different probability model entirely.  Again, there is evidence that centering has drawbacks, so the uncentered model is used by default.The third term is the pairwise term, which handles the association between the random variables.  Parameter boldsymbolLambda is a symmetric matrix.  If it has a nonzero entry at position (ij), then variables i and j share an edge in the graph associated with the model, and the value of the entry controls the strength of association between those two variables.The autogologistic model is a probabilistic graphical model, more specifically a Markov random field, meaning it has an undirected graph that encodes conditional probability relationships among the variables. Autologistic.jl uses LightGraphs.jl to represent the graph."
},

{
    "location": "Background/#The-Autologistic-Regression-(ALR)-Model-1",
    "page": "Background",
    "title": "The Autologistic Regression (ALR) Model",
    "category": "section",
    "text": "TODO"
},

{
    "location": "Background/#The-Symmetric-Model-and-Logistic-Regression-1",
    "page": "Background",
    "title": "The Symmetric Model and Logistic Regression",
    "category": "section",
    "text": "(show conditional form and logistic regression connection; mention transforming to make comparable parmaters between the symmetric ALR model and the logistic model)"
},

{
    "location": "Design/#",
    "page": "Design of the Package",
    "title": "Design of the Package",
    "category": "page",
    "text": ""
},

{
    "location": "Design/#Design-of-the-Package-1",
    "page": "Design of the Package",
    "title": "Design of the Package",
    "category": "section",
    "text": "In the Background section, it was strongly encouraged to use the symmetric autologistic model.  Still, the package allows the user to construct AL/ALR models with different choices of centering and coding, to compare the merits of different choices for themselves. The package was also built to allow different ##### TODO #####In this package, the responses are always stored as arrays of type Bool, to separate the configuration of low/high responses from the choice of coding. If M is an AL or ALR model type, the field M.coding holds the numeric coding as a 2-tuple.TODO - list type design and how it\'s planned to be used."
},

{
    "location": "BasicUsage/#",
    "page": "Basic Usage",
    "title": "Basic Usage",
    "category": "page",
    "text": ""
},

{
    "location": "BasicUsage/#Basic-Usage-1",
    "page": "Basic Usage",
    "title": "Basic Usage",
    "category": "section",
    "text": "Typical usage of the package will involve the following three steps:1. Create a model object.All particular AL/ALR models are instances of subtypes of AbstractAutologisticModel.  Each subtype is defined by a particular choice for the parametrization of the unary and pairwise parts.  At present the options are:ALfull: A model with type FullUnary as the unary part, and type FullPairwise as the pairwise part (parameters α Λ).\nALsimple: A model with type FullUnary as the unary part, and type SiimplePairwise as the pairwise part (parameters α λ).\nALRsimple: A model with type LinPredUnary as the unary part, and type SimplePairwise as the pairwise part (parameters β λ).The first two types above are mostly for research or exploration purposes.  Most users doing data analysis will use the ALRsimple model.  Each of the above types have various constructors defined.  For example, ALRsimple(G, X) will create an ALRsimple model with graph G and predictor matrix X.  Type, e.g., ?ALRsimple at the REPL to see the constructors #### <== TODO ####.The package is designed to be extensible if other parametrizations of the unary or pairwise parts are desired.  For example, it is planned eventually to add a new pairwise type that will allow the level of association to vary across the grpah.  When such a type appears, additional ALR model types will be created.Any of the above model types can be used with any of the supported forms of centering, and with any desired coding. This facilitates comparison of different model variants.2. Set parameters.Depending on the constructor used, the model just initialized will have either default  parameter values or user-specified parameter values.  Usually it will be desired to choose some appropriate values from data.fit_ml! uses maximum likelihood to estimate the parameters.  It is only useful for cases where the number of vertices in the graph is small.\nfit_pl! uses pseudolikelihood to estimate the parameters.\nsetparameters!, setunaryparameters!, and setpairwiseparameters! can be used to set the parameters of the model directly.\ngetparameters, getunaryparameters, and getpairwiseparameters can be used to retrieve the parameter values.Changing the parameters directly, through the fields of the model object, is discouraged.  It is preferable for safety to use the above get and set functions.3. Inference and exploration.After parameter estimation, one typically wants to use the fitted model to answer inference questions, make plots, and the like.For small-graph cases:fit_ml! returns p-values and 95% confidence intervals that can be used directly.\nfullPMF, conditionalprobabilities, marginalprobabilities can be used to get desired quantities from the fitted distribution.\nsample can be used to draw random samples from the fitted distribution.For large-graph cases:If using fit_pl!, ##### TODO #####\nSampling can be used to estimate desired quantities like marginal probabilities.  The sample function implements Gibbs sampling as well as several perfect sampling algorithms.Plotting can be done using standard Julia capabilities.  The Examples section shows how to make a few relevant plots."
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
    "text": "These examples demonstrate most of the functionality of the package, its typical usage, and how to make some plots you might want to use.The examples:An Ising Model shows how to use the package to explore the autologistic probability distribution, without concern about covariates or parameter estimation.\nClustered Binary Data (Small n) shows how to use the package for regression analysis of correlated binary responses when the graph is small enough to permit computation of the normalizing constant.\nSpatial Binary Regression shows how to use the package for autologistic regression analysis for larger, spatially-referenced graphs."
},

{
    "location": "Examples/#An-Ising-Model-1",
    "page": "Examples",
    "title": "An Ising Model",
    "category": "section",
    "text": "The term \"Ising model\" is usually used to refer to a Markov random field of dichotomous random variables on a regular lattice.  The graph is such that each variable shares an edge only with its nearest neighbors in each dimension.  It\'s a traditional model for magnetic spins, where the coding (-11) is usually used. There\'s one parameter per vertex (a \"local magnetic field\") that increases or decreases the chance of getting a +1 state at that vertex; and there\'s a single pairwise parameter that controls the strength of interaction between neighbor states.In our terminology it\'s just an autologistic model with the appropriate graph. Specifically, it\'s an ARsimple model: one with FullUnary type unary parameter, and SimplePairwise type pairwise parameter.We can create such a model once we have the graph.  For example, let\'s create two 30-by-30 lattices: one without any special handling of the boundary, and one with periodic boundary conditions. This can be done with LightGraphs.jl\'s Grid function.n = 30  # numer of rows and columns\r\nusing LightGraphs\r\nG1 = Grid([n, n], periodic=false)\r\nG2 = Grid([n, n], periodic=true)\r\nnothing # hideNow create an AL model for each case. Initialize the unary parameters to Gaussian white noise. By default the pairwise parameter is set to zero, which implies independence of the variables. Use the same parameters for the two models, so the only difference betwen them is the graph.using Random, Autologistic\r\nRandom.seed!(8888)\r\nα = randn(n^2)\r\nM1 = ALsimple(G1, α)\r\nM2 = ALsimple(G2, α)The REPL output shows information about the model.  It\'s an ALsimple type with one observation of length 900.The conditionalprobabilities function returns the probablity of observing a +1 state at each vertex, conditional on the vertex\'s neighbor values. These can be visualized as an image, using a heatmap (from Plots.jl):using Plots\r\ncondprobs = conditionalprobabilities(M1)\r\nhm = heatmap(reshape(condprobs, n, n), c=:grays, aspect_ratio=1,\r\n             title=\"probability of +1 under independence\")\r\nplot(hm)Since the association parameter is zero, there are no neighborhood effects.  The above conditional probabilities are equal to the marginal probabilities.Next, set the association parameters to 0.75, a fairly strong association level, to introduce a neighbor effect.setpairwiseparameters!(M1, [0.75])\r\nsetpairwiseparameters!(M2, [0.75])\r\nnothing # hideA quick way to see the effect of this parameter is to observe random samples from the models. The sample function can be used to do this. For this example, use perfect sampling using a bounding chain algorithm (the enumeration SamplingMethods lists the available sampling options).s1 = sample(M1, method=perfect_bounding_chain)\r\ns2 = sample(M2, method=perfect_bounding_chain)\r\nnothing #hideThe samples can also be visualized using heatmap:pl1 = heatmap(reshape(s1, n, n), c=:grays, colorbar=false, title=\"regular boundary\");\r\npl2 = heatmap(reshape(s2, n, n), c=:grays, colorbar=false, title=\"periodic boundary\");\r\nplot(pl1, pl2, size=(800,400), aspect_ratio=1)In these plots, black indicates the low state, and white the high state.  A lot of local clustering is occurring in the samples due to the neighbor effects.To see the long-run differences between the two models, we can look at the marginal probabilities. They can be estimated by drawing many samples and averaging them (note that running this code chunk can take a couple of minutes):marg1 = sample(M1, 500, method=perfect_bounding_chain, verbose=true, average=true)\r\nmarg2 = sample(M2, 500, method=perfect_bounding_chain, verbose=true, average=true)\r\npl3 = heatmap(reshape(marg1, n, n), c=:grays, colorbar=false, title=\"regular boundary\");\r\npl4 = heatmap(reshape(marg2, n, n), c=:grays, colorbar=false, title=\"periodic boundary\");\r\nplot(pl3, pl4, size=(800,400), aspect_ratio=1)\r\nsavefig(\"marginal-probs.png\")The figure marginal-probs.png looks like this:(Image: marginal-probs.png)Although the differences between the two marginal distributions are not striking, the extra edges connecting top to bottom and left to right do have some influence on the probabilities at the periphery of the square.As a final demonstration, perform Gibbs sampling for model M2, starting from a random state.  Display a gif animation of the progress.nframes = 150\r\ngibbs_steps = sample(M2, nframes, method=Gibbs)\r\nanim = @animate for i =  1:nframes\r\n    heatmap(reshape(gibbs_steps[:,i], n, n), c=:grays, colorbar=false, \r\n            aspect_ratio=1, title=\"Gibbs sampling: step $(i)\")\r\nend\r\ngif(anim, \"ising_gif.gif\", fps=10)(Image: ising_gif.gif)"
},

{
    "location": "Examples/#Clustered-Binary-Data-(Small-n)-1",
    "page": "Examples",
    "title": "Clustered Binary Data (Small n)",
    "category": "section",
    "text": "The retinitis pigmentosa data set obtained here is an opthalmology data set.  The data comes from 444 patients that had both eyes examined.  The data can be loaded with Autologistic.datasets:using Autologistic, DataFrames, LightGraphs\r\ndf = Autologistic.datasets(\"pigmentosa\");\r\nfirst(df, 6)\r\ndescribe(df)The response for each eye is va, an indicator of poor visual acuity (coded 0 = no, 1 = yes in the loaded data set). Seven covariates were also recorded for each eye:aut_dom: autosomal dominant (0=no, 1=yes)\naut_rec: autosomal recessive (0=no, 1=yes)\nsex_link: sex-linked (0=no, 1=yes)\nage: age (years, range 6-80)\nsex: gender (0=female, 1=male)\npsc: posterior subscapsular cataract (0=no, 1=yes)\neye: which eye is it? (0=left, 1=right)The last four factors are relevant clinical observations, and the first three are genetic factors. The data set also includes an ID column with an ID number specific to each patient.  Eyes with the same ID come from the same person.The natural unit of analysis is the eye, but pairs of observations from the same patient are \"clustered\" because the occurrence of acuity loss in the left and right eye is probably correlated. We can model each person\'s two eyes\' va outcomes as two dichotomous random variables with a 2-vertex, 1-edge graph.G = Graph(2,1)Each of the 444 observations has this graph, and each has its own set of covariates.If we include all seven predictors, plus intercept, in our model, we have 2 variables per observation, 8 predictors, and 444 obsrevations. Before creating the model we need to re-structure the covariates. The data in df has one row per eye, with the variable ID indicating which eyes belong to the same patient.  We need to rearrange the responses (Y) and the predictors (X) into arrays suitable for our autologistic models, namely:Y is 2 times 444 with one observation per column.\nX is 2 times 8 times 444 with one 2 times 8 matrix of predictors for each observation.  The first column of each predictor matrix is an intercept column, and   columns 2 through 8 are for aut_dom, aut_rec, sex_link, age, sex, psc, and eye, respectively.X = Array{Float64,3}(undef, 2, 8, 444);\r\nY = Array{Float64,2}(undef, 2, 444);\r\nfor i in 1:2:888\r\n    subject = Int((i+1)/2)\r\n    X[1,:,subject] = [1 permutedims(Vector(df[i,2:8]))]\r\n    X[2,:,subject] = [1 permutedims(Vector(df[i+1,2:8]))]\r\n    Y[:,subject] = convert(Array, df[i:i+1, 9])\r\nendFor example, patient 100 had responsesY[:,100]Indicating visual acuity loss in the left eye, but not in the right. The predictors for this individual areX[:,:,100]Now we can create our autologistic regression model.model = ALRsimple(G, X, Y=Y)This creates a model with the \"simple pairwise\" structure, using a single association parameter. The default is to use no centering adjustment, and to use coding (-11) for the responses.  This \"symmetric\" version of the model is recommended for a variety of reasons.  Using different coding or centering choices is only recommended if you have a thorough understanding of what you are doing; but if you wish to use different choices, this can easily be done using keyword arguments. For example, ALRsimple(G, X, Y=Y, coding=(0,1), centering=expectation) creates the \"centered autologistic model\" that has appeared in the literature, e.g. here and hereThe model has nine parameters (eight regression coefficients plus the association parameter).  All parameters are initialized to zero:getparameters(model)When we call getparameters, the vector returned always has the unary parameters first, with the pairwise parameter(s) appended at the end.Because there are only two vertices in the graph, we can use the full likelihood (fit_ml! function) to do parameter estimation.  This function returns a structure with the estimates as well as standard errors, p-values, and 95% confidence intervals for the  parameter estimates.out = fit_ml!(model)To view the estimation results, use summary:summary(out, parnames = [\"icept\", \"aut_dom\", \"aut_rec\", \"sex_link\", \"age\", \"sex\", \r\n        \"psc\", \"eye\", \"λ\"])From this we see that the association parameter is fairly large (0.818), supporting the idea that the left and right eyes are associated.  It is also highly statistically significant.  Among the covariates, sex_link, age, and psc are all statistically significant."
},

{
    "location": "Examples/#Spatial-Binary-Regression-1",
    "page": "Examples",
    "title": "Spatial Binary Regression",
    "category": "section",
    "text": "TODO.  Create the graph using spatialgraph and plot the endogenous probabilities (gplot); fit the ALRsimple model and do inference with parametric bootstrap; show how alternative models (e.g. centered ALR model) can be made and compared.  Show the estimated fitted probabilities of symmetric and centered models (and/or predicted probs for altitude=0?)"
},

{
    "location": "api/#",
    "page": "Reference",
    "title": "Reference",
    "category": "page",
    "text": ""
},

{
    "location": "api/#Reference-1",
    "page": "Reference",
    "title": "Reference",
    "category": "section",
    "text": "CurrentModule = Autologistic"
},

{
    "location": "api/#Index-1",
    "page": "Reference",
    "title": "Index",
    "category": "section",
    "text": ""
},

{
    "location": "api/#Autologistic.ALfit",
    "page": "Reference",
    "title": "Autologistic.ALfit",
    "category": "type",
    "text": "ALfit\n\nA type to hold estimation output for autologistic models.  Fitting functions return an  object of this type.\n\nDepending on the fitting method, some fields might not be set.  Fields that are not used are set to nothing or to zero-dimensional arrays.  The fields are:\n\nestimate: A vector of parameter estimates\nse: A vector of standard errors for the estimates\npvalues: A vector of p-values for testing the null hypothesis that the parameters equal zero (one-at-a time hypothesis tests).\nCIs: A vector of 95% confidence intervals for the parameters (a vector of 2-tuples).\noptim: the output of the call to optimize used to get the estimates.\nHinv (used by fit_ml!): The inverse of the Hessian matrix of the objective function,  evaluated at the estimate.\nnboot - (fit_pl!) number of bootstrap samples to use for error estimation\nkwargs - (fit_pl!) string representation of extra keyword arguments passed in the call (a record of which arguments were passed to sample)\nbootsamples - (fit_pl!) the bootstrap samples\nbootestimates - (fit_pl!) the bootstrap parameter estimates\nconvergence - either a Boolean indicating optimization convergence ( for fit_ml!), or a vector of such values for the optimizations done to estimate bootstrap replicates.\n\nThe empty constructor ALfit() will initialize an object with all fields empty, so the needed fields can be filled afterwards.\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.AbstractAutologisticModel",
    "page": "Reference",
    "title": "Autologistic.AbstractAutologisticModel",
    "category": "type",
    "text": "AbstractAutologisticModel\n\nAbstract type representing autologistic models.  This type has methods defined for most operations one will want to perform, so that concrete subtypes should not have to define too many methods unless more specialized and efficient algorithms for the specific subtype.\n\nAll concrete subtypes should have the following fields:\n\nresponses::Array{Bool,2} – The binary observations. Rows are for nodes in the    graph, and columns are for independent (vector) observations.  It is a 2D array even if    there is only one observation.\nunary<:AbstractUnaryParameter – Specifies the unary part of the model.\npairwise<:AbstractPairwiseParameter  – Specifies the pairwise part of the model    (including the graph).\ncentering<:CenteringKinds – Specifies the form of centering used, if any.\ncoding::Tuple{T,T} where T<:Real – Gives the numeric coding of the responses.\nlabels::Tuple{String,String} – Provides names for the high and low states.\ncoordinates<:SpatialCoordinates – Provides 2D or 3D coordinates for each vertex in    the graph.\n\nThe following functions are defined for the abstract type, and are considered part of the  type\'s interface (in this list, M is of type inheriting from AbstractAutologisticModel).\n\ngetparameters(M) and setparameters!(M, newpars::Vector{Float64})\ngetunaryparameters(M) and setunaryparameters!(M, newpars::Vector{Float64})\ngetpairwiseparameters(M) and setpairwiseparameters!(M, newpars::Vector{Float64})\nmakecoded(M, Y)\ncenteringterms(M, kind::Union{Nothing,CenteringKinds})\npseudolikelihood(M)\nnegpotential(M)\nfullPMF(M; indices, force::Bool)\nmarginalprobabilities(M; indices, force::Bool)\nconditionalprobabilities(M; vertices, indices)\nsample(M, k::Int, method::SamplingMethods, indices::Int, average::Bool, start,    burnin::Int, verbose::Bool)\n\nThe sample() function is a wrapper for a variety of random sampling algorithms enumerated in SamplingMethods.\n\nExamples\n\njulia> M = ALsimple(Graph(4,4));\njulia> typeof(M)\nALsimple{CenteringKinds,Int64,Nothing}\njulia> isa(M, AbstractAutologisticModel)\ntrue\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.AbstractPairwiseParameter",
    "page": "Reference",
    "title": "Autologistic.AbstractPairwiseParameter",
    "category": "type",
    "text": "AbstractPairwiseParameter\n\nAbstract type representing the pairwise part of an autologistic regression model.\n\nAll concrete subtypes should have the following fields:\n\nG::SimpleGraph{Int} – The graph for the model.\ncount::Int  – The number of observations.\n\nIn addition to getindex() and setindex!(), any concrete subtype  P<:AbstractPairwiseParameter should also have the following methods defined:\n\ngetparameters(P), returning a Vector{Float64}\nsetparameters!(P, newpar::Vector{Float64}) for setting parameter values.\n\nNote that indexing is performance-critical and should be implemented carefully in  subtypes.  \n\nThe intention is that each subtype should implement a different way of parameterizing the association matrix. The way parameters are stored and values computed is up to the subtypes. \n\nThis type inherits from AbstractArray{Float64, 3}.  The third index is to allow for  multiple observations. P[:,:,r] should return the association matrix of the rth observation in an appropriate subtype of AbstractMatrix.  It is not intended that the third  index will be used for range or vector indexing like P[:,:,1:5] (though this may work  due to AbstractArray fallbacks). \n\nExamples\n\njulia> M = ALsimple(Graph(4,4));\njulia> typeof(M.pairwise)\nSimplePairwise\njulia> isa(M.pairwise, AbstractPairwiseParameter)\ntrue\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.CenteringKinds",
    "page": "Reference",
    "title": "Autologistic.CenteringKinds",
    "category": "type",
    "text": "CenteringKinds\n\nAn enumeration to facilitate choosing a form of centering for the model.  Available choices are: \n\nnone: no centering (centering adjustment equals zero).\nexpectation: the centering adjustment is the expected value of the response under the   assumption of independence (this is what has been used in the \"centered autologistic    model\").\nonehalf: a constant value of centering adjustment equal to 0.5.\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.FullPairwise",
    "page": "Reference",
    "title": "Autologistic.FullPairwise",
    "category": "type",
    "text": "FullPairwise\n\nA type representing an association matrix with a \"saturated\" parametrization–one parameter  for each edge in the graph.\n\nIn this type, the association matrix for each observation is a symmetric matrix with the  same pattern of nonzeros as the graph\'s adjacency matrix, but with arbitrary values in those locations. The package convention is to provide parameters as a vector of Float64.  So  getparameters and setparameters! use a vector of ne(G) values that correspond to the  nonzero locations in the upper triangle of the adjacency matrix, in the same (lexicographic) order as edges(G).\n\nThe association matrix is stored as a SparseMatrixCSC{Float64,Int64} in the field Λ.\n\nAs with SimplePairwise, the association matrix can not be different for different observations.  So while size returns a 3-dimensional result, the third index is ignored when accessing the array\'s elements.\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.SamplingMethods",
    "page": "Reference",
    "title": "Autologistic.SamplingMethods",
    "category": "type",
    "text": "SamplingMethods\n\nAn enumeration to facilitate choosing a method for sampling. Available choices are:\n\nGibbs  TODO\nperfect_bounding_chain  TODO\nperfect_reuse_samples  TODO \nperfect_reuse_seeds  TODO\nperfect_read_once  TODO \n\n\n\n\n\n"
},

{
    "location": "api/#Types-and-Constructors-1",
    "page": "Reference",
    "title": "Types and Constructors",
    "category": "section",
    "text": "Modules = [Autologistic]\r\nOrder   = [:type]"
},

{
    "location": "api/#Autologistic.fullPMF-Tuple{AbstractAutologisticModel}",
    "page": "Reference",
    "title": "Autologistic.fullPMF",
    "category": "method",
    "text": "fullPMF(M::AbstractAutologisticModel; indices=1:size(M.unary,2), force::Bool=false)\n\nCompute the PMF of an AbstractAutologisticModel, and return a NamedTuple (:table, :partition).\n\nFor an AutologisticModel with n variables and m observations, :table is a 2^n(n+1)m  array of Float64. Each page of the 3D array holds a probability table for an observation.   Each row of the table holds a specific configuration of the responses, with the  corresponding probability in the last column.  In the m=1 case,  :table is a 2D array.\n\nOutput :partition is a vector of normalizing constant (a.k.a. partition function) values. In the m=1 case, it is a scalar Float64.\n\nArguments\n\nM: an autologistic model.\nindices: indices of specific observations from which to obtain the output. By  default, all observations are used.\nforce: calling the function with n20 will throw an error unless  force=true. \n\nExamples\n\njulia> M = ALRsimple(Graph(3,0),ones(3,1));\njulia> pmf = fullPMF(M);\njulia> pmf.table\n8×4 Array{Float64,2}:\n -1.0  -1.0  -1.0  0.125\n -1.0  -1.0   1.0  0.125\n -1.0   1.0  -1.0  0.125\n -1.0   1.0   1.0  0.125\n  1.0  -1.0  -1.0  0.125\n  1.0  -1.0   1.0  0.125\n  1.0   1.0  -1.0  0.125\n  1.0   1.0   1.0  0.125\njulia> pmf.partition\n 8.0\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.sample",
    "page": "Reference",
    "title": "Autologistic.sample",
    "category": "function",
    "text": "sample(\n    M::AbstractAutologisticModel, \n    k::Int = 1;\n    method::SamplingMethods = Gibbs,\n    indices = 1:size(M.unary,2), \n    average::Bool = false, \n    start = nothing, \n    burnin::Int = 0,\n    verbose::Bool = false\n)\n\nDraws k random samples from autologistic model M, and either returns the samples  themselves, or the estimated probabilities of observing the \"high\" level at each vertex.\n\nIf the model has more than one observation, then k samples are drawn for each observation\'s parameter settings. To restrict the samples to a subset of observations, use argument indices. \n\nFor a model M with n vertices in its graph:\n\nWhen average=false, the return value is n × length(indices) × k, with singleton   dimensions dropped. \nWhen average=true, the return value is n  × length(indices), with singleton   dimensions dropped.\n\nKeyword Arguments\n\nmethod is a member of the enum SamplingMethods, specifying which sampling method will be used.  The default uses Gibbs sampling.  Where feasible, it is recommended  to use one of the perfect sampling alternatives. See SamplingMethods for more.\n\nindices gives the indices of the observation to use for sampling. The default is all indices, in which case each sample is of the same size as M.responses. \n\naverage controls the form of the output. When average=true, the return value is the  proportion of \"high\" samples at each vertex. (Note that this is not actually the arithmetic average of the samples, unless the coding is (0,1). Rather, it is an estimate of  the probability of getting a \"high\" outcome.)  When average=false, the full set of samples is returned. \n\nstart allows a starting configuration of the random variables to be provided. Only used if method=Gibbs. Any vector with two unique values can be used as start. By default a random configuration is used.\n\nburnin specifies the number of initial samples to discard from the results.  Only used if method=Gibbs.\n\nskip specifies how many samples to throw away between returned samples.  Only used  if method=Gibbs. \n\nverbose controls output to the console.  If true, intermediate information about  sampling progress is printed to the console. Otherwise no output is shown.\n\nExamples\n\njulia> M = ALsimple(Graph(4,4));\njulia> M.coding = (-2,3);\njulia> r = sample(M,10);\njulia> size(r)\n(4, 10)\njulia> sort(unique(r))\n2-element Array{Float64,1}:\n -2.0\n  3.0\n\n\n\n\n\n"
},

{
    "location": "api/#Methods-1",
    "page": "Reference",
    "title": "Methods",
    "category": "section",
    "text": "Modules = [Autologistic]\r\nOrder   = [:function]"
},

]}
