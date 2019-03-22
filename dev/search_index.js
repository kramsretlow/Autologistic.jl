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
    "text": "Typical usage of the package will involve the following three steps:1. Create a model object.All particular AL/ALR models are instances of subtypes of AbstractAutologisticModel.  Each subtype is defined by a particular choice for the parametrization of the unary and pairwise parts.  At present the options are:ALfull: A model with type FullUnary as the unary part, and type FullPairwise as the pairwise part (parameters α Λ).\nALsimple: A model with type FullUnary as the unary part, and type SimplePairwise as the pairwise part (parameters α λ).\nALRsimple: A model with type LinPredUnary as the unary part, and type SimplePairwise as the pairwise part (parameters β λ).The first two types above are mostly for research or exploration purposes.  Most users doing data analysis will use the ALRsimple model.  Each of the above types have various constructors defined.  For example, ALRsimple(G, X) will create an ALRsimple model with graph G and predictor matrix X.  Type, e.g., ?ALRsimple at the REPL to see the constructors #### <== TODO ####.The package is designed to be extensible if other parametrizations of the unary or pairwise parts are desired.  For example, it is planned eventually to add a new pairwise type that will allow the level of association to vary across the grpah.  When such a type appears, additional ALR model types will be created.Any of the above model types can be used with any of the supported forms of centering, and with any desired coding. This facilitates comparison of different model variants.2. Set parameters.Depending on the constructor used, the model just initialized will have either default  parameter values or user-specified parameter values.  Usually it will be desired to choose some appropriate values from data.fit_ml! uses maximum likelihood to estimate the parameters.  It is only useful for cases where the number of vertices in the graph is small.\nfit_pl! uses pseudolikelihood to estimate the parameters.\nsetparameters!, setunaryparameters!, and setpairwiseparameters! can be used to set the parameters of the model directly.\ngetparameters, getunaryparameters, and getpairwiseparameterscan be used to retrieve the parameter values.Changing the parameters directly, through the fields of the model object, is discouraged.  It is preferable for safety to use the above get and set functions.3. Inference and exploration.After parameter estimation, one typically wants to use the fitted model to answer inference questions, make plots, and the like.For small-graph cases:fit_ml! returns p-values and 95% confidence intervals that can be used directly.\nfullPMF, conditionalprobabilities, marginalprobabilities can be used to get desired quantities from the fitted distribution.\nsample can be used to draw random samples from the fitted distribution.For large-graph cases:If using fit_pl!, ##### TODO #####\nSampling can be used to estimate desired quantities like marginal probabilities.  The sample function implements Gibbs sampling as well as several perfect sampling algorithms.Plotting can be done using standard Julia capabilities.  The Examples section shows how to make a few relevant plots."
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
    "text": "These examples demonstrate most of the functionality of the package, its typical usage, and how to make some plots you might want to use.The examples:An Ising Model shows how to use the package to explore the autologistic probability distribution, without concern about covariates or parameter estimation.\nClustered Binary Data (Small n) shows how to use the package for regression analysis when the graph is small enough to permit computation of the normalizing constant. In this case standard maximum likelihood methods of inference can be used.\nSpatial Binary Regression shows how to use the package for autologistic regression analysis for larger, spatially-referenced graphs. In this case pseudolikelihood is used for estimation, and a (possibly parallelized) parametric bootstrap is used for inference."
},

{
    "location": "Examples/#An-Ising-Model-1",
    "page": "Examples",
    "title": "An Ising Model",
    "category": "section",
    "text": "The term \"Ising model\" is usually used to refer to a Markov random field of dichotomous random variables on a regular lattice.  The graph is such that each variable shares an edge only with its nearest neighbors in each dimension.  It\'s a traditional model for magnetic spins, where the coding (-11) is usually used. There\'s one parameter per vertex (a \"local magnetic field\") that increases or decreases the chance of getting a +1 state at that vertex; and there\'s a single pairwise parameter that controls the strength of interaction between neighbor states.In our terminology it\'s just an autologistic model with the appropriate graph. Specifically, it\'s an ALsimple model: one with FullUnary type unary parameter, and SimplePairwise type pairwise parameter.We can create such a model once we have the graph.  For example, let\'s create two 30-by-30 lattices: one without any special handling of the boundary, and one with periodic boundary conditions. This can be done with LightGraphs.jl\'s Grid function.n = 30  # numer of rows and columns\r\nusing LightGraphs\r\nG1 = Grid([n, n], periodic=false)\r\nG2 = Grid([n, n], periodic=true)\r\nnothing # hideNow create an AL model for each case. Initialize the unary parameters to Gaussian white noise. By default the pairwise parameter is set to zero, which implies independence of the variables. Use the same parameters for the two models, so the only difference betwen them is the graph.using Random, Autologistic\r\nRandom.seed!(8888)\r\nα = randn(n^2)\r\nM1 = ALsimple(G1, α)\r\nM2 = ALsimple(G2, α)\r\nnothing #hideTyping M2 at the REPL shows information about the model.  It\'s an ALsimple type with one observation of length 900.M2The conditionalprobabilities function returns the probablity of observing a +1 state at each vertex, conditional on the vertex\'s neighbor values. These can be visualized as an image, using a heatmap (from Plots.jl):using Plots\r\ncondprobs = conditionalprobabilities(M1)\r\nhm = heatmap(reshape(condprobs, n, n), c=:grays, aspect_ratio=1,\r\n             title=\"probability of +1 under independence\")\r\nplot(hm)Since the association parameter is zero, there are no neighborhood effects.  The above conditional probabilities are equal to the marginal probabilities.Next, set the association parameters to 0.75, a fairly strong association level, to introduce a neighbor effect.setpairwiseparameters!(M1, [0.75])\r\nsetpairwiseparameters!(M2, [0.75])\r\nnothing # hideA quick way to see the effect of this parameter is to observe random samples from the models. The sample function can be used to do this. For this example, use perfect sampling using a bounding chain algorithm.s1 = sample(M1, method=perfect_bounding_chain)\r\ns2 = sample(M2, method=perfect_bounding_chain)\r\nnothing #hideOther options are available for sampling.  The enumeration SamplingMethods lists them. The samples we have just drawn can also be visualized using heatmap:pl1 = heatmap(reshape(s1, n, n), c=:grays, colorbar=false, title=\"regular boundary\");\r\npl2 = heatmap(reshape(s2, n, n), c=:grays, colorbar=false, title=\"periodic boundary\");\r\nplot(pl1, pl2, size=(800,400), aspect_ratio=1)In these plots, black indicates the low state, and white the high state.  A lot of local clustering is occurring in the samples due to the neighbor effects.To see the long-run differences between the two models, we can look at the marginal probabilities. They can be estimated by drawing many samples and averaging them (note that running this code chunk can take a couple of minutes):marg1 = sample(M1, 500, method=perfect_bounding_chain, verbose=true, average=true)\r\nmarg2 = sample(M2, 500, method=perfect_bounding_chain, verbose=true, average=true)\r\npl3 = heatmap(reshape(marg1, n, n), c=:grays,\r\n              colorbar=false, title=\"regular boundary\");\r\npl4 = heatmap(reshape(marg2, n, n), c=:grays,\r\n              colorbar=false, title=\"periodic boundary\");\r\nplot(pl3, pl4, size=(800,400), aspect_ratio=1)\r\nsavefig(\"marginal-probs.png\")The figure marginal-probs.png looks like this:(Image: marginal-probs.png)Although the differences between the two marginal distributions are not striking, the extra edges connecting top/bottom and left/right do have some influence on the probabilities at the periphery of the square.As a final demonstration, perform Gibbs sampling for model M2, starting from a random state.  Display a gif animation of the progress.nframes = 150\r\ngibbs_steps = sample(M2, nframes, method=Gibbs)\r\nanim = @animate for i =  1:nframes\r\n    heatmap(reshape(gibbs_steps[:,i], n, n), c=:grays, colorbar=false, \r\n            aspect_ratio=1, title=\"Gibbs sampling: step $(i)\")\r\nend\r\ngif(anim, \"ising_gif.gif\", fps=10)(Image: ising_gif.gif)"
},

{
    "location": "Examples/#Clustered-Binary-Data-(Small-n)-1",
    "page": "Examples",
    "title": "Clustered Binary Data (Small n)",
    "category": "section",
    "text": "The retinitis pigmentosa data set (obtained from this source) is an opthalmology data set.  The data comes from 444 patients that had both eyes examined.  The data can be loaded with Autologistic.datasets:using Autologistic, DataFrames, LightGraphs\r\ndf = Autologistic.datasets(\"pigmentosa\");\r\nfirst(df, 6)\r\ndescribe(df)The response for each eye is va, an indicator of poor visual acuity (coded 0 = no, 1 = yes in the data set). Seven covariates were also recorded for each eye:aut_dom: autosomal dominant (0=no, 1=yes)\naut_rec: autosomal recessive (0=no, 1=yes)\nsex_link: sex-linked (0=no, 1=yes)\nage: age (years, range 6-80)\nsex: gender (0=female, 1=male)\npsc: posterior subscapsular cataract (0=no, 1=yes)\neye: which eye is it? (0=left, 1=right)The last four factors are relevant clinical observations, and the first three are genetic factors. The data set also includes an ID column with an ID number specific to each patient.  Eyes with the same ID come from the same person.The natural unit of analysis is the eye, but pairs of observations from the same patient are \"clustered\" because the occurrence of acuity loss in the left and right eye is probably correlated. We can model each person\'s two va outcomes as two dichotomous random variables with a 2-vertex, 1-edge graph.G = Graph(2,1)Each of the 444 bivariate observations has this graph, and each has its own set of covariates.If we include all seven predictors, plus intercept, in our model, we have 2 variables per observation, 8 predictors, and 444 observations.Before creating the model we need to re-structure the covariates. The data in df has one row per eye, with the variable ID indicating which eyes belong to the same patient.  We need to rearrange the responses (Y) and the predictors (X) into arrays suitable for our autologistic models, namely:Y is 2 times 444 with one observation per column.\nX is 2 times 8 times 444 with one 2 times 8 matrix of predictors for each observation.  The first column of each predictor matrix is an intercept column, and   columns 2 through 8 are for aut_dom, aut_rec, sex_link, age, sex, psc, and eye, respectively.X = Array{Float64,3}(undef, 2, 8, 444);\r\nY = Array{Float64,2}(undef, 2, 444);\r\nfor i in 1:2:888\r\n    patient = Int((i+1)/2)\r\n    X[1,:,patient] = [1 permutedims(Vector(df[i,2:8]))]\r\n    X[2,:,patient] = [1 permutedims(Vector(df[i+1,2:8]))]\r\n    Y[:,patient] = convert(Array, df[i:i+1, 9])\r\nendFor example, patient 100 had responsesY[:,100]Indicating visual acuity loss in the left eye, but not in the right. The predictors for this individual areX[:,:,100]Now we can create our autologistic regression model.model = ALRsimple(G, X, Y=Y)This creates a model with the \"simple pairwise\" structure, using a single association parameter. The default is to use no centering adjustment, and to use coding (-11) for the responses.  This \"symmetric\" version of the model is recommended for a variety of reasons.  Using different coding or centering choices is only recommended if you have a thorough understanding of what you are doing; but if you wish to use different choices, this can easily be done using keyword arguments. For example, ALRsimple(G, X, Y=Y, coding=(0,1), centering=expectation) creates the \"centered autologistic model\" that has appeared in the literature (e.g., here and here).The model has nine parameters (eight regression coefficients plus the association parameter).  All parameters are initialized to zero:getparameters(model)When we call getparameters, the vector returned always has the unary parameters first, with the pairwise parameter(s) appended at the end.Because there are only two vertices in the graph, we can use the full likelihood (fit_ml! function) to do parameter estimation.  This function returns a structure with the estimates as well as standard errors, p-values, and 95% confidence intervals for the  parameter estimates.out = fit_ml!(model)To view the estimation results, use summary:summary(out, parnames = [\"icept\", \"aut_dom\", \"aut_rec\", \"sex_link\", \"age\", \"sex\", \r\n        \"psc\", \"eye\", \"λ\"])From this we see that the association parameter is fairly large (0.818), supporting the idea that the left and right eyes are associated.  It is also highly statistically significant.  Among the covariates, sex_link, age, and psc are all statistically significant."
},

{
    "location": "Examples/#Spatial-Binary-Regression-1",
    "page": "Examples",
    "title": "Spatial Binary Regression",
    "category": "section",
    "text": "ALR models are natural candidates for analysis of spatial binary data, where locations in the same neighborhood are more likely to have the same outcome than sites that are far apart. The hydrocotyle data provide a typical example.  The response in this data set is the presence/absence of a certain plant species in a grid of 2995 regions covering Germany. The data set is included in Autologistic.jl:using Autologistic, DataFrames, LightGraphs\r\ndf = Autologistic.datasets(\"hydrocotyle\")In the data frame, the variables X and Y give the spatial coordinates of each region (in dimensionless integer units), obs gives the presence/absence data (1 = presence), and altitude and temperature are covariates.We will use an ALRsimple model for these data.  The graph can be formed using makespatialgraph:locations = [(df.X[i], df.Y[i]) for i in 1:size(df,1)]\r\ng = makespatialgraph(locations, 1.0)\r\nnothing # hidemakespatialgraph creates the graph by adding edges between any vertices with Euclidean distance smaller than a cutoff distance (Lightgraphs.jl has a euclidean_graph function that does the same thing).  For these data arranged on a grid, a threshold of 1.0 will make a 4-nearest-neighbors lattice. Letting the threshold be sqrt(2) would make an 8-nearest-neighbors lattice.We can visualize the graph, the responses, and the predictors using GraphRecipes.jl (there are several other options for plotting graphs as well).using GraphRecipes, Plots\r\n\r\n# Function to convert a value to a gray shade\r\nmakegray(x, lo, hi) = RGB([(x-lo)/(hi-lo) for i=1:3]...)  \r\n\r\n# Function to plot the graph with node shading determined by v.\r\n# Plot each node as a square and don\'t show the edges.\r\nfunction myplot(v, lohi=nothing)  \r\n    if lohi==nothing\r\n        colors = makegray.(v, minimum(v), maximum(v))\r\n    else\r\n        colors = makegray.(v, lohi[1], lohi[2])\r\n    end\r\n    return graphplot(g.G, x=df.X, y=df.Y, background_color = :lightblue,\r\n                marker = :square, markersize=2, markerstrokewidth=0,\r\n                markercolor = colors, yflip = true, linecolor=nothing)\r\nend\r\n\r\n# Make the plot\r\nplot(myplot(df.obs), myplot(df.altitude), myplot(df.temperature),\r\n     layout=(1,3), size=(800,300), titlefontsize=8,\r\n     title=hcat(\"Species Presence (white = yes)\", \"Altitude (lighter = higher)\",\r\n                \"Temperature (lighter = higher)\"))"
},

{
    "location": "Examples/#Constructing-the-model-1",
    "page": "Examples",
    "title": "Constructing the model",
    "category": "section",
    "text": "We can see that the species primarily is found at low-altitude locations. To model the effect of altitude and temperature on species presence, construct an ALRsimple model.# Autologistic.jl requres predictors to be a matrix of Float64\r\nXmatrix = Array{Float64}([ones(2995) df.altitude df.temperature])\r\n\r\n# Create the model\r\nhydro = ALRsimple(g.G, Xmatrix, Y=df.obs)The model hydro has four parameters: three regression coefficients (interceept, altitude, and temperature) plus an association parameter.  It is a \"symmetric\" autologistic model, because it has a coding symmetric around zero and no centering term."
},

{
    "location": "Examples/#Fitting-the-model-by-pseudolikelihood-1",
    "page": "Examples",
    "title": "Fitting the model by pseudolikelihood",
    "category": "section",
    "text": "With 2995 nodes in the graph, the likelihood is intractable for this case.  Use fit_pl! to do parameter estimation by pseudolikelihood instead.  The fitting function uses the BFGS algorithm via Optim.jl.  Any of Optim\'s general options can be passed to fit_pl! to control the optimization.  We have found that allow_f_increases often aids convergence.  It is used here:fit1 = fit_pl!(hydro, allow_f_increases=true)\r\nparnames = [\"intercept\", \"altitude\", \"temperature\", \"association\"];\r\nsummary(fit1, parnames=parnames)fit_pl! mutates the model object by setting its parameters to the optimal values. It also returns an object, of type ALfit, which holds information about the result. Calling summary(fit1) produces a summary table of the estimates.  For now there are no standard errors.  This will be addressed below.To quickly visualize the quality of the fitted model, we can use sampling to get the marginal probabilities, and to observe specific samples.# Average 500 samples to estimate marginal probability of species presence\r\nmarginal1 = sample(hydro, 500, method=perfect_bounding_chain, average=true)\r\n\r\n# Draw 2 random samples for visualizing generated data.\r\ndraws = sample(hydro, 2, method=perfect_bounding_chain)\r\n\r\n# Plot them\r\nplot(myplot(marginal1, (0,1)), myplot(draws[:,1]), myplot(draws[:,2]),\r\n     layout=(1,3), size=(800,300), titlefontsize=8,\r\n     title=[\"Marginal Probability\" \"Random sample 1\" \"Random Sample 2\"])In the above code, perfect sampling was used to draw samples from the fitted distribution. The marginal plot shows consistency with the observed data, and the two generated data sets show a level of spatial clustering similar to the observed data."
},

{
    "location": "Examples/#Error-estimation-1:-bootstrap-after-the-fact-1",
    "page": "Examples",
    "title": "Error estimation 1: bootstrap after the fact",
    "category": "section",
    "text": "A parametric bootstrap can be used to get an estimate of the precision of the estimates returned by fit_pl!.  The function oneboot has been included in the package to facilitate this.  Each call of oneboot draws a random sample from the fitted distribution, then re-fits the model using this sample as the responses. It returns a named tuple giving the sample, the parameter estimates, and a convergence flag.  Any extra keyword arguments are passed on to sample or optimize as appropriate to control the process.# Do one bootstrap replication for demonstration purposes.\r\noneboot(hydro, allow_f_increases=true, method=perfect_bounding_chain)An array of the tuples produced by oneboot can be fed to addboot! to update the fitting summary with precision estimates:nboot = 2000\r\nboots = [oneboot(hydro, allow_f_increases=true, method=perfect_bounding_chain) for i=1:nboot]\r\naddboot!(fit1, boots)At the time of writing, this took about 5.7 minutes on the author\'s workstation. After adding the bootstrap information, the fitting results look like this:julia> summary(fit1,parnames=parnames)\r\nname          est       se       95% CI\r\nintercept     -0.192    0.319     (-0.858, 0.4)\r\naltitude      -0.0573   0.015    (-0.0887, -0.0296)\r\ntemperature    0.0498   0.0361   (-0.0163, 0.126)\r\nassociation    0.361    0.018      (0.326, 0.397)Confidence intervals for altitude and the association parameter both exclude zero, so we conclude that they are statistically significant."
},

{
    "location": "Examples/#Error-estimation-2:-(parallel)-bootstrap-when-fitting-1",
    "page": "Examples",
    "title": "Error estimation 2: (parallel) bootstrap when fitting",
    "category": "section",
    "text": "Alternatively, the bootstrap inference procedure can be done at the same time as fitting by providing the keyword argument nboot (which specifies the number of bootstrap samples to generate) when calling fit_pl!. If you do this, and you have more than one worker process available, then the bootstrap will be done in parallel across the workers (using an @distributed for loop).  This makes it easy to achieve speed gains from parallelism on multicore workstations.using Distributed                  # needed for parallel computing\r\naddprocs(6)                        # create 6 worker processes\r\n@everywhere using Autologistic     # workers need the package loaded\r\nfit2 = fit_pl!(hydro, nboot=2000,\r\n               allow_f_increases=true, method=perfect_bounding_chain)In this case the 2000 bootstrap replications took about 1.1 minutes on the same 6-core workstation. The output object fit2 already includes the confidence intervals:julia> summary(fit2, parnames=parnames)\r\nname          est       se       95% CI\r\nintercept     -0.192    0.33        (-0.9, 0.407)\r\naltitude      -0.0573   0.0157   (-0.0897, -0.0297)\r\ntemperature    0.0498   0.0372   (-0.0169, 0.13)\r\nassociation    0.361    0.0179     (0.327, 0.396)For parallel computing of the bootstrap in other settings (eg. on a cluster), it should be fairly simple implement in a script, using the oneboot/addboot! approach of the previous section."
},

{
    "location": "Examples/#Comparison-to-logistic-regression-1",
    "page": "Examples",
    "title": "Comparison to logistic regression",
    "category": "section",
    "text": "If we ignore spatial association, and just fit the model with ordinary logistic regression, we get the following result:using GLM\r\nLR = glm(@formula(obs ~ altitude + temperature), df, Bernoulli(), LogitLink());\r\ncoef(LR)The logistic regression coefficients are not directly comparable to the ALR coefficients, because the ALR model uses coding (-1 1).  If we want to compare the two models, we can transform the symmetric model to use the (0 1) coding.note: The symmetric ALR model with (0,1) coding\nThe symmetric ALR model with (-1 1) coding is equivalent to a model with (01) coding and a constant centering adjustment of 0.5.  If the original model has coefficients (β λ), the transformed model has coefficients (2β 4λ).To compare the symmetric ALR model to a logistic regression model, either(recommended) Fit the (-11) ALRsimple model and transform the parameters, or\nFit an ALRsimple model with coding=(0,1) and centering=onehalf.Using option 1 with model hydro, we havetransformed_pars = [2*getunaryparameters(hydro); 4*getpairwiseparameters(hydro)]We see that the association parameter is large (1.45), but the regression parameters are small compared to the logistic regression model.  This is typical: ignoring spatial association tends to result in overestimation of the regression effects.To see that option 2 is also valid, we can fit the transformed model directly:same_as_hydro = ALRsimple(g.G, Xmatrix, Y=df.obs, coding=(0,1), centering=onehalf)\r\nfit3 = fit_pl!(same_as_hydro, allow_f_increases=true)\r\nfit3.estimateWe see that the parameter estimates from same_as_hydro are equal to the hydro estimates after transformation."
},

{
    "location": "Examples/#Comparison-to-the-centered-model-1",
    "page": "Examples",
    "title": "Comparison to the centered model",
    "category": "section",
    "text": "The centered autologistic model can be easily constructed for comparison with the symmetric one.  We can start with a copy of the symmetric model we have already created.The pseudolikelihood function for the centered model is not convex.  Three different local optima were found.  For this demonstration we are using the start argument to let optimization start from a point close to the best minimum found.centered_hydro = deepcopy(hydro)\r\ncentered_hydro.coding = (0,1)\r\ncentered_hydro.centering = expectation\r\nfit4 = fit_pl!(centered_hydro, nboot=2000, start=[-1.7, -0.17, 0.0, 1.5],\r\n               allow_f_increases=true, method=perfect_bounding_chain)julia> summary(fit4, parnames=parnames)\r\nname          est       se       95% CI\r\nintercept     -2.29     1.07       (-4.6, -0.345)\r\naltitude      -0.16     0.0429   (-0.258, -0.088)\r\ntemperature    0.0634   0.115    (-0.138, 0.32)\r\nassociation    1.51     0.0505     (1.42, 1.61)\r\n\r\njulia> round.([fit3.estimate fit4.estimate], digits=3)\r\n4×2 Array{Float64,2}:\r\n -0.383  -2.29\r\n -0.115  -0.16\r\n  0.1     0.063\r\n  1.446   1.506The main difference between the symmetric ALR model and the centered one is the intercept, which changes from -0.383 to -2.29 when changing to the centered model.  This is not a small difference.  To see this, compare what the two models predict in the absence of spatial association.# Change models to have association parameters equal to zero\r\n# Remember parameters are always Array{Float64,1}.\r\nsetpairwiseparameters!(centered_hydro, [0.0])\r\nsetpairwiseparameters!(hydro, [0.0])\r\n\r\n# Sample to estimate marginal probabilities\r\ncentered_marg = sample(centered_hydro, 500, method=perfect_bounding_chain, average=true)\r\nsymmetric_marg = sample(hydro, 500, method=perfect_bounding_chain, average=true)\r\n\r\n# Plot to compare\r\nplot(myplot(centered_marg, (0,1)), myplot(symmetric_marg, (0,1)),\r\n     layout=(1,2), size=(500,300), titlefontsize=8,\r\n     title=[\"Centered Model\" \"Symmetric Model\"])(Image: noassociation.png)If we remove the spatial association term, the centered model predicts a very low probability of seeing the plant anywhere–including in locations with low elevation, where the plant is plentiful in reality. This is a manifestation of a problem with the centered model, where parameter interpretability is lost when association becomes strong."
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
    "text": "CurrentModule = Autologistic\r\nDocTestSetup = :(using Autologistic, LightGraphs)"
},

{
    "location": "api/#Index-1",
    "page": "Reference",
    "title": "Index",
    "category": "section",
    "text": ""
},

{
    "location": "api/#Autologistic.ALRsimple",
    "page": "Reference",
    "title": "Autologistic.ALRsimple",
    "category": "type",
    "text": "ALRsimple\n\nAn autologistic regression model with \"simple smoothing\":  the unary parameter is of type LinPredUnary, and the pairwise parameter is of type SimplePairwise.\n\nConstructors\n\nALRsimple(unary::LinPredUnary, pairwise::SimplePairwise;\n    Y::Union{Nothing,<:VecOrMat} = nothing,\n    centering::CenteringKinds = none, \n    coding::Tuple{Real,Real} = (-1,1),\n    labels::Tuple{String,String} = (\"low\",\"high\"), \n    coordinates::SpatialCoordinates = [(0.0,0.0) for i=1:size(unary,1)]\n)\nALRsimple(graph::SimpleGraph{Int}, X::Float2D3D; \n    Y::VecOrMat = Array{Bool,2}(undef,nv(graph),size(X,3)),\n    β::Vector{Float64} = zeros(size(X,2)),\n    λ::Float64 = 0.0, \n    centering::CenteringKinds = none, \n    coding::Tuple{Real,Real} = (-1,1),\n    labels::Tuple{String,String} = (\"low\",\"high\"),\n    coordinates::SpatialCoordinates = [(0.0,0.0) for i=1:nv(graph)]\n)\n\nArguments\n\nY: the array of dichotomous responses.  Any array with 2 unique values will work. If the array has only one unique value, it must equal one of the coding values. The  supplied object will be internally represented as a Boolean array.\nβ: the regression coefficients.\nλ: the association parameter.\ncentering: controls what form of centering to use.\ncoding: determines the numeric coding of the dichotomous responses. \nlabels: a 2-tuple of text labels describing the meaning of Y. The first element is the label corresponding to the lower coding value.\ncoordinates: an array of 2- or 3-tuples giving spatial coordinates of each vertex in the graph. \n\nExamples\n\njulia> using LightGraphs\njulia> X = rand(10,3);            #-predictors\njulia> Y = rand([-2, 3], 10);     #-responses\njulia> g = Graph(10,20);          #-graph\njulia> u = LinPredUnary(X);\njulia> p = SimplePairwise(g);\njulia> model1 = ALRsimple(u, p, Y=Y);\njulia> model2 = ALRsimple(g, X, Y=Y);\njulia> all([getfield(model1, fn)==getfield(model2, fn) for fn in fieldnames(ALRsimple)])\ntrue\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.ALfit",
    "page": "Reference",
    "title": "Autologistic.ALfit",
    "category": "type",
    "text": "ALfit\n\nA type to hold estimation output for autologistic models.  Fitting functions return an  object of this type.\n\nDepending on the fitting method, some fields might not be set.  Fields that are not used are set to nothing or to zero-dimensional arrays.  The fields are:\n\nestimate: A vector of parameter estimates.\nse: A vector of standard errors for the estimates.\npvalues: A vector of p-values for testing the null hypothesis that the parameters equal zero (one-at-a time hypothesis tests).\nCIs: A vector of 95% confidence intervals for the parameters (a vector of 2-tuples).\noptim: the output of the call to optimize used to get the estimates.\nHinv (used by fit_ml!): The inverse of the Hessian matrix of the objective function,  evaluated at the estimate.\nnboot (fit_pl!): number of bootstrap samples to use for error estimation.\nkwargs (fit_pl!): holds extra keyword arguments passed in the call to the fitting function.\nbootsamples (fit_pl!): the bootstrap samples.\nbootestimates (fit_pl!): the bootstrap parameter estimates.\nconvergence: either a Boolean indicating optimization convergence ( for fit_ml!), or a vector of such values for the optimizations done to estimate bootstrap replicates.\n\nThe empty constructor ALfit() will initialize an object with all fields empty, so the needed fields can be filled afterwards.\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.ALsimple",
    "page": "Reference",
    "title": "Autologistic.ALsimple",
    "category": "type",
    "text": "ALsimple\n\nAn autologistic model with a FullUnary unary parameter type and a SimplePairwise pairwise parameter type.   This model has the maximum number of unary parameters  (one parameter per variable per observation), and a single association parameter.\n\nConstructors\n\nALsimple(unary::FullUnary, pairwise::SimplePairwise; \n    Y::Union{Nothing,<:VecOrMat} = nothing, \n    centering::CenteringKinds = none, \n    coding::Tuple{Real,Real} = (-1,1),\n    labels::Tuple{String,String} = (\"low\",\"high\"), \n    coordinates::SpatialCoordinates = [(0.0,0.0) for i=1:size(unary,1)]\n)\nALsimple(graph::SimpleGraph{Int}, alpha::Float1D2D; \n    Y::VecOrMat = Array{Bool,2}(undef,nv(graph),size(alpha,2)), \n    λ::Float64 = 0.0, \n    centering::CenteringKinds = none, \n    coding::Tuple{Real,Real} = (-1,1),\n    labels::Tuple{String,String} = (\"low\",\"high\"),\n    coordinates::SpatialCoordinates = [(0.0,0.0) for i=1:nv(graph)]\n)\nALsimple(graph::SimpleGraph{Int}, count::Int = 1; \n    Y::VecOrMat = Array{Bool,2}(undef,nv(graph),size(alpha,2)), \n    λ::Float64=0.0, \n    centering::CenteringKinds=none, \n    coding::Tuple{Real,Real}=(-1,1),\n    labels::Tuple{String,String}=(\"low\",\"high\"),\n    coordinates::SpatialCoordinates=[(0.0,0.0) for i=1:nv(graph)]\n)\n\nArguments\n\nY: the array of dichotomous responses.  Any array with 2 unique values will work. If the array has only one unique value, it must equal one of th coding values. The  supplied object will be internally represented as a Boolean array.\nλ: the association parameter.\ncentering: controls what form of centering to use.\ncoding: determines the numeric coding of the dichotomous responses. \nlabels: a 2-tuple of text labels describing the meaning of Y. The first element is the label corresponding to the lower coding value.\ncoordinates: an array of 2- or 3-tuples giving spatial coordinates of each vertex in the graph. \n\nExamples\n\njulia> alpha = zeros(10, 4);       #-unary parameter values\njulia> Y = rand([0, 1], 10, 4);    #-responses\njulia> g = Graph(10, 20);          #-graph\njulia> u = FullUnary(alpha);\njulia> p = SimplePairwise(g, 4);\njulia> model1 = ALsimple(u, p, Y=Y);\njulia> model2 = ALsimple(g, alpha, Y=Y);\njulia> model3 = ALsimple(g, 4, Y=Y);\njulia> all([getfield(model1, fn)==getfield(model2, fn)==getfield(model3, fn)\n            for fn in fieldnames(ALsimple)])\ntrue\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.AbstractAutologisticModel",
    "page": "Reference",
    "title": "Autologistic.AbstractAutologisticModel",
    "category": "type",
    "text": "AbstractAutologisticModel\n\nAbstract type representing autologistic models. All concrete subtypes should have the following fields:\n\nresponses::Array{Bool,2} – The binary observations. Rows are for nodes in the    graph, and columns are for independent (vector) observations.  It is a 2D array even if    there is only one observation.\nunary<:AbstractUnaryParameter – Specifies the unary part of the model.\npairwise<:AbstractPairwiseParameter  – Specifies the pairwise part of the model    (including the graph).\ncentering<:CenteringKinds – Specifies the form of centering used, if any.\ncoding::Tuple{T,T} where T<:Real – Gives the numeric coding of the responses.\nlabels::Tuple{String,String} – Provides names for the high and low states.\ncoordinates<:SpatialCoordinates – Provides 2D or 3D coordinates for each vertex in    the graph.\n\nThis type has the following functions defined, considered part of the type\'s interface. They cover most operations one will want to perform.  Concrete subtypes should not have to define custom overrides unless more specialized or efficient algorithms exist for the  subtype.\n\ngetparameters and setparameters!\ngetunaryparameters and setunaryparameters!\ngetpairwiseparameters and setpairwiseparameters!\ncenteringterms\nnegpotential, pseudolikelihood, and loglikelihood\nfullPMF, marginalprobabilities, and conditionalprobabilities\nfit_pl! and fit_ml!\nsample and oneboot\nshowfields\n\nExamples\n\njulia> M = ALsimple(Graph(4,4));\njulia> typeof(M)\nALsimple{CenteringKinds,Int64,Nothing}\njulia> isa(M, AbstractAutologisticModel)\ntrue\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.AbstractPairwiseParameter",
    "page": "Reference",
    "title": "Autologistic.AbstractPairwiseParameter",
    "category": "type",
    "text": "AbstractPairwiseParameter\n\nAbstract type representing the pairwise part of an autologistic regression model.\n\nAll concrete subtypes should have the following fields:\n\nG::SimpleGraph{Int} – The graph for the model.\ncount::Int  – The number of observations.\n\nIn addition to getindex() and setindex!(), any concrete subtype  P<:AbstractPairwiseParameter should also have the following methods defined:\n\ngetparameters(P), returning a Vector{Float64}\nsetparameters!(P, newpar::Vector{Float64}) for setting parameter values.\n\nNote that indexing is performance-critical and should be implemented carefully in  subtypes.  \n\nThe intention is that each subtype should implement a different way of parameterizing the association matrix. The way parameters are stored and values computed is up to the subtypes. \n\nThis type inherits from AbstractArray{Float64, 3}.  The third index is to allow for  multiple observations. P[:,:,r] should return the association matrix of the rth observation in an appropriate subtype of AbstractMatrix.  It is not intended that the third  index will be used for range or vector indexing like P[:,:,1:5] (though this may work  due to AbstractArray fallbacks). \n\nExamples\n\njulia> M = ALsimple(Graph(4,4));\njulia> typeof(M.pairwise)\nSimplePairwise\njulia> isa(M.pairwise, AbstractPairwiseParameter)\ntrue\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.AbstractUnaryParameter",
    "page": "Reference",
    "title": "Autologistic.AbstractUnaryParameter",
    "category": "type",
    "text": "AbstractUnaryParameter\n\nAbstract type representing the unary part of an autologistic regression model.\n\nThis type inherits from AbstractArray{Float64, 2}. The first dimension is for vertices/variables in the graph, and the second dimension is for observations.  It is two-dimensional even if there is only one observation. \n\nImplementation details are left to concrete subtypes, and will depend on how the unary terms are parametrized.  Note that indexing is performance-critical.\n\nConcrete subtypes should implement getparameters, setparameters!, and showfields.\n\nExamples\n\njulia> M = ALsimple(Graph(4,4));\njulia> typeof(M.unary)\nFullUnary\njulia> isa(M.unary, AbstractUnaryParameter)\ntrue\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.CenteringKinds",
    "page": "Reference",
    "title": "Autologistic.CenteringKinds",
    "category": "type",
    "text": "CenteringKinds\n\nAn enumeration to facilitate choosing a form of centering for the model.  Available choices are: \n\nnone: no centering (centering adjustment equals zero).\nexpectation: the centering adjustment is the expected value of the response under the assumption of independence (this is what has been used in the \"centered autologistic  model\").\nonehalf: a constant value of centering adjustment equal to 0.5 (this produces the \"symmetric autologistic model\" when used with 0,1 coding).\n\nThe default/recommended model has centering of none with (-1, 1) coding.\n\nExamples\n\njulia> CenteringKinds\nEnum CenteringKinds:\nnone = 0\nexpectation = 1\nonehalf = 2\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.FullPairwise",
    "page": "Reference",
    "title": "Autologistic.FullPairwise",
    "category": "type",
    "text": "FullPairwise\n\nA type representing an association matrix with a \"saturated\" parametrization–one parameter  for each edge in the graph.\n\nIn this type, the association matrix for each observation is a symmetric matrix with the  same pattern of nonzeros as the graph\'s adjacency matrix, but with arbitrary values in those locations. The package convention is to provide parameters as a vector of Float64.  So  getparameters and setparameters! use a vector of ne(G) values that correspond to the  nonzero locations in the upper triangle of the adjacency matrix, in the same (lexicographic) order as edges(G).\n\nThe association matrix is stored as a SparseMatrixCSC{Float64,Int64} in the field Λ.\n\nAs with SimplePairwise, the association matrix can not be different for different observations.  So while size returns a 3-dimensional result, the third index is ignored when accessing the array\'s elements.\n\nConstructors\n\nFullPairwise(G::SimpleGraph, count::Int=1)\nFullPairwise(n::Int, count::Int=1)\nFullPairwise(λ::Real, G::SimpleGraph)\nFullPairwise(λ::Real, G::SimpleGraph, count::Int)\nFullPairwise(λ::Vector{Float64}, G::SimpleGraph)\n\nIf provide only a graph, set λ = zeros(nv(G)). If provide only an integer, set λ to zeros and make a totally disconnected graph. If provide a graph and a scalar, convert the scalar to a vector of the right length.\n\nExamples\n\njulia> g = makegrid4(2,2).G;\njulia> λ = [1.0, 2.0, -1.0, -2.0];\njulia> p = FullPairwise(λ, g);\njulia> typeof(p.Λ)\nSparseArrays.SparseMatrixCSC{Float64,Int64}\n\njulia> Matrix(p[:,:])\n4×4 Array{Float64,2}:\n 0.0   1.0   2.0   0.0\n 1.0   0.0   0.0  -1.0\n 2.0   0.0   0.0  -2.0\n 0.0  -1.0  -2.0   0.0\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.FullUnary",
    "page": "Reference",
    "title": "Autologistic.FullUnary",
    "category": "type",
    "text": "FullUnary\n\nThe unary part of an autologistic model, with one parameter per vertex per observation. The type has only a single field, for holding an array of parameters.\n\nConstructors\n\nFullUnary(alpha::Array{Float64,1}) \nFullUnary(n::Int)                     #-initializes parameters to zeros\nFullUnary(n::Int, m::Int)             #-initializes parameters to zeros\n\nExamples\n\njulia> u = FullUnary(5, 3);\njulia> u[:,:]\n5×3 Array{Float64,2}:\n 0.0  0.0  0.0\n 0.0  0.0  0.0\n 0.0  0.0  0.0\n 0.0  0.0  0.0\n 0.0  0.0  0.0\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.LinPredUnary",
    "page": "Reference",
    "title": "Autologistic.LinPredUnary",
    "category": "type",
    "text": "LinPredUnary\n\nThe unary part of an autologistic model, parametrized as a regression linear predictor. Its fields are X, an n-by-p-by-m matrix (n obs, p predictors, m observations), and β, a p-vector of parameters (the same for all observations).\n\nConstructors\n\nLinPredUnary(X::Matrix{Float64}, β::Vector{Float64})\nLinPredUnary(X::Matrix{Float64})\nLinPredUnary(X::Array{Float64, 3})\nLinPredUnary(n::Int,p::Int)\nLinPredUnary(n::Int,p::Int,m::Int)\n\nAny quantities not provided in the constructors are initialized to zeros.\n\nExamples\n\njulia> u = LinPredUnary(ones(5,3,2), [1.0, 2.0, 3.0]);\njulia> u[:,:]\n5×2 Array{Float64,2}:\n 6.0  6.0\n 6.0  6.0\n 6.0  6.0\n 6.0  6.0\n 6.0  6.0\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.SamplingMethods",
    "page": "Reference",
    "title": "Autologistic.SamplingMethods",
    "category": "type",
    "text": "SamplingMethods\n\nAn enumeration to facilitate choosing a method for random sampling from autologistic models. Available choices are:\n\nGibbs:  Gibbs sampling.\nperfect_bounding_chain: Perfect sampling, using a bounding chain algorithm.\nperfect_reuse_samples: Perfect sampling. CFTP implemented by reusing random numbers.\nperfect_reuse_seeds: Perfect sampling. CFTP implemented by reusing RNG seeds.\nperfect_read_once: Perfect sampling. Read-once CFTP implementation.\n\nAll of the perfect sampling methods are implementations of coupling from the past (CFTP). perfect_bounding_chain uses a bounding chain approach that holds even when Λ contains negative elements; the other three options rely on a monotonicity argument that requires Λ to have only positive elements (though they should work similar to Gibbs sampling in that case).\n\nDifferent perfect sampling implementations might work best for different models, and parameter settings exist where perfect sampling coalescence might take a prohibitively long time.  For these reasons, Gibbs sampling is the default in sample.\n\nExamples\n\njulia> SamplingMethods\nEnum SamplingMethods:\nGibbs = 0\nperfect_reuse_samples = 1\nperfect_reuse_seeds = 2\nperfect_read_once = 3\nperfect_bounding_chain = 4\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.SimplePairwise",
    "page": "Reference",
    "title": "Autologistic.SimplePairwise",
    "category": "type",
    "text": "SimplePairwise\n\nPairwise association matrix, parametrized as a scalar parameter times the adjacency matrix.\n\nConstructors\n\nSimplePairwise(G::SimpleGraph, count::Int=1) SimplePairwise(n::Int, count::Int=1) SimplePairwise(λ::Real, G::SimpleGraph) SimplePairwise(λ::Real, G::SimpleGraph, count::Int)\n\nIf provide only a graph, set λ = 0. If provide only an integer, set λ = 0 and make a totally disconnected graph. If provide a graph and a scalar, convert the scalar to a length-1 vector.\n\nUnlike FullPairwise, every observation must have the same association matrix in this case. So while we internally treat it like an n-by-n-by-m matrix, just return a 2D n-by-n matrix to the user. \n\nExamples\n\njulia> g = makegrid4(2,2).G;\njulia> λ = 1.0;\njulia> p = SimplePairwise(λ, g, 4);    #-4 observations\njulia> size(p)\n(4, 4, 4)\n\njulia> Matrix(p[:,:,:])\n4×4 Array{Float64,2}:\n 0.0  1.0  1.0  0.0\n 1.0  0.0  0.0  1.0\n 1.0  0.0  0.0  1.0\n 0.0  1.0  1.0  0.0\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.SpatialCoordinates",
    "page": "Reference",
    "title": "Autologistic.SpatialCoordinates",
    "category": "constant",
    "text": "Type alias for Union{Array{NTuple{2,T},1},Array{NTuple{3,T},1}} where T<:Real \n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.Float1D2D",
    "page": "Reference",
    "title": "Autologistic.Float1D2D",
    "category": "constant",
    "text": "Type alias for Union{Array{Float64,1},Array{Float64,2}} \n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.Float2D3D",
    "page": "Reference",
    "title": "Autologistic.Float2D3D",
    "category": "constant",
    "text": "Type alias for Union{Array{Float64,2},Array{Float64,3}} \n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.VecOrMat",
    "page": "Reference",
    "title": "Autologistic.VecOrMat",
    "category": "constant",
    "text": "Type alias for Union{Array{T,1}, Array{T,2}} where T \n\n\n\n\n\n"
},

{
    "location": "api/#Types-and-Constructors-1",
    "page": "Reference",
    "title": "Types and Constructors",
    "category": "section",
    "text": "Modules = [Autologistic]\r\nOrder   = [:type, :constant]"
},

{
    "location": "api/#Autologistic.addboot!-Tuple{ALfit,Array{Float64,3},Array{Float64,2},Array{Bool,1}}",
    "page": "Reference",
    "title": "Autologistic.addboot!",
    "category": "method",
    "text": "addboot!(fit::ALfit, bootsamples::Array{Float64,3}, \n         bootestimates::Array{Float64,2}, convergence::Vector{Bool})\n\nAdd parametric bootstrap information in arrays bootsamples, bootestimates, and convergence to model fitting information fit.  If fit already contains bootstrap data, the new data is appended to the existing data, and statistics are recomputed.\n\nExamples\n\njulia> using Random;\njulia> Random.seed!(1234);\njulia> G = makegrid4(4,3).G;\njulia> Y=[[fill(-1,4); fill(1,8)] [fill(-1,3); fill(1,9)] [fill(-1,5); fill(1,7)]];\njulia> model = ALRsimple(G, ones(12,1,3), Y=Y);\njulia> fit = fit_pl!(model, start=[-0.4, 1.1]);\njulia> samps = zeros(12,3,10);\njulia> ests = zeros(2,10);\njulia> convs = fill(false, 10);\njulia> for i = 1:10\n           temp = oneboot(model, start=[-0.4, 1.1])\n           samps[:,:,i] = temp.sample\n           ests[:,i] = temp.estimate\n           convs[i] = temp.convergence\n       end\njulia> addboot!(fit, samps, ests, convs)\njulia> summary(fit)\nname          est     se      95% CI\nparameter 1   -0.39   0.442      (-1.09, 0.263)\nparameter 2    1.1    0.279   (-0.00664, 0.84)\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.addboot!-Union{Tuple{T}, Tuple{ALfit,Array{T,1}}} where T<:(NamedTuple{(:sample, :estimate, :convergence),T} where T<:Tuple)",
    "page": "Reference",
    "title": "Autologistic.addboot!",
    "category": "method",
    "text": "addboot!(fit::ALfit, bootresults::Array{T,1}) where \n    T <: NamedTuple{(:sample, :estimate, :convergence)}\n\nAn addboot! method taking bootstrap data as an array of named tuples. Tuples are of the form produced by oneboot.\n\nExamples\n\njulia>     using Random;\njulia>     Random.seed!(1234);\njulia>     G = makegrid4(4,3).G;\njulia>     Y=[[fill(-1,4); fill(1,8)] [fill(-1,3); fill(1,9)] [fill(-1,5); fill(1,7)]];\njulia>     model = ALRsimple(G, ones(12,1,3), Y=Y);\njulia>     fit = fit_pl!(model, start=[-0.4, 1.1]);\njulia>     boots = [oneboot(model, start=[-0.4, 1.1]) for i = 1:10];\njulia>     addboot!(fit, boots)\njulia>     summary(fit)\nname          est     se      95% CI\nparameter 1   -0.39   0.442      (-1.09, 0.263)\nparameter 2    1.1    0.279   (-0.00664, 0.84)\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.centeringterms",
    "page": "Reference",
    "title": "Autologistic.centeringterms",
    "category": "function",
    "text": "centeringterms(M::AbstractAutologisticModel, kind::Union{Nothing,CenteringKinds}=nothing)\n\nReturns an Array{Float64,2} of the same dimension as M.unary, giving the centering adjustments for autologistic model M. centeringterms(M,kind) returns the centering adjustment that would be used if centering were of type kind.\n\nExamples\n\njulia> G = makegrid8(2,2).G;\njulia> X = [ones(4) [-2; -1; 1; 2]];\njulia> M1 = ALRsimple(G, X, β=[-1.0, 2.0]);                 #-No centering (default)\njulia> M2 = ALRsimple(G, X, β=[-1.0, 2.0], centering=expectation);  #-Centered model\njulia> [centeringterms(M1) centeringterms(M2) centeringterms(M1, onehalf)]\n4×3 Array{Float64,2}:\n 0.0  -0.999909  0.5\n 0.0  -0.995055  0.5\n 0.0   0.761594  0.5\n 0.0   0.995055  0.5\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.conditionalprobabilities-Tuple{AbstractAutologisticModel}",
    "page": "Reference",
    "title": "Autologistic.conditionalprobabilities",
    "category": "method",
    "text": "conditionalprobabilities(M::AbstractAutologisticModel; vertices=1:size(M.unary)[1], \n                         indices=1:size(M.unary,2))\n\nCompute the conditional probability that variables in autologistic model M take the high state, given the current values of all of their neighbors. If vertices or indices are provided, the results are only computed for the desired variables & observations.   Otherwise results are computed for all variables and observations.\n\nExamples\n\njulia> Y = [ones(9) zeros(9)];\njulia> G = makegrid4(3,3).G;\njulia> model = ALsimple(G, ones(9,2), Y=Y, λ=0.5);    #-Variables on a 3×3 grid, 2 obs.\njulia> conditionalprobabilities(model, vertices=5)    #-Cond. probs. of center vertex.\n1×2 Array{Float64,2}:\n 0.997527  0.119203\n\njulia> conditionalprobabilities(model, indices=2)     #-Cond probs, 2nd observation.\n9×1 Array{Float64,2}:\n 0.5\n 0.26894142136999516\n 0.5\n 0.26894142136999516\n 0.11920292202211756\n 0.26894142136999516\n 0.5\n 0.26894142136999516\n 0.5\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.fit_ml!-Tuple{AbstractAutologisticModel}",
    "page": "Reference",
    "title": "Autologistic.fit_ml!",
    "category": "method",
    "text": "fit_ml!(M::AbstractAutologisticModel;\n    start=zeros(length(getparameters(M))),\n    verbose::Bool=false,\n    force::Bool=false,\n    kwargs...\n)\n\nFit autologistic model M using maximum likelihood. Will fail for models with more than  20 vertices, unless force=true.  Use fit_pl! for larger models.\n\nArguments\n\nstart: initial value to use for optimization.\nverbose: should progress information be printed to the console?\nforce: set to true to force computation of the likelihood for large models.\nkwargs... extra keyword arguments that are passed on to optimize().\n\nExamples\n\njulia> G = makegrid4(4,3).G;\njulia> model = ALRsimple(G, ones(12,1), Y=[fill(-1,4); fill(1,8)]);\njulia> mle = fit_ml!(model);\njulia> summary(mle)\nname          est      se      p-value   95% CI\nparameter 1   0.0791   0.163   0.628       (-0.241, 0.399)\nparameter 2   0.425    0.218   0.0511    (-0.00208, 0.852)\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.fit_pl!-Tuple{AbstractAutologisticModel}",
    "page": "Reference",
    "title": "Autologistic.fit_pl!",
    "category": "method",
    "text": "fit_pl!(M::AbstractAutologisticModel;\n    start=zeros(length(getparameters(M))), \n    verbose::Bool=false,\n    nboot::Int = 0,\n    kwargs...)\n\nFit autologistic model M using maximum pseudolikelihood. \n\nArguments\n\nstart: initial value to use for optimization.\nverbose: should progress information be printed to the console?\nnboot: number of samples to use for parametric bootstrap error estimation. If nboot=0 (the default), no bootstrap is run.\nkwargs... extra keyword arguments that are passed on to optimize() or sample(), as appropriate.\n\nExamples\n\njulia> Y=[[fill(-1,4); fill(1,8)] [fill(-1,3); fill(1,9)] [fill(-1,5); fill(1,7)]];\njulia> model = ALRsimple(G, ones(12,1,3), Y=Y);\njulia> fit = fit_pl!(model, start=[-0.4, 1.1]);\njulia> summary(fit)\nname          est     se   p-value   95% CI\nparameter 1   -0.39\nparameter 2    1.1\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.fullPMF-Tuple{AbstractAutologisticModel}",
    "page": "Reference",
    "title": "Autologistic.fullPMF",
    "category": "method",
    "text": "fullPMF(M::AbstractAutologisticModel; \n    indices=1:size(M.unary,2), \n    force::Bool=false\n)\n\nCompute the PMF of an AbstractAutologisticModel, and return a NamedTuple (:table, :partition).\n\nFor an AbstractAutologisticModel with n variables and m observations, :table is a 2^n(n+1)m array of Float64. Each page of the 3D array holds a probability table for  an observation.  Each row of the table holds a specific configuration of the responses, with the corresponding probability in the last column.  In the m=1 case,  :table is a 2D  array.\n\nOutput :partition is a vector of normalizing constant (a.k.a. partition function) values. In the m=1 case, it is a scalar Float64.\n\nArguments\n\nindices: indices of specific observations from which to obtain the output. By  default, all observations are used.\nforce: calling the function with n20 will throw an error unless  force=true. \n\nExamples\n\njulia> M = ALRsimple(Graph(3,0),ones(3,1));\njulia> pmf = fullPMF(M);\njulia> pmf.table\n8×4 Array{Float64,2}:\n -1.0  -1.0  -1.0  0.125\n -1.0  -1.0   1.0  0.125\n -1.0   1.0  -1.0  0.125\n -1.0   1.0   1.0  0.125\n  1.0  -1.0  -1.0  0.125\n  1.0  -1.0   1.0  0.125\n  1.0   1.0  -1.0  0.125\n  1.0   1.0   1.0  0.125\njulia> pmf.partition\n 8.0\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.getpairwiseparameters-Tuple{AbstractAutologisticModel}",
    "page": "Reference",
    "title": "Autologistic.getpairwiseparameters",
    "category": "method",
    "text": "getpairwiseparameters(M::AbstractAutologisticModel)\n\nExtracts the pairwise parameters from an autologistic model. Parameters are always returned as an Array{Float64,1}.\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.getparameters-Tuple{AbstractAutologisticModel}",
    "page": "Reference",
    "title": "Autologistic.getparameters",
    "category": "method",
    "text": "getparameters(x)\n\nA generic function for extracting the parameters from an autologistic model, a unary term, or a pairwise term.  Parameters are always returned as an Array{Float64,1}.  If  typeof(x) <: AbstractAutologisticModel, the returned vector is partitioned with the unary parameters first.\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.getunaryparameters-Tuple{AbstractAutologisticModel}",
    "page": "Reference",
    "title": "Autologistic.getunaryparameters",
    "category": "method",
    "text": "getunaryparameters(M::AbstractAutologisticModel)\n\nExtracts the unary parameters from an autologistic model. Parameters are always returned as an Array{Float64,1}.\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.loglikelihood-Tuple{AbstractAutologisticModel}",
    "page": "Reference",
    "title": "Autologistic.loglikelihood",
    "category": "method",
    "text": "loglikelihood(M::AbstractAutologisticModel; \n    force::Bool=false\n)\n\nCompute the natural logarithm of the likelihood for autologistic model M.  This will throw an error for models with more than 20 vertices, unless force=true.\n\nExamples\n\njulia> model = ALRsimple(makegrid4(2,2)[1], ones(4,2,3), centering = expectation,\n                         coding = (0,1), Y = repeat([true, true, false, false],1,3));\njulia> setparameters!(model, [1.0, 1.0, 1.0]);\njulia> loglikelihood(model)\n-11.86986109487605\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.makebool",
    "page": "Reference",
    "title": "Autologistic.makebool",
    "category": "function",
    "text": "makebool(v::VecOrMat, vals=nothing)\n\nMakes a 2D array of Booleans out of a 1- or 2-D input.  The 2nd argument vals optionally can be a 2-tuple (low, high) specifying the two possible values in v (useful for the case where all elements of v take one value or the other).\n\nIf v has more than 2 unique values, throws an error.\nIf v has exactly 2 unique values, use those to set the coding (ignore vals).\nIf v has 1 unique value, use vals to determine if it\'s the high or low value (throw an error if the single value isn\'t in vals).\n\nExamples\n\njulia> makebool([1.0 2.0; 1.0 2.0])\n2×2 Array{Bool,2}:\n false  true\n false  true\n\njulia> makebool([\"yes\", \"no\", \"no\"])\n3×1 Array{Bool,2}:\n  true\n false\n false\n\njulia> [makebool([1, 1, 1], (-1,1)) makebool([1, 1, 1], (1, 2))]\n3×2 Array{Bool,2}:\n true  false\n true  false\n true  false\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.makecoded-Tuple{AbstractAutologisticModel}",
    "page": "Reference",
    "title": "Autologistic.makecoded",
    "category": "method",
    "text": "makecoded(M::AbstractAutologisticModel)\n\nA convenience method for makecoded(M.responses, M.coding).  Use it to retrieve a model\'s responses in coded form.\n\nExamples\n\njulia> M = ALRsimple(Graph(4,3), rand(4,2), Y=[true, false, false, true], coding=(-1,1));\njulia> makecoded(M)\n4×1 Array{Float64,2}:\n  1.0\n -1.0\n -1.0\n  1.0\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.makecoded-Tuple{Union{Array{T,1}, Array{T,2}} where T,Tuple{Real,Real}}",
    "page": "Reference",
    "title": "Autologistic.makecoded",
    "category": "method",
    "text": "makecoded(b::VecOrMat, coding::Tuple{Real,Real})\n\nConvert Boolean responses into coded values.  The first argument is boolean. Returns a 2D array of Float64.  \n\nExamples\n\njulia> makecoded([true, false, false, true], (-1, 1))\n4×1 Array{Float64,2}:\n  1.0\n -1.0\n -1.0\n  1.0\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.makegrid4",
    "page": "Reference",
    "title": "Autologistic.makegrid4",
    "category": "function",
    "text": "makegrid4(r::Int, c::Int, xlim::Tuple{Real,Real}=(0.0,1.0), \n          ylim::Tuple{Real,Real}=(0.0,1.0))\n\nReturns a named tuple (:G, :locs), where :G is a graph, and :locs is an array of  numeric tuples.  Vertices of :G are laid out in a rectangular, 4-connected grid with  r rows and c columns.  The tuples in :locs contain the spatial coordinates of each vertex.  Optional arguments xlim and ylim determine the bounds of the rectangular  layout.\n\nExamples\n\njulia> out4 = makegrid4(11, 21, (-1,1), (-10,10));\njulia> nv(out4.G) == 11*21                  #231\ntrue\njulia> ne(out4.G) == 11*20 + 21*10          #430\ntrue\njulia> out4.locs[11*10 + 6] == (0.0, 0.0)   #location of center vertex.\ntrue\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.makegrid8",
    "page": "Reference",
    "title": "Autologistic.makegrid8",
    "category": "function",
    "text": "makegrid8(r::Int, c::Int, xlim::Tuple{Real,Real}=(0.0,1.0), \n          ylim::Tuple{Real,Real}=(0.0,1.0))\n\nReturns a named tuple (:G, :locs), where :G is a graph, and :locs is an array of  numeric tuples.  Vertices of :G are laid out in a rectangular, 8-connected grid with  r rows and c columns.  The tuples in :locs contain the spatial coordinates of each vertex.  Optional arguments xlim and ylim determine the bounds of the rectangular  layout.\n\nExamples\n\njulia> out8 = makegrid8(11, 21, (-1,1), (-10,10));\njulia> nv(out8.G) == 11*21                      #231\ntrue\njulia> ne(out8.G) == 11*20 + 21*10 + 2*20*10    #830\ntrue\njulia> out8.locs[11*10 + 6] == (0.0, 0.0)       #location of center vertex.\ntrue\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.makespatialgraph-Union{Tuple{C}, Tuple{C,Real}} where C<:(Union{Array{Tuple{T,T},1}, Array{Tuple{T,T,T},1}} where T<:Real)",
    "page": "Reference",
    "title": "Autologistic.makespatialgraph",
    "category": "method",
    "text": "makespatialgraph(coords::C, δ::Real) where C<:SpatialCoordinates\n\nReturns a named tuple (:G, :locs), where :G is a graph, and :locs is an array of  numeric tuples.  Each element of coords is a 2- or 3-tuple of spatial coordinates, and this argument is returned unchanged as :locs.  The graph :G has length(coords) vertices, with edges connecting every pair of vertices within Euclidean distance δ of each other. \n\nExamples\n\njulia> c = [(Float64(i), Float64(j)) for i = 1:5 for j = 1:5];\njulia> out = makespatialgraph(c, sqrt(2));\njulia> out.G\n{25, 72} undirected simple Int64 graph\n\njulia> length(out.locs)\n25\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.marginalprobabilities-Tuple{AbstractAutologisticModel}",
    "page": "Reference",
    "title": "Autologistic.marginalprobabilities",
    "category": "method",
    "text": "marginalprobabilities(M::AbstractAutologisticModel;\n    indices=1:size(M.unary,2), \n    force::Bool=false\n)\n\nCompute the marginal probability that variables in autologistic model M takes the high state. For a model with n vertices and m observations, returns an n-by-m array  (or an n-vector if  m==1). The [i,j]th element is the marginal probability of the high state in the ith variable at the jth observation.  \n\nThis function computes the exact marginals. For large models, approximate the marginal  probabilities by sampling, e.g. sample(M, ..., average=true).\n\nArguments\n\nindices: used to return only the probabilities for certain observations.  \nforce: the function will throw an error for n > 20 unless force=true.\n\nExamples\n\njulia> M = ALsimple(Graph(3,0), [[-1.0; 0.0; 1.0] [-1.0; 0.0; 1.0]])\njulia> marginalprobabilities(M)\n3×2 Array{Float64,2}:\n 0.119203  0.119203\n 0.5       0.5\n 0.880797  0.880797\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.negpotential-Tuple{AbstractAutologisticModel}",
    "page": "Reference",
    "title": "Autologistic.negpotential",
    "category": "method",
    "text": "negpotential(M::AbstractAutologisticModel)\n\nReturns an m-vector of Float64 negpotential values, where m is the number of observations found in M.responses.\n\nExamples\n\njulia> M = ALsimple(makegrid4(3,3).G, ones(9));\njulia> f = fullPMF(M);\njulia> exp(negpotential(M)[1])/f.partition ≈ exp(loglikelihood(M))\ntrue\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.oneboot-Tuple{AbstractAutologisticModel,Array{Float64,1}}",
    "page": "Reference",
    "title": "Autologistic.oneboot",
    "category": "method",
    "text": "oneboot(M::AbstractAutologisticModel, params::Vector{Float64};\n    start=zeros(length(getparameters(M))),\n    verbose::Bool=false,\n    kwargs...\n)\n\nComputes one bootstrap replicate using model M, but using parameters params for  generating samples, instead of getparameters(M).\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.oneboot-Tuple{AbstractAutologisticModel}",
    "page": "Reference",
    "title": "Autologistic.oneboot",
    "category": "method",
    "text": "oneboot(M::AbstractAutologisticModel; \n    start=zeros(length(getparameters(M))),\n    verbose::Bool=false,\n    kwargs...\n)\n\nPerforms one parametric bootstrap replication from autologistic model M: draw a random sample from M, use that sample as the responses, and re-fit the model.  Returns a named tuple (:sample, :estimate, :convergence), where :sample holds the random sample, :estimate holds the parameter estimates, and :convergence holds a bool indicating whether or not the optimization converged.  The parameters of M remain unchanged by calling oneboot.\n\nArguments\n\nstart: starting parameter values to use for optimization\nverbose: should progress information be written to the console?\nkwargs...: extra keyword arguments that are passed to optimize() or sample(), as appropriate.\n\nExamples\n\njldoctest julia> G = makegrid4(4,3).G; julia> model = ALRsimple(G, ones(12,1), Y=[fill(-1,4); fill(1,8)]); julia> theboot = oneboot(model, method=Gibbs, burnin=250); julia> fieldnames(typeof(theboot)) (:sample, :estimate, :convergence)\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.pseudolikelihood-Tuple{AbstractAutologisticModel}",
    "page": "Reference",
    "title": "Autologistic.pseudolikelihood",
    "category": "method",
    "text": "pseudolikelihood(M::AbstractAutologisticModel)\n\nComputes the negative log pseudolikelihood for autologistic model M. Returns a Float64.\n\nExamples\n\njulia> X = [1.1 2.2\n            1.0 2.0\n            2.1 1.2\n            3.0 0.3];\njulia> Y = [0; 0; 1; 0];\njulia> M3 = ALRsimple(makegrid4(2,2)[1], cat(X,X,dims=3), Y=cat(Y,Y,dims=2), \n                      β=[-0.5, 1.5], λ=1.25, centering=expectation);\njulia> pseudolikelihood(M3)\n12.333549445795818\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.sample",
    "page": "Reference",
    "title": "Autologistic.sample",
    "category": "function",
    "text": "sample(M::AbstractAutologisticModel, k::Int = 1;\n    method::SamplingMethods = Gibbs,\n    indices = 1:size(M.unary,2), \n    average::Bool = false, \n    config = nothing, \n    burnin::Int = 0,\n    skip::Int = 0,\n    verbose::Bool = false\n)\n\nDraws k random samples from autologistic model M. For a model with n vertices in its graph, the return value is:\n\nWhen average=false, an n × length(indices) × k array, with singleton dimensions dropped. This array holds the random samples.\nWhen average=true, an n  × length(indices) array, with singleton dimensions dropped. This array holds the estimated marginal probabilities of observing the \"high\" level at  each vertex.\n\nArguments\n\nmethod: a member of the enum SamplingMethods, specifying which sampling method will be used.  The default is Gibbs sampling.  Where feasible, it is recommended  to use one of the perfect sampling alternatives. See SamplingMethods for more.\nindices: gives the indices of the observation to use for sampling. If the model has  more than one observation, then k samples are drawn for each observation\'s parameter  settings. Use indices to restrict the samples to a subset of observations. \naverage: controls the form of the output. When average=true, the return value is the  proportion of \"high\" samples at each vertex. (Note that this is not actually the arithmetic average of the samples, unless the coding is (0,1). Rather, it is an estimate of  the probability of getting a \"high\" outcome.)  When average=false, the full set of samples is returned. \nconfig: allows a starting configuration of the random variables to be provided. Only used if method=Gibbs. Any vector of the correct length, with two unique values, can be  used as config. By default a random configuration is used.\nburnin: specifies the number of initial samples to discard from the results.  Only used if method=Gibbs.\nskip: specifies how many samples to throw away between returned samples.  Only used  if method=Gibbs. \nverbose: controls output to the console.  If true, intermediate information about  sampling progress is printed to the console. Otherwise no output is shown.\n\nExamples\n\njulia> M = ALsimple(Graph(4,4));\njulia> M.coding = (-2,3);\njulia> r = sample(M,10);\njulia> size(r)\n(4, 10)\njulia> sort(unique(r))\n2-element Array{Float64,1}:\n -2.0\n  3.0\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.setpairwiseparameters!-Tuple{AbstractAutologisticModel,Array{Float64,1}}",
    "page": "Reference",
    "title": "Autologistic.setpairwiseparameters!",
    "category": "method",
    "text": "setpairwiseparameters!(M::AbstractAutologisticModel, newpars::Vector{Float64})\n\nSets the pairwise parameters of autologistic model M to the values in newpars.\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.setparameters!-Tuple{AbstractAutologisticModel,Array{Float64,1}}",
    "page": "Reference",
    "title": "Autologistic.setparameters!",
    "category": "method",
    "text": "setparameters!(x, newpars::Vector{Float64})\n\nA generic function for setting the parameter values of an autologistic model, a unary term, or a pairwise term.  Parameters are always passed as an Array{Float64,1}.  If  typeof(x) <: AbstractAutologisticModel, the newpars is assumed partitioned with the unary parameters first.\n\n\n\n\n\n"
},

{
    "location": "api/#Autologistic.setunaryparameters!-Tuple{AbstractAutologisticModel,Array{Float64,1}}",
    "page": "Reference",
    "title": "Autologistic.setunaryparameters!",
    "category": "method",
    "text": "setunaryparameters!(M::AbstractAutologisticModel, newpars::Vector{Float64})\n\nSets the unary parameters of autologistic model M to the values in newpars.\n\n\n\n\n\n"
},

{
    "location": "api/#Methods-1",
    "page": "Reference",
    "title": "Methods",
    "category": "section",
    "text": "Modules = [Autologistic]\r\nOrder   = [:function]"
},

]}
