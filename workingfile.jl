# Development scratch file

# NB: in this script I'm trying to use begin/end blocks to allow "cell mode"
# functionality.  Use ALT+Enter to run the code in the begin/end "cell".

# === Set up the package development environment (e.g. after restarting kernel)
begin #-Setup tasks
    pwd()  #should be \"C:\\\\GitWorking\\\\Autologistic"
    using Pkg
    Pkg.activate(pwd())   #At cmd line, could use ]activate .
    
    using Revise
    using Autologistic
    using Plots, Profile, ProfileView, LinearAlgebra

    include("test\\runtests.jl")
end


# === Trying to speed things up ===
using BenchmarkTools
n1 = 25
G = grid4(n1,n1)
M = ALRsimple(G[1], rand(n1^2,3))

# NB: use $ before variables in @btime to "interpolate" them into the expression
# to avoid problems benchmarking with global variables.
@btime sample($M, 100, average=true);

# === Test out perfect sampling (performance) ===
# TODO: allocations and run time go up inordinately with 
# the number of samples...
setparameters!(M, [-2, 1, 1, 0.5])
@btime sample($M, 100, method=CFTPlarge, average=true);
@btime sample($M, 100, method=CFTPsmall, average=true);
@btime sample($M, 100, method=ROCFTP, average=true);
@btime sample($M, 100, method=CFTPbound, average=true);


# === Test out perfect sampling (plot) ===
setparameters!(M, [-2, 1, 1, 0.5])
S = sample(M, method=ROCFTP, verbose=true);
using GraphPlot
#In VS Code, running gplot causes left/right arrow keys to be siezed bythe 
# plot window, not usable in editor...
gplot(G.G, [G.locs[i][1] for i=1:n1^2], [G.locs[i][2] for i=1:n1^2],
      NODESIZE=0.02, nodefillc = map(x -> x==-1 ? "red" : "green", S[:]))



#Check pefect sampling averages are consistent with the truth in small cases.
n = 10
maxedges = n*(n-1)/2
our_edge_range = 0:Int(floor(maxedges/2))
G = Graph(n, rand(our_edge_range))
M = ALmodel(FullUnary(randn(n)), SimplePairwise(0.75, G))
truemarg = marginalprobabilities(M);

setpairwiseparameters!(M, [2.0])
sample(M, method=CFTPbound, verbose=true)
sample(M, method=CFTPsmall, verbose=true)

sampmarg = sample(M, 1000, method=CFTPbound, average=true);
round.([truemarg sampmarg abs.(truemarg .- sampmarg)], digits=4)






