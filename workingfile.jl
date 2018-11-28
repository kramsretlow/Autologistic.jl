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
    using Plots, Profile, ProfileView

    include("test\\runtests.jl")
end


# === Trying to speed things up ===
using BenchmarkTools
n1 = 35
G = grid4(n1,n1)
M = ALRsimple(G[1], rand(n1^2,3))

# NB: use $ before variables in @btime to "interpolate" them into the expression
# to avoid problems benchmarking with global variables.
@btime sample($M, 100, average=false);

# === Test out perfect sampling (performance) ===
# TODO: allocations and run time go up inordinately with 
# the number of samples...
setparameters!(M, [-2, 1, 1, 0.5])
@btime sample($M, 10, method=perfect);


# === Test out perfect sampling (plot) ===
setparameters!(M, [-2, 1, 1, 0.5])
S = sample(M, method=perfect, verbose=true);
using GraphPLot
gplot(G.G, [G.locs[i][1] for i=1:n1^2], [G.locs[i][2] for i=1:n1^2],
      NODESIZE=0.02, nodefillc = map(x -> x==-1 ? "red" : "green", S[:]))






function wrap2(k, avg=true)
    n1 = 35
    M = ALRsimple(grid4(n1,n1)[1], rand(n1^2,3))
    @time S = sample(M,k,average=avg);
end


# === Playing with allocations
function foo(n)
    A = rand(n,n)
    A[A .< 0.5] .= 0
    B = sparse(A)

    v = rand(n)
    @time A[:,1]' * v
    tot = 0.0
    @time begin
        for i = 1:n
            tot += A[i,1]*v[i]
        end
    end
end


