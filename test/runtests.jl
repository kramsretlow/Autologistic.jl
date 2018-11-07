# NOTE: @test_warn and @test_nowarn give Base.IOError exception when this 
# file is included in jupyter notebook.
using Test
using LightGraphs
using Autologistic

println("Running tests:")

@testset "FullUnary constructors and interfaces" begin
    M = [1.1 4.4 7.7
         2.2 5.5 8.8
         3.3 4.4 9.9]
    u1 = FullUnary(M[:,1])
    u2 = FullUnary(M)
    
    @test values(u1) == reshape([1.1; 2.2; 3.3], (3,1))
    @test values(u2) == M
    @test u1[2] == 2.2
    @test u2[2,3] == 8.8
    @test size(u1) == (3,1)
    @test size(u2) == (3,3)
    @test getparameters(u1) == [1.1; 2.2; 3.3]

    setparameters!(u1, [0.1, 0.2, 0.3])
    setparameters!(u2, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    u1[2] = 2.22
    u2[2,3] = 8.88    

    @test values(u1) == reshape([0.1, 2.22, 0.3], (3,1))
    @test values(u2) == [0.1 0.4 0.7
                         0.2 0.5 8.88
                         0.3 0.6 0.9] 

    u3 = FullUnary(10)
    u4 = FullUnary(10,4)

    @test size(u3) == (10,1)
    @test size(u4) == (10,4)
end

@testset "LinPredUnary constructors and interfaces" begin
    X1 = [1.0 2.0 3.0
         1.0 4.0 5.0
         1.0 6.0 7.0
         1.0 8.0 9.0]
    X = cat(X1, 2*X1, dims=3)
    beta = [1.0, 2.0, 3.0]
    u1 = LinPredUnary(X, beta)
    u2 = LinPredUnary(X1, beta)
    u3 = LinPredUnary(X1)
    u4 = LinPredUnary(X)
    u5 = LinPredUnary(4, 3)
    u6 = LinPredUnary(4, 3, 2)
    Xbeta = [14.0 28.0
             24.0 48.0
             34.0 68.0
             44.0 88.0]
    X1beta = reshape(Xbeta[:,1], (4,1))

    @test size(u1) == size(u4) == size(u6) == (4,2)
    @test size(u2) == size(u3) == size(u5) == (4,1)
    @test values(u1) == Xbeta 
    @test values(u2) == X1beta
    @test u1[3,2] == u1[7] == 68.0
    @test getparameters(u1) == beta

    setparameters!(u1, [2.0, 3.0, 4.0])

    @test getparameters(u1) == [2.0, 3.0, 4.0]

end

@testset "SimplePairwise constructors and interfaces" begin
    n = 10                                                       # length of y_i
    m = 3                                                 # number of replicates
    λ = 1.0
    G = Graph(n, Int(floor(n*(n-1)/4)))
    p1 = SimplePairwise([λ], G, m)
    p2 = SimplePairwise(G)
    p3 = SimplePairwise(G, m)
    p4 = SimplePairwise(n)
    p5 = SimplePairwise(n, m)
    p6 = SimplePairwise(λ, G)
    p7 = SimplePairwise(λ, G, m)

    @test any(i -> (i!==(n,n,m)), [size(j) for j in [p1, p2, p3, p4, p5, p6, p7]])
    @test values(p1) == values(p6) == values(p7) == λ*adjacency_matrix(G)
    @test p1[2,2,2] == p1[2,2] == λ*adjacency_matrix(G)[2,2]
    @test p1[:,:,1] == p1[:,:,2] == p1[:,:,3] == values(p1)

    setparameters!(p1, [2.0])

    @test getparameters(p1) == [2.0]
end

@testset "ALmodel constructors" begin
    (n, p, m) = (100, 4, 1)
    X = rand(n,p,m)
    β = [1.0, 2.0, 3.0, 4.0]
    Y = makebool(round.(rand(n,m)))
    unary = LinPredUnary(X, β)
    pairwise = SimplePairwise(n, m)
    m1 = ALmodel(Y, unary, pairwise, none, (-1.0,1.0), ("low","high"))
    m2 = ALmodel(unary, pairwise)
    m3 = ALRsimple(Graph(n, Int(floor(n*(n-1)/4))), X, Y=Y, β=β, λ = 1.0)

    @test getparameters(m3) == [β; 1.0]
    @test getunaryparameters(m3) == β
    @test getpairwiseparameters(m3) == [1.0]
    
    setparameters!(m1, [1.1, 2.2, 3.3, 4.4, -1.0])
    setunaryparameters!(m2, [1.1, 2.2, 3.3, 4.4])
    setpairwiseparameters!(m2, [-1.0])

    @test getparameters(m1) == getparameters(m2) == [1.1, 2.2, 3.3, 4.4, -1.0]
end

@testset "Helper functions" begin
    y1 = [false, false, true]
    y2 = [1 2; 1 2]
    y3 = [1.0 2.0; 1.0 2.0]
    y4 = ["yes", "no", "no"]

    @test makebool(y1) == reshape([false, false, true], (3,1))
    @test makebool(y2) == makebool(y3) == [false true; false true]
    @test makebool(y4) == reshape([true, false, false], (3,1))
end