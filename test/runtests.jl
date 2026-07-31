using TrustRegionRadius

using Test
using ADNLPModels
using NLPModels
using LinearAlgebra
using SolverCore

@testset "TrustRegionRadius.jl" begin
    println("------------- Test RULES -------------------")
    include("test_rules.jl")
    println("------------- Test Models -------------------")
    include("test_models.jl")
    println("------------- Test Subproblem -------------------")
    include("test_subproblem.jl")
    println("------------- Test Solver -------------------")
    include("test_solver.jl")
    println("------------- Test Profiles -------------------")
    include("test_profiles.jl")
    println("-------------- Test Thresholds -------------------")
    include("test_thresholds.jl")
    println("--------------- DONE -----------------")
end
