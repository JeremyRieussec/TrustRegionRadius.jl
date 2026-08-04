using TrustRegionRadius

using Test
using ADNLPModels
using NLPModels
using LinearAlgebra
using Statistics
using Random
using SolverCore

@testset "TrustRegionRadius.jl" begin
    include("test_rules.jl")
    include("test_models.jl")
    include("test_subproblem.jl")
    include("test_solver.jl")
    include("test_profiles.jl")
    include("test_thresholds.jl")
    include("test_second_order.jl")
    include("test_stochastic.jl")
    include("test_likelihood.jl")
    include("test_sampling.jl")
end
