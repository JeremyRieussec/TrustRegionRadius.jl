using Test
using TrustRegionRadius
using ADNLPModels
using NLPModels
using LinearAlgebra

@testset "TrustRegionRadius.jl" begin
    include("test_rules.jl")
    include("test_models.jl")
    include("test_subproblem.jl")
    include("test_solver.jl")
    include("test_profiles.jl")
end
