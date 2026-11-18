using TrustRegionRadius

using Test
using ADNLPModels
using NLPModels
using LinearAlgebra
using Statistics
using Random
using SolverCore

"""
    exported_concrete(T) -> Set{Symbol}

The concrete subtypes of `T` that the package exports, read off the export list.

The rule lists in these tests are compared with it, so that a rule added to the
package and not to a list fails the suite. Two radius rules and three sampling
rules once went untested because the lists were kept by hand and nothing checked
them.
"""
exported_concrete(T::Type) = Set(
    n for n in names(TrustRegionRadius)
    if isdefined(TrustRegionRadius, n) &&
       (v = getfield(TrustRegionRadius, n);
        v isa Type && v <: T && v !== T && !isabstracttype(v)))

@testset "TrustRegionRadius.jl" begin
    include("test_problem_classes.jl")
    include("test_rules.jl")
    include("test_models.jl")
    include("test_subproblem.jl")
    include("test_solver.jl")
    include("test_profiles.jl")
    include("test_thresholds.jl")
    include("test_diagnostics.jl")
    include("test_recording.jl")
    include("test_flat_well.jl")
    include("test_second_order.jl")
    include("test_stochastic.jl")
    include("test_likelihood.jl")
    include("test_probit.jl")
    include("test_sampling.jl")
    include("test_sampling_features.jl")
end
