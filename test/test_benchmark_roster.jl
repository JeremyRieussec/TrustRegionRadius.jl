# The roster of sampling rules that the sampled experiments draw from lives in
# benchmark/config.jl. It is loaded here, in a module of its own, and compared
# with the rules the package exports.
#
# The reason is the one behind the guards in test_sampling.jl and
# test_thresholds.jl: a list kept by hand falls behind the package, and nothing
# says so until a rule has gone unrun for months. The benchmark scripts are not
# executed by the test suite, so without this the roster would be checked by
# nobody.
#
# config.jl needs nothing but the package to load. Its functions that build
# CUTEst or ADNLPModels problems name those packages only inside their bodies.

module BenchmarkConfig
    using TrustRegionRadius
    include(joinpath(@__DIR__, "..", "benchmark", "config.jl"))
end

@testset "the benchmark roster of sampling rules" begin
    roster = BenchmarkConfig.SAMPLING_RULES

    # Every sampling rule the package exports is in the roster, and nothing else.
    @test Set(nameof(typeof(f())) for (_, f) in roster) == exported_concrete(SamplingRule)
    @test allunique(first.(roster))

    # Each entry is a factory. Several rules carry state between iterations, so a
    # run must be handed an instance of its own.
    @test all(f isa Function for (_, f) in roster)
    stateful = [f for (_, f) in roster if ismutabletype(typeof(f()))]
    @test !isempty(stateful)
    @test all(f() !== f() for f in stateful)

    # Selection is by name, in the order asked for, and an unknown name is an
    # error, so that a typing mistake cannot shrink an experiment silently.
    pick = BenchmarkConfig.sampling_rules("NormTest(0.5)", "FixedSample(64)")
    @test first.(pick) == ["NormTest(0.5)", "FixedSample(64)"]
    @test pick[1][2]() isa NormTest
    @test pick[2][2]() isa FixedSample
    @test isempty(BenchmarkConfig.sampling_rules())
    @test_throws ArgumentError BenchmarkConfig.sampling_rules("NormTest(0.05)")

    # No entry carries a cap of its own: on a finite sum the cap is the
    # population, and the oracle refuses a rule that brings another.
    p = LogisticRegression(K = 3, M = 200, seed = 1)
    for (name, f) in roster
        @test FiniteSumNLP(p, f(); seed = 1) isa FiniteSumNLP
    end
end
