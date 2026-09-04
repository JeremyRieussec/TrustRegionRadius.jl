# The two conventions introduced with the decoupled interface, checked at unit
# level for every rule:
#
#   1. 0 ≤ η ≤ η1 ≤ η2 < 1, with η governing acceptance and η1, η2 governing
#      scaling only.  In particular ρ ∈ [η, η1) is accepted *and* contracts, and
#      η = 0 is legal.
#   2. 0 < γ1 ≤ γ2 < 1 < γ3 in every rule, rejected at construction.
#
# Plus the property the whole first-order theory rests on: every rule returns
# Δ_{k+1} < Δ_k on an unsuccessful iteration.  That is the test the retrospective
# rules failed when a classical ρ was compared against retrospective thresholds.
#
# End-to-end versions of the same invariants — measured on real runs rather than
# synthetic arguments — live in test_solver.jl; per-rule semantics live in
# test_rules.jl.

@testset "thresholds and factors" begin

    # Every rule under its defaults. Named so a failure says which one.
    RULES = [
        ("RDelta",           RDelta()),
        ("RStep",            RStep()),
        ("RDeltaStep",       RDeltaStep()),
        ("RDFO",             RDFO()),
        ("RGrad",            RGrad()),
        ("RGradCapped",      RGradCapped(μ_max = 8.0)),
        ("RAdaptiveStep",    RAdaptiveStep()),
        ("RAdaptiveGrad",    RAdaptiveGrad()),
        ("RAdaptiveGradCapped", RAdaptiveGradCapped(μ_max = 8.0)),
        ("RRTR",             RRTR()),
        ("RRTRGrad",         RRTRGrad()),
    ]

    @testset "the threshold chain is enforced" begin
        @test_throws ArgumentError TRParams(η = 0.5, η1 = 0.2, η2 = 0.9)   # η > η1
        @test_throws ArgumentError TRParams(η1 = 0.9, η2 = 0.2)            # η1 > η2
        @test_throws ArgumentError TRParams(η = -0.1)                      # η < 0
        @test_throws ArgumentError TRParams(η1 = 1.0, η2 = 1.0)            # η2 ≥ 1

        # η defaults to η1: the classical coupled algorithm, unchanged.
        @test TRParams().η == TRParams().η1
        @test TRParams(η1 = 0.2).η == 0.2

        # η = 0 is legal — the case Part I covers and Curtis & Scheinberg do not.
        @test TRParams(η = 0.0, η1 = 0.1, η2 = 0.9).η == 0.0

        # Genuine decoupling: the middle band is non-empty.
        p = TRParams(η = 0.0, η1 = 0.25, η2 = 0.75)
        @test p.η < p.η1 < p.η2
    end

    @testset "factor convention 0 < γ1 ≤ γ2 < 1 < γ3" begin
        # γ2 in the expansion slot is the pre-refactor mistake; it must be
        # rejected rather than silently reinterpreted.
        @test_throws ArgumentError RGrad(γ2 = 2.0)
        @test_throws ArgumentError RDelta(γ2 = 1.5)
        @test_throws ArgumentError RRTR(γ2 = 2.5)
        @test_throws ArgumentError RRTRGrad(γ1 = 1.5)

        # γ1 ≤ γ2 is enforced, not merely documented.
        @test_throws ArgumentError RDelta(γ1 = 0.8, γ2 = 0.3)
        @test_throws ArgumentError RStep(γ1 = 0.8, γ2 = 0.3)
        @test_throws ArgumentError RDFO(γ1 = 0.8, γ2 = 0.3)

        # γ3 must exceed 1.
        @test_throws ArgumentError RDelta(γ3 = 0.9)
        @test_throws ArgumentError RGrad(γ3 = 1.0)

        # The Hei family needs M > γ3: γ3 is the value of R at the threshold and
        # M its asymptote. This replaced the earlier γ3 > 1 + γ2, which belonged
        # to the normalisation in which R(η1) = 1 + γ2 and γ3 was the asymptote.
        @test_throws ArgumentError RAdaptiveStep(γ3 = 2.0, M = 1.5)
        @test_throws ArgumentError RAdaptiveGrad(γ3 = 2.0, M = 1.5)
        @test_throws ArgumentError RAdaptiveGradCapped(γ3 = 2.0, M = 1.5)
        @test RAdaptiveStep(γ2 = 0.5, γ3 = 1.4, M = 5.0) isa RAdaptiveStep
        @test RAdaptiveStep(γ2 = 0.5, γ3 = 1.6, M = 5.0) isa RAdaptiveStep

        # And the defaults of every rule satisfy the chain they advertise.
        for (name, r) in RULES
            fs = fieldnames(typeof(r))
            if :γ1 in fs && :γ2 in fs && :γ3 in fs
                @test 0 < r.γ1 <= r.γ2 < 1 < r.γ3
            elseif :γ1 in fs && :γ3 in fs
                @test 0 < r.γ1 < 1 < r.γ3
            end
        end
    end

    @testset "unsuccessful iterations contract, every rule" begin
        # ρ < η ≤ η1, the step rejected. Reproduces the failure mode of the old
        # retrospective rules, where ρ was compared against η̃₁ and could leave Δ
        # untouched — after which the same subproblem is re-solved for ever.
        η, η1, η2 = 0.1, 0.25, 0.75
        Δ, s_norm, g_old, g_new = 1.0, 0.8, 2.0, 2.0   # rejected ⟹ ‖g‖ unchanged

        for (name, rule) in RULES
            @testset "$name" begin
                for ρ in (-1.0, -1e-8, 0.0, 0.05, 0.099)   # all strictly below η
                    r = deepcopy(rule)
                    reset_rule!(r)
                    # initial_radius, not Δ directly: for the μ-based rules the
                    # radius and the multiplier must start consistent, as they do
                    # in a run.
                    Δ0 = initial_radius(r, Δ, g_old)
                    Δ1 = update_radius!(r, Δ0, ρ, false, η1, η2, s_norm, g_old, g_new)
                    @test Δ1 < Δ0
                    @test Δ1 > 0
                end
            end
        end
    end

    @testset "accepted-but-contracting iterations" begin
        # ρ ∈ [η, η1): the step is taken and the radius still shrinks. The rule is
        # told accepted = true and must nevertheless contract, because it branches
        # on η1. This band does not exist when η = η1.
        η, η1, η2 = 0.0, 0.25, 0.75
        ρ = 0.1
        Δ, s_norm, g_old, g_new = 1.0, 0.9, 2.0, 1.5    # accepted ⟹ ‖g‖ fell

        for (name, rule) in RULES
            needs_retrospective(rule) && continue        # ρ̃ ≠ ρ; see test_rules.jl
            @testset "$name" begin
                r = deepcopy(rule); reset_rule!(r)
                Δ0 = initial_radius(r, Δ, g_old)
                Δ1 = update_radius!(r, Δ0, ρ, true, η1, η2, s_norm, g_old, g_new)
                @test Δ1 < Δ0
            end
        end
    end
end
