# The two conventions introduced with the decoupled interface, checked at unit
# level for every rule:
#
#   1. 0 ≤ η ≤ η₁ ≤ η₂ < 1, with η governing acceptance and η₁, η₂ governing
#      scaling only.  In particular ρ ∈ [η, η₁) is accepted *and* contracts, and
#      η = 0 is legal.
#   2. 0 < γ₁ ≤ γ₂ < 1 < γ₃ in every rule, rejected at construction.
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
        ("RDFO",             RDFO()),
        ("RGrad",            RGrad()),
        ("RGradCapped",      RGradCapped(μ_max = 8.0)),
        ("RAdaptiveStep",    RAdaptiveStep()),
        ("RAdaptiveGrad",    RAdaptiveGrad()),
        ("RRTR",             RRTR()),
        ("RRTRGrad",         RRTRGrad()),
    ]

    @testset "the threshold chain is enforced" begin
        @test_throws ArgumentError TRParams(η = 0.5, η₁ = 0.2, η₂ = 0.9)   # η > η₁
        @test_throws ArgumentError TRParams(η₁ = 0.9, η₂ = 0.2)            # η₁ > η₂
        @test_throws ArgumentError TRParams(η = -0.1)                      # η < 0
        @test_throws ArgumentError TRParams(η₁ = 1.0, η₂ = 1.0)            # η₂ ≥ 1

        # η defaults to η₁: the classical coupled algorithm, unchanged.
        @test TRParams().η == TRParams().η₁
        @test TRParams(η₁ = 0.2).η == 0.2

        # η = 0 is legal — the case Part I covers and Curtis & Scheinberg do not.
        @test TRParams(η = 0.0, η₁ = 0.1, η₂ = 0.9).η == 0.0

        # Genuine decoupling: the middle band is non-empty.
        p = TRParams(η = 0.0, η₁ = 0.25, η₂ = 0.75)
        @test p.η < p.η₁ < p.η₂
    end

    @testset "factor convention 0 < γ₁ ≤ γ₂ < 1 < γ₃" begin
        # γ₂ in the expansion slot is the pre-refactor mistake; it must be
        # rejected rather than silently reinterpreted.
        @test_throws ArgumentError RGrad(γ₂ = 2.0)
        @test_throws ArgumentError RDelta(γ₂ = 1.5)
        @test_throws ArgumentError RRTR(γ₂ = 2.5)
        @test_throws ArgumentError RRTRGrad(γ₁ = 1.5)

        # γ₁ ≤ γ₂ is enforced, not merely documented.
        @test_throws ArgumentError RDelta(γ₁ = 0.8, γ₂ = 0.3)
        @test_throws ArgumentError RStep(γ₁ = 0.8, γ₂ = 0.3)
        @test_throws ArgumentError RDFO(γ₁ = 0.8, γ₂ = 0.3)

        # γ₃ must exceed 1.
        @test_throws ArgumentError RDelta(γ₃ = 0.9)
        @test_throws ArgumentError RGrad(γ₃ = 1.0)

        # The Hei family needs the stronger γ₃ > 1 + γ₂.
        @test_throws ArgumentError RAdaptiveStep(γ₂ = 0.5, γ₃ = 1.4)
        @test RAdaptiveStep(γ₂ = 0.5, γ₃ = 1.6) isa RAdaptiveStep

        # And the defaults of every rule satisfy the chain they advertise.
        for (name, r) in RULES
            fs = fieldnames(typeof(r))
            if :γ₁ in fs && :γ₂ in fs && :γ₃ in fs
                @test 0 < r.γ₁ <= r.γ₂ < 1 < r.γ₃
            elseif :γ₁ in fs && :γ₃ in fs
                @test 0 < r.γ₁ < 1 < r.γ₃
            end
        end
    end

    @testset "unsuccessful iterations contract, every rule" begin
        # ρ < η ≤ η₁, the step rejected. Reproduces the failure mode of the old
        # retrospective rules, where ρ was compared against η̃₁ and could leave Δ
        # untouched — after which the same subproblem is re-solved for ever.
        η, η₁, η₂ = 0.1, 0.25, 0.75
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
                    Δ1 = update_radius!(r, Δ0, ρ, false, η₁, η₂, s_norm, g_old, g_new)
                    @test Δ1 < Δ0
                    @test Δ1 > 0
                end
            end
        end
    end

    @testset "accepted-but-contracting iterations" begin
        # ρ ∈ [η, η₁): the step is taken and the radius still shrinks. The rule is
        # told accepted = true and must nevertheless contract, because it branches
        # on η₁. This band does not exist when η = η₁.
        η, η₁, η₂ = 0.0, 0.25, 0.75
        ρ = 0.1
        Δ, s_norm, g_old, g_new = 1.0, 0.9, 2.0, 1.5    # accepted ⟹ ‖g‖ fell

        for (name, rule) in RULES
            needs_retrospective(rule) && continue        # ρ̃ ≠ ρ; see test_rules.jl
            @testset "$name" begin
                r = deepcopy(rule); reset_rule!(r)
                Δ0 = initial_radius(r, Δ, g_old)
                Δ1 = update_radius!(r, Δ0, ρ, true, η₁, η₂, s_norm, g_old, g_new)
                @test Δ1 < Δ0
            end
        end
    end
end
