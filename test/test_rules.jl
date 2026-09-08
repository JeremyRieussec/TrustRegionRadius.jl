# Radius rules: the update contract, and the invariants the survey relies on.
#
# The signature is
#
#     update_radius!(rule, Δ, ρ, accepted, η1, η2, s_norm, g_norm_old, g_norm_new)
#
# with `accepted` third. The threshold chain 0 ≤ η ≤ η1 ≤ η2 < 1, the factor
# convention 0 < γ1 ≤ γ2 < 1 < γ3, and the contraction obligation are exercised
# for every rule at once in test_thresholds.jl; what follows is per-rule
# semantics.

@testset "rules" begin
    η, η1, η2 = 0.1, 0.2, 0.9

    @testset "every rule implements the contract" begin
        rules = [RDelta(), RStep(), RDeltaStep(), RDFO(), RGrad(), RGradCapped(),
                 RAdaptiveStep(), RAdaptiveGrad(), RAdaptiveGradCapped(),
                 RRTR(), RRTRGrad()]
        for r in rules
            @test r isa RadiusRule
            Δ0 = initial_radius(r, 1.0, 2.0)
            @test Δ0 > 0
            Δ1 = update_radius!(r, 1.0, 0.95, true, η1, η2, 0.8, 2.0, 1.5)
            @test Δ1 > 0 && isfinite(Δ1)
            @test reset_rule!(r) === nothing
            @test asymptotic_regime(r) in (:vanishing, :step_summable, :bounded_below)
        end
    end


    @testset "RDeltaStep: step on failure, radius on success" begin
        # The appendix's R-delta-step. Distinct from RStep, which rebuilds the
        # radius from the step on EVERY branch: here only the failure branch does.
        r = RDeltaStep(γ1 = 0.25, γ2 = 0.5, γ3 = 2.0)
        Δ, sn = 2.0, 0.4
        @test update_radius!(r, Δ, 0.05, false, η1, η2, sn, 1.0, 1.0) == 0.25 * sn
        @test last_branch(r) === :contract
        @test update_radius!(r, Δ, 0.5, true, η1, η2, sn, 1.0, 1.0) == 0.5 * Δ
        @test last_branch(r) === :shrink
        @test update_radius!(r, Δ, 0.95, true, η1, η2, sn, 1.0, 1.0) == 2.0 * Δ
        @test last_branch(r) === :expand

        # and it is NOT RStep: RStep contracts on γ1‖s‖ too but shrinks and
        # expands on ‖s‖, where this rule uses Δ.
        rs = RStep(γ1 = 0.25, γ2 = 0.5, γ3 = 2.0)
        @test update_radius!(rs, Δ, 0.5, true, η1, η2, sn, 1.0, 1.0) == 0.5 * sn
        @test update_radius!(r,  Δ, 0.5, true, η1, η2, sn, 1.0, 1.0) == 0.5 * Δ

        # γ2 = 1 is admissible here and nowhere else.
        @test RDeltaStep(γ2 = 1.0) isa RDeltaStep
        @test_throws ArgumentError RDeltaStep(γ2 = 1.5)
        @test_throws ArgumentError RDeltaStep(γ1 = 0.8, γ2 = 0.3)
        @test_throws ArgumentError RDeltaStep(γ3 = 0.9)
        @test asymptotic_regime(RDeltaStep()) === :bounded_below
        @test !is_criticality_anchored(RDeltaStep())
    end

    @testset "RAdaptiveGradCapped: the capped adaptive multiplier" begin
        # RAdaptiveGrad(ω, μ̄): the hold branch of the uncapped rule, plus the cap.
        r = RAdaptiveGradCapped(μ = 1.0, μ_max = 1.0)
        update_radius!(r, 1.0, 0.95, true, η1, η2, 0.9, 1.0, 1.0)   # wants to grow
        @test last_branch(r) === :expand_capped
        @test r.μ == 1.0                                            # the cap held it

        r2 = RAdaptiveGradCapped(μ = 1.0, μ_max = 100.0)
        before = r2.μ
        update_radius!(r2, 1.0, 0.95, true, η1, η2, 0.1, 1.0, 1.0)  # ‖s‖ ≤ Δ/2
        @test last_branch(r2) === :hold
        @test r2.μ == before
        update_radius!(r2, 1.0, 0.95, true, η1, η2, 0.9, 1.0, 1.0)  # ‖s‖ > Δ/2
        @test last_branch(r2) === :expand
        @test r2.μ > before

        # μ0 is clamped to the cap rather than rejected, so a sweep over μ̄ at
        # fixed μ0 stays constructible below μ0.
        @test RAdaptiveGradCapped(μ = 1.0, μ_max = 1e-3).μ == 1e-3
        @test is_criticality_anchored(RAdaptiveGradCapped())
        @test asymptotic_regime(RAdaptiveGradCapped()) === :vanishing
    end

    @testset "the R-function is normalised to R(η1) = γ3 and R(∞) = M" begin
        γ1, γ2, γ3, M, λ1, λ2 = 0.01, 0.9, 1.5, 5.0, 5.0, 5.0
        R(t) = TrustRegionRadius._r_exp(t, η1, γ1, γ2, γ3, M, λ1, λ2)
        @test R(η1) ≈ γ3                       # value at the threshold
        @test R(1e6) ≈ M rtol = 1e-8           # asymptote
        @test R(-1e6) ≈ γ1 rtol = 1e-8         # the other asymptote
        @test R(-Inf) == γ1                    # the solver's failure convention
        @test R(prevfloat(η1)) <= γ2           # below the threshold R ≤ γ2 < 1
        # non-decreasing across the join
        ts = range(-2.0, 2.0; length = 400)
        @test issorted(R.(ts))
        # M > γ3 is the constraint, and it is enforced
        @test_throws ArgumentError RAdaptiveStep(γ3 = 2.0, M = 1.5)
        @test_throws ArgumentError RAdaptiveGrad(γ3 = 2.0, M = 1.5)
        @test_throws ArgumentError RAdaptiveGradCapped(γ3 = 2.0, M = 1.5)
    end

    @testset "the two-way and three-way classifications agree" begin
        # A criticality-anchored rule drives Δ → 0, so it must report :vanishing.
        for r in (RDelta(), RStep(), RDeltaStep(), RDFO(), RGrad(), RGradCapped(),
                  RAdaptiveStep(), RAdaptiveGrad(), RAdaptiveGradCapped(),
                  RRTR(), RRTRGrad())
            is_criticality_anchored(r) && @test asymptotic_regime(r) === :vanishing
        end
        # The middle regime is what the two-way predicate cannot express.
        @test asymptotic_regime(RDelta()) === :bounded_below
        @test asymptotic_regime(RStep())  === :step_summable
        @test asymptotic_regime(RRTR())   === :step_summable
        @test asymptotic_regime(RDFO())   === :vanishing
        @test !is_criticality_anchored(RStep())
    end

    @testset "RDelta: monotone in ρ" begin
        r = RDelta(γ1 = 0.25, γ2 = 0.5, γ3 = 2.0)
        bad  = update_radius!(r, 1.0, 0.0,  false, η1, η2, 1.0, 1.0, 1.0)
        ok   = update_radius!(r, 1.0, 0.5,  true,  η1, η2, 1.0, 1.0, 1.0)
        good = update_radius!(r, 1.0, 0.95, true,  η1, η2, 1.0, 1.0, 1.0)
        @test bad < ok < good
        @test good == 2.0
    end

    @testset "RDelta ignores `accepted`" begin
        # The branch is chosen by η1 and η2 alone, so with η < η1 a step can be
        # accepted and the radius contracted at the same iteration.
        r = RDelta()
        @test update_radius!(r, 1.0, 0.05, true,  η1, η2, 1.0, 1.0, 1.0) ==
              update_radius!(r, 1.0, 0.05, false, η1, η2, 1.0, 1.0, 1.0)
    end

    @testset "RStep: Δmin prevents collapse" begin
        # An accepted step of length ~0 would drive Δ to 0 without the floor,
        # after which every subsequent step is 0 and the solver cannot recover.
        r = RStep(Δmin = 1e-14)
        @test update_radius!(r, 1.0, 0.95, true, η1, η2, 0.0, 1.0, 1.0) >= 1e-14
        runguarded = RStep(Δmin = 0.0)
        @test update_radius!(runguarded, 1.0, 0.95, true, η1, η2, 0.0, 1.0, 1.0) == 0.0
    end

    @testset "RStep: η1 = 0 is refused" begin
        # At η1 = 0 the aggressive branch is unreachable and the lower-bound
        # constant of Part I degenerates.
        @test_throws ArgumentError validate_thresholds(RStep(), 0.0, 0.0, 0.9)
        @test validate_thresholds(RStep(), 0.0, 0.1, 0.9) === nothing
        @test validate_thresholds(RDelta(), 0.0, 0.0, 0.9) === nothing
    end

    @testset "RDFO: expands only when Δ ≤ ζ‖g‖" begin
        r = RDFO(γ2 = 0.5, γ3 = 2.0, ζ = 1.0)
        @test update_radius!(r, 0.5, 0.95, true, η1, η2, 0.5, 1.0, 1.0) == 1.0  # expand
        @test update_radius!(r, 2.0, 0.95, true, η1, η2, 2.0, 1.0, 1.0) == 1.0  # shrink
        @test is_criticality_anchored(r)
        # ‖g_k‖ *before* the decision drives it, not ‖g_{k+1}‖: passing a large
        # g_new must not change the branch.
        @test update_radius!(r, 2.0, 0.95, true, η1, η2, 2.0, 1.0, 99.0) == 1.0
    end

    @testset "RGrad: μ is the radius-to-criticality ratio" begin
        r = RGrad(μ = 1.0, γ3 = 2.0)
        @test initial_radius(r, 99.0, 3.0) == 3.0        # Δ0 ignored
        Δ = update_radius!(r, 1.0, 0.95, true, η1, η2, 0.9, 1.0, 2.0)
        @test r.μ == 2.0 && Δ == 4.0
        reset_rule!(r)
        @test r.μ == 1.0
    end

    @testset "RGrad: the four branches of Update (R-grad)" begin
        # μ ← γ1μ below η1; γ2μ on [η1, η2); γ3μ above η2 when ‖s‖ > ½Δ; and μ
        # unchanged otherwise. The γ2 branch is what makes the climb of μ
        # non-monotone, and it is the one an earlier implementation omitted —
        # the boundedness argument for μ is stated for all four.
        cases = [(0.05, 0.9, 0.25),    # ρ < η1              → γ1
                 (0.50, 0.9, 0.50),    # η1 ≤ ρ < η2         → γ2
                 (0.95, 0.9, 2.00),    # ρ ≥ η2, ‖s‖ > ½Δ    → γ3
                 (0.95, 0.4, 1.00)]    # ρ ≥ η2, ‖s‖ ≤ ½Δ    → unchanged
        for (ρ, s_norm, expected) in cases
            r = RGrad(μ = 1.0, γ1 = 0.25, γ2 = 0.5, γ3 = 2.0)
            update_radius!(r, 1.0, ρ, ρ >= η, η1, η2, s_norm, 1.0, 1.0)
            @test r.μ == expected
        end
    end

    @testset "RGrad: uncapped μ crosses any threshold" begin
        r = RGrad(μ = 1e-6, γ3 = 2.0)
        for _ in 1:60
            update_radius!(r, 1.0, 0.95, true, η1, η2, 0.9, 1.0, 1.0)
        end
        @test r.μ > 1e6      # geometric growth is unbounded
    end

    @testset "RGradCapped: μ respects μ_max" begin
        r = RGradCapped(μ = 1.0, μ_max = 4.0, γ3 = 2.0)
        for _ in 1:20
            update_radius!(r, 1.0, 0.95, true, η1, η2, 0.9, 1.0, 1.0)
        end
        @test r.μ == 4.0
    end

    @testset "RGrad: the ½Δ guard" begin
        r = RGrad(μ = 1.0, γ3 = 2.0)
        update_radius!(r, 1.0, 0.95, true, η1, η2, 0.4, 1.0, 1.0)   # ‖s‖ < ½Δ
        @test r.μ == 1.0                                            # no expansion
        update_radius!(r, 1.0, 0.95, true, η1, η2, 0.6, 1.0, 1.0)   # ‖s‖ > ½Δ
        @test r.μ == 2.0
    end

    @testset "Hei factor: the limits and the jump at η1" begin
        # The normalisation of Part III: R(η1) = γ3 and lim_{t→+∞} R = M, with
        # 0 < γ1 ≤ γ2 < 1 < γ3 < M. It replaces the earlier one, in which
        # R(η1) = 1 + γ2 and γ3 was the asymptote.
        γ1, γ2, γ3, M, λ₁, λ₂ = 0.0625, 0.5, 1.5, 4.0, 5.0, 5.0
        R(t) = TrustRegionRadius._r_exp(t, η1, γ1, γ2, γ3, M, λ₁, λ₂)

        @test R(-1e3) ≈ γ1 atol = 1e-9          # lim_{t→−∞} R = γ1
        @test R(η1)   ≈ γ3                      # R(η1) = γ3
        @test R(1e3)  ≈ M  atol = 1e-9          # lim_{t→+∞} R = M

        # Below the threshold the factor is a contraction, above it an expansion.
        # The discontinuity at η1 is required by those two conditions, not a
        # defect: no continuous function satisfies both.
        left  = R(η1 - 1e-9)
        right = R(η1 + 1e-9)
        @test γ1 <= left < 1 < γ3 <= right <= M
        @test left ≈ γ2 atol = 1e-6
        @test right > left

        # Non-decreasing on each side, and across the join.
        @test issorted([R(t) for t in range(-2.0, η1 - 1e-6; length = 50)])
        @test issorted([R(t) for t in range(η1, 2.0; length = 50)])
        @test issorted([R(t) for t in range(-2.0, 2.0; length = 400)])

        # -Inf is the solver's convention for a non-positive predicted reduction
        # and for a non-finite trial objective: the most aggressive contraction.
        @test R(-Inf) == γ1
    end

    @testset "retrospective rules are flagged" begin
        @test needs_retrospective(RRTR())
        @test needs_retrospective(RRTRGrad())
        @test !needs_retrospective(RDelta())
        @test !needs_retrospective(RGrad())
    end

    @testset "retrospective rules branch on `accepted`, not on η̃₁" begin
        # On a rejected step ρ̃ is undefined — there is no new model — and the
        # solver passes ρ_k in that slot. Comparing it against η̃₁ used to return
        # Δ_{k+1} = Δ_k for ρ ∈ [η̃₁, η), leaving the radius, the model and the
        # iterate all unchanged: an identical subproblem, re-solved for ever, and
        # an unsuccessful iteration with no contraction.
        ρ_stuck = 0.07                          # η̃₁ = 0.05 ≤ 0.07 < η = 0.1
        Δ, s_norm = 1.0, 0.8

        r = RRTR(γ1 = 0.0625, γ2 = 0.25, γ3 = 2.5)
        @test 0.05 <= ρ_stuck < η                # the window is non-empty
        Δnew = update_radius!(r, Δ, ρ_stuck, false, η1, η2, s_norm, 2.0, 2.0)
        @test Δnew == r.γ2 * s_norm
        @test Δnew < Δ

        rg = RRTRGrad(μ = 1.0, γ1 = 0.25)
        Δg = update_radius!(rg, Δ, ρ_stuck, false, η1, η2, s_norm, 2.0, 2.0)
        @test rg.μ == 0.25
        @test Δg < rg.μ0 * 2.0

        # Accepted with a good ρ̃: step-driven and non-decreasing at once, which
        # is what buys RRTR eventual inactivity without a condition on a
        # user-chosen constant.
        acc = RRTR()
        @test update_radius!(acc, Δ, 0.99, true, η1, η2, s_norm, 2.0, 2.0) >= Δ
    end

    @testset "retrospective_ratio" begin
        s = [1.0, 0.0]; g_new = [-1.0, 0.0]; Hs = [1.0, 0.0]
        # predicted = -(-1) + 0.5(1) = 1.5
        @test retrospective_ratio(1.5, s, g_new, Hs) ≈ 1.0
        # non-positive predicted reduction => most conservative branch
        @test retrospective_ratio(1.0, s, [1.0, 0.0], [-4.0, 0.0]) == -Inf
    end
end
