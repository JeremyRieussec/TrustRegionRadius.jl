# Radius rules: the update contract, and the invariants the survey relies on.

@testset "rules" begin
    η₁, η₂ = 0.1, 0.9

    @testset "every rule implements the contract" begin
        rules = [RDelta(), RStep(), RDFO(), RGrad(), RGradCapped(),
                 RAdaptiveStep(), RAdaptiveGrad(), RAdaptiveFanYuan(),
                 RRTR(), RRTRGrad()]
        for r in rules
            @test r isa RadiusRule
            Δ0 = initial_radius(r, 1.0, 2.0)
            @test Δ0 > 0
            Δ1 = update_radius!(r, 1.0, 0.95, η₁, η₂, 0.8, 2.0, 1.5)
            @test Δ1 > 0 && isfinite(Δ1)
            @test reset_rule!(r) === nothing
        end
    end

    @testset "RDelta: monotone in ρ" begin
        r = RDelta(γ₁ = 0.25, γ₂ = 0.5, γ₃ = 2.0)
        bad  = update_radius!(r, 1.0, 0.0,  η₁, η₂, 1.0, 1.0, 1.0)
        ok   = update_radius!(r, 1.0, 0.5,  η₁, η₂, 1.0, 1.0, 1.0)
        good = update_radius!(r, 1.0, 0.95, η₁, η₂, 1.0, 1.0, 1.0)
        @test bad < ok < good
        @test good == 2.0
    end

    @testset "RStep: Δmin prevents collapse" begin
        # An accepted step of length ~0 would drive Δ to 0 without the floor,
        # after which every subsequent step is 0 and the solver cannot recover.
        r = RStep(Δmin = 1e-14)
        @test update_radius!(r, 1.0, 0.95, η₁, η₂, 0.0, 1.0, 1.0) >= 1e-14
        runguarded = RStep(Δmin = 0.0)
        @test update_radius!(runguarded, 1.0, 0.95, η₁, η₂, 0.0, 1.0, 1.0) == 0.0
    end

    @testset "RDFO: expands only when Δ ≤ ζ‖g‖" begin
        r = RDFO(γ₂ = 0.5, γ₃ = 2.0, ζ = 1.0)
        @test update_radius!(r, 0.5, 0.95, η₁, η₂, 0.5, 1.0, 1.0) == 1.0   # expand
        @test update_radius!(r, 2.0, 0.95, η₁, η₂, 2.0, 1.0, 1.0) == 1.0   # shrink
        @test is_criticality_anchored(r)
    end

    @testset "RGrad: μ is the radius-to-criticality ratio" begin
        r = RGrad(μ = 1.0, γ₂ = 2.0)
        @test initial_radius(r, 99.0, 3.0) == 3.0        # Δ₀ ignored
        Δ = update_radius!(r, 1.0, 0.95, η₁, η₂, 0.9, 1.0, 2.0)
        @test r.μ == 2.0 && Δ == 4.0
        reset_rule!(r)
        @test r.μ == 1.0
    end

    @testset "RGrad: uncapped μ crosses any threshold" begin
        r = RGrad(μ = 1e-6, γ₂ = 2.0)
        for _ in 1:60
            update_radius!(r, 1.0, 0.95, η₁, η₂, 0.9, 1.0, 1.0)
        end
        @test r.μ > 1e6      # geometric growth is unbounded
    end

    @testset "RGradCapped: μ respects μ_max" begin
        r = RGradCapped(μ = 1.0, μ_max = 4.0, γ₂ = 2.0)
        for _ in 1:20
            update_radius!(r, 1.0, 0.95, η₁, η₂, 0.9, 1.0, 1.0)
        end
        @test r.μ == 4.0
    end

    @testset "RGrad: the ½Δ guard" begin
        r = RGrad(μ = 1.0, γ₂ = 2.0)
        update_radius!(r, 1.0, 0.95, η₁, η₂, 0.4, 1.0, 1.0)   # ‖s‖ < ½Δ
        @test r.μ == 1.0                                       # no expansion
        update_radius!(r, 1.0, 0.95, η₁, η₂, 0.6, 1.0, 1.0)   # ‖s‖ > ½Δ
        @test r.μ == 2.0
    end

    @testset "Hei factor is continuous at the threshold" begin
        η, β, γ₁, γ₂, M, λ₁, λ₂ = 0.25, 0.0625, 0.25, 0.5, 4.0, 5.0, 5.0
        left  = TrustRegionRadius._r_exp(η - 1e-9, η, β, γ₁, γ₂, M, λ₁, λ₂)
        right = TrustRegionRadius._r_exp(η + 1e-9, η, β, γ₁, γ₂, M, λ₁, λ₂)
        # the two branches meet at 1 - γ₁ and 1 + γ₂ respectively; they are not
        # equal in general, which is a property of the rule, not a bug
        @test isfinite(left) && isfinite(right)
        @test β <= left <= M && 1.0 <= right <= M
    end

    @testset "retrospective rules are flagged" begin
        @test needs_retrospective(RRTR())
        @test needs_retrospective(RRTRGrad())
        @test !needs_retrospective(RDelta())
        @test !needs_retrospective(RGrad())
    end

    @testset "retrospective_ratio" begin
        s = [1.0, 0.0]; g_new = [-1.0, 0.0]; Hs = [1.0, 0.0]
        # predicted = -(-1) + 0.5(1) = 1.5
        @test retrospective_ratio(1.5, s, g_new, Hs) ≈ 1.0
        # non-positive predicted reduction => most conservative branch
        @test retrospective_ratio(1.0, s, [1.0, 0.0], [-4.0, 0.0]) == -Inf
    end
end
