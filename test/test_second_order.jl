# The second-order layer: τ = max{‖g‖, −λ_min(B)}, the SecondOrder wrapper,
# curvature estimation, the eigenpoint, and the second-order stopping test.
#
# The centrepiece is the last testset: near an exact saddle a ‖g‖-anchored rule
# reports the radius it would use at a solution, and the two anchored families
# fail in different ways. That is the proposition the whole layer exists for, so
# it is checked rather than asserted.

@testset "second order" begin

    # A quartic with a saddle at the origin: f = x⁴/4 − x²/2 + y²/2 has critical
    # points (0,0) — a saddle, λ_min = −1 — and (±1,0) — minimisers.
    saddle_f(v) = v[1]^4/4 - v[1]^2/2 + v[2]^2/2

    @testset "τ = max{‖g‖, −λ_min}" begin
        @test tau_criticality(3.0, 5.0)   == 3.0    # positive curvature ignored
        @test tau_criticality(3.0, -5.0)  == 5.0    # curvature dominates
        @test tau_criticality(0.0, -2.0)  == 2.0    # the saddle case: τ > 0 = ‖g‖
        @test tau_criticality(0.0,  2.0)  == 0.0    # a genuine minimiser
        @test tau_criticality(4.0, -4.0)  == 4.0
        # τ ≥ ‖g‖ always, with equality exactly when B ⪰ 0.
        for g in (0.0, 1e-8, 1.0), λ in (-3.0, -1e-9, 0.0, 1e-9, 3.0)
            @test tau_criticality(g, λ) >= g
            λ >= 0 && @test tau_criticality(g, λ) == g
        end
    end

    @testset "the wrapper forwards everything but the measure" begin
        inner = RGrad(μ = 2.0, γ1 = 0.25, γ2 = 0.5, γ3 = 2.0)
        r = SecondOrder(inner)

        @test r isa RadiusRule
        @test needs_curvature(r) && !needs_curvature(inner)
        @test criticality(r, 1.0, -4.0) == 4.0
        @test criticality(inner, 1.0, -4.0) == 1.0

        # traits pass through
        @test asymptotic_regime(r)      === asymptotic_regime(inner)
        @test needs_retrospective(r)    == needs_retrospective(inner)
        @test is_criticality_anchored(r)
        @test r.μ == 2.0                       # field forwarding
        @test r.inner === inner

        # The update is the inner rule's, argument for argument.
        a = SecondOrder(RGrad(μ = 1.0)); b = RGrad(μ = 1.0)
        for (ρ, s, c) in ((0.05, 0.9, 2.0), (0.5, 0.9, 2.0), (0.95, 0.9, 2.0),
                          (0.95, 0.1, 2.0))
            @test update_radius!(a, 1.0, ρ, ρ >= 0.1, 0.1, 0.9, s, 2.0, c) ==
                  update_radius!(b, 1.0, ρ, ρ >= 0.1, 0.1, 0.9, s, 2.0, c)
            @test a.μ == b.μ
        end
    end

    @testset "only anchored rules may be wrapped" begin
        # RDelta and RStep never read the criticality slot, so wrapping them
        # would be a silent no-op.
        @test_throws ArgumentError SecondOrder(RDelta())
        @test_throws ArgumentError SecondOrder(RStep())
        @test_throws ArgumentError SecondOrder(RAdaptiveStep())
        for r in (RGrad(), RGradCapped(μ_max = 4.0), RDFO(),
                  RAdaptiveGrad(), RRTRGrad())
            @test SecondOrder(r) isa SecondOrder
        end
    end

    @testset "aliases" begin
        @test RGradTau(μ = 2.0)        isa SecondOrder{<:RGrad}
        @test RDFOTau(ζ = 0.5)         isa SecondOrder{<:RDFO}
        @test RGradCappedTau(μ_max = 4.0) isa SecondOrder{<:RGradCapped}
        @test RGradTau(μ = 2.0).μ == 2.0
        @test RDFOTau(ζ = 0.5).ζ  == 0.5
        # The factor convention is still enforced through the wrapped ctor.
        @test_throws ArgumentError RGradTau(γ2 = 2.0)
    end

    @testset "λ_min: dense and Lanczos agree" begin
        nlp = ADNLPModel(saddle_f, [0.0, 0.0], name = "saddle")
        λ = lambda_min_estimate(ExactHessian(), nlp, [0.0, 0.0])
        @test λ ≈ -1.0 rtol = 1e-10             # ∇²f(0) = diag(-1, 1)
        λ2, v = lambda_min_estimate(ExactHessian(), nlp, [0.0, 0.0]; vector = true)
        @test λ2 ≈ λ
        @test abs(abs(v[1]) - 1.0) < 1e-8       # the negative direction is e₁

        # Lanczos on a known spectrum. `nmax = 0` forces the iterative branch.
        n = 60
        D = Diagonal(collect(range(-2.0, 5.0; length = n)))
        λl = TrustRegionRadius._lanczos_min(D, n; k = 40)
        @test λl ≈ -2.0 rtol = 1e-6
        λv, vv = TrustRegionRadius._lanczos_min(D, n; k = 40, vector = true)
        @test λv ≈ λl
        @test norm(D * vv - λv * vv) < 1e-5 * norm(vv)
        # A Ritz value is an upper bound on λ_min, never below it.
        @test TrustRegionRadius._lanczos_min(D, n; k = 5) >= -2.0 - 1e-9
    end

    @testset "EigenPoint exploits negative curvature" begin
        nlp = ADNLPModel(saddle_f, [0.0, 0.0], name = "saddle")
        x = [0.0, 0.0]; g = [0.0, 0.0]; Δ = 0.5
        s = zeros(2); Hbuf = zeros(2)

        # At the saddle g = 0, so CG stops immediately with the zero step.
        solve_subproblem!(SteihaugCG(), ExactHessian(), nlp, x, g, Δ, s, Hbuf)
        @test norm(s) == 0                       # no progress available to CG

        # The eigenpoint moves to the boundary along the negative direction and
        # decreases the model by at least ½|λ|Δ².
        fill!(s, 0.0)
        active = solve_subproblem!(EigenPoint(SteihaugCG()), ExactHessian(),
                                   nlp, x, g, Δ, s, Hbuf)
        @test active
        @test norm(s) ≈ Δ rtol = 1e-8
        B = dense_hessian(ExactHessian(), nlp, x)
        decrease = -dot(g, s) - 0.5 * dot(s, B * s)
        @test decrease >= 0.5 * 1.0 * Δ^2 - 1e-10

        # With positive definite curvature the inner step is returned untouched.
        nlp2 = ADNLPModel(v -> 0.5 * (v[1]^2 + v[2]^2), [1.0, 1.0])
        x2 = [1.0, 1.0]; g2 = [1.0, 1.0]
        s_in = zeros(2); s_ep = zeros(2)
        solve_subproblem!(SteihaugCG(), ExactHessian(), nlp2, x2, g2, 10.0, s_in, Hbuf)
        solve_subproblem!(EigenPoint(SteihaugCG()), ExactHessian(), nlp2, x2, g2,
                          10.0, s_ep, Hbuf)
        @test s_in ≈ s_ep
    end

    @testset "second_order_status" begin
        @test second_order_status(1e-9, 0.5, 1e-6, 1e-6) === :second_order
        @test second_order_status(1e-9, -1e-9, 1e-6, 1e-6) === :second_order
        @test second_order_status(1e-9, -1.0, 1e-6, 1e-6) === :first_order
        @test second_order_status(1.0, 0.5, 1e-6, 1e-6) === :unknown
    end

    @testset "TRParams: tol_H" begin
        @test TRParams().tol_H == -1                 # disabled by default
        @test TRParams(tol_H = 1e-6).tol_H == 1e-6
        @test_throws ArgumentError TRParams(tol_H = 0.0)
        @test_throws ArgumentError TRParams(tol_H = -2.0)
    end

    @testset "the trace carries λ_min and τ" begin
        nlp = ADNLPModel(saddle_f, [0.5, 0.5])
        st = tr_solve(nlp; rule = RGradTau(μ = 1.0), model = ExactHessian(),
                      subsolver = EigenPoint(SteihaugCG()), trace = true,
                      params = TRParams(tol = 1e-8, tol_H = 1e-6,
                                        max_iterations = 5_000))
        ss = st.solver_specific
        @test haskey(ss, :lambda_min_trajectory)
        @test haskey(ss, :tau_trajectory)
        @test length(ss[:tau_trajectory]) == length(ss[:grad_trajectory])
        # τ ≥ ‖g‖ pointwise, by construction.
        @test all(ss[:tau_trajectory] .>= ss[:grad_trajectory] .- 1e-12)

        # A first-order rule pays for nothing it does not use.
        st1 = tr_solve(nlp; rule = RGrad(μ = 1.0), trace = true,
                       params = TRParams(tol = 1e-8))
        @test !haskey(st1.solver_specific, :lambda_min_trajectory)
    end

    @testset "second-order termination reaches a minimiser, not a saddle" begin
        # Started on the y-axis, the gradient stays on it: ∇f = (x³−x, y), so
        # x = 0 is invariant and every ‖g‖-based method converges to the saddle.
        # The second-order test refuses to stop there.
        nlp = ADNLPModel(saddle_f, [0.0, 0.7])

        first_order = tr_solve(nlp; rule = RGrad(μ = 1.0), model = ExactHessian(),
                               params = TRParams(tol = 1e-8, max_iterations = 2_000))
        @test first_order.status === :first_order
        @test norm(first_order.solution) < 1e-6          # stopped at the saddle

        second_order = tr_solve(nlp; rule = RGradTau(μ = 1.0), model = ExactHessian(),
                                subsolver = EigenPoint(SteihaugCG()),
                                params = TRParams(tol = 1e-8, tol_H = 1e-6,
                                                  max_iterations = 2_000))
        @test second_order.status === :second_order
        @test abs(abs(second_order.solution[1]) - 1.0) < 1e-5   # a real minimiser
        @test second_order.objective < first_order.objective
    end

    @testset "the ‖g‖ anchors fail at a saddle, the τ anchors do not" begin
        # Proposition (first-order anchor stalls). At an exact saddle ‖g‖ = 0, so
        # a ‖g‖-anchored rule sets the radius it would use at a solution:
        #   RGrad(‖g‖): Δ = μ·0 = 0 — halts outright;
        #   RDFO(‖g‖):  Δ > ζ·0 always — contracts geometrically to the floor.
        # Under τ both stay proportional to the curvature that is available.
        λ, gn = -1.0, 0.0                      # the exact saddle
        η1, η2 = 0.1, 0.9

        rg = RGrad(μ = 1.0)
        @test initial_radius(rg, 1.0, criticality(rg, gn, λ)) == 0.0     # dead

        rgτ = RGradTau(μ = 1.0)
        @test initial_radius(rgτ, 1.0, criticality(rgτ, gn, λ)) == 1.0   # alive

        # R-DFO: with ‖g‖ the "radius too large" branch fires at every iteration,
        # so the radius is multiplied by γ2 < 1 for ever.
        rd, rdτ = RDFO(ζ = 1.0, γ2 = 0.5), RDFOTau(ζ = 1.0, γ2 = 0.5)
        Δ, Δτ = 1.0, 1.0
        for _ in 1:40
            Δ  = update_radius!(rd,  Δ,  0.95, true, η1, η2, Δ,  gn, criticality(rd,  gn, λ))
            Δτ = update_radius!(rdτ, Δτ, 0.95, true, η1, η2, Δτ, gn, criticality(rdτ, gn, λ))
        end
        @test Δ < 1e-10                        # collapsed
        @test Δτ >= 1.0                        # held at the curvature scale
    end

    @testset "τ ≡ ‖g‖ over a positive definite model" begin
        # LBFGSModel enforces B ≻ 0, so λ_min > 0, so the second-order variant is
        # an expensive no-op. Worth knowing before running a grid over it.
        nlp = ADNLPModel(saddle_f, [0.5, 0.5])
        st = tr_solve(nlp; rule = RGradTau(μ = 1.0), model = LBFGSModel(mem = 5),
                      trace = true, params = TRParams(tol = 1e-8, max_iterations = 2_000))
        ss = st.solver_specific
        @test all(ss[:lambda_min_trajectory] .> 0)
        @test ss[:tau_trajectory] ≈ ss[:grad_trajectory]
    end
end
