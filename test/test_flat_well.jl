# The flat well, P2 of the manuscript: f(x,y) = ½x² + ε(¼y⁴ − ½y²), ε ∈ (0, ½).
#
# The one problem in the survey on which the radius thresholds are available in
# closed form rather than estimated. The minimisers sit at (0, ±1) with
# ∇²f = diag(1, 2ε), so λ* = 2ε and
#
#     RDFO trapped below   0.1/ε        exact   1/(2ε)        κ̄ = 4/ε
#
# and ε moves the curvature at the solution without moving the solution. Two
# dimensions and analytic, so this belongs in the suite rather than only in the
# benchmark.
#
# The sharpest assertion here is the rate identity ‖g_{k+1}‖ = ‖g_k‖(1 − λ*θ_k).
# It ties the recorded radius, the recorded gradient and the step together, so it
# fails on a wrong step, a wrong Δ, or a trajectory misaligned by one.

@testset "flat well" begin

    fw(ε; x0 = [0.0, 1.2]) =
        ADNLPModel(p -> 0.5p[1]^2 + ε * (p[2]^4 / 4 - p[2]^2 / 2), copy(float.(x0)),
                   name = "P2(eps=$ε)")

    γ1, γ2, γ3 = 0.25, 0.5, 2.0
    mk_capped(μ) = RGradCapped(γ1 = γ1, γ2 = γ2, γ3 = γ3, μ = min(1.0, μ), μ_max = μ)
    mk_grad(μ0)  = RGrad(γ1 = γ1, γ2 = γ2, γ3 = γ3, μ = μ0)
    mk_rdfo(ζ)   = RDFO(γ1 = γ1, γ2 = γ2, γ3 = γ3, ζ = ζ)

    # Same clean prefix the experiment uses: a trapped run does not terminate,
    # and once ared underflows every subsequent step is recorded active with
    # ‖s‖ = Δ, so activity measured over the whole run is the right answer for
    # the wrong reason.
    function clean_prefix(stats; ρlo = 0.9, ρhi = 1.1, gmin = 1e-12)
        ρ = stats.solver_specific[:ratio_trajectory]
        g = stats.solver_specific[:grad_trajectory]
        n = min(length(ρ), max(length(g) - 1, 0))
        for k in 1:n
            (isfinite(ρ[k]) && ρlo <= ρ[k] <= ρhi && g[k] > gmin) || return k - 1
        end
        return n
    end

    run_fw(rule, ε; x0 = [0.0, 1.2], kmax = 400) = begin
        nlp = fw(ε; x0 = x0)
        g0  = norm(grad(nlp, collect(float.(x0))))
        Δ0  = 0.5 * g0 / (ε * (3x0[2]^2 - 1))
        tr_solve(nlp; rule = rule, model = ExactHessian(), subsolver = SteihaugCG(),
                 params = TRParams(η = 0.1, η1 = 0.1, η2 = 0.9, Δ0 = Δ0,
                                   tol = 1e-10, max_iterations = kmax),
                 x_ref = [0.0, 1.0], trace = true)
    end

    @testset "the curvature at the minimiser is 2ε" begin
        for ε in (0.1, 0.01, 0.001)
            @test lambda_min_estimate(ExactHessian(), fw(ε), [0.0, 1.0]) ≈ 2ε rtol = 1e-10
        end
        # and ε < ½ is what makes 2ε the smaller eigenvalue rather than 1
        @test lambda_min_estimate(ExactHessian(), fw(0.9), [0.0, 1.0]) ≈ 1.0 rtol = 1e-10
    end

    @testset "kappa_bar is 4/ε, and 2/ε in the eigenvalue convention" begin
        for ε in (0.1, 0.01, 0.001)
            @test kappa_bar(fw(ε), [0.0, 1.0]) ≈ 4 / ε rtol = 1e-8
            @test kappa_bar(fw(ε), [0.0, 1.0]; convention = :eigenvalue) ≈ 2 / ε rtol = 1e-8
        end
    end

    @testset "the line x = 0 is invariant" begin
        # ∂f/∂x = x, so a run started on the line stays on it exactly, not
        # approximately: any drift means the step picked up a first component.
        xs = [[0.0, 1.2]]
        st = tr_solve(fw(0.05); rule = RDelta(), model = ExactHessian(),
                      subsolver = SteihaugCG(),
                      params = TRParams(tol = 1e-10, max_iterations = 400),
                      callback = (_n, _s, s) -> push!(xs, copy(s.solution)))
        @test all(p -> p[1] == 0.0, xs)
        @test length(xs) > 1
        @test st.solution[2] ≈ 1.0 atol = 1e-6      # the + branch, not the saddle
    end

    @testset "RGradCapped is trapped below the threshold and free above it" begin
        ε = 0.005                       # 1/(2ε) = 100
        trapped = run_fw(mk_capped(50.0), ε)
        m = clean_prefix(trapped)
        @test m > 0
        a = trapped.solver_specific[:active_trajectory][1:m]
        @test count(a) / m == 1.0                     # active throughout the clean prefix
        @test inactivity_index(trapped) === nothing

        free = run_fw(mk_capped(500.0), ε)            # 500 > 100
        @test inactivity_index(free) isa Int
    end

    @testset "RGrad uncapped always escapes" begin
        # The one unconditional result in the survey: μ climbs past any
        # threshold, so the constraint goes inactive whatever ε is.
        for ε in (0.2, 0.02, 0.002)
            k = inactivity_index(run_fw(mk_grad(1.0), ε))
            @test k isa Int
        end
    end

    @testset "the rate identity on the trapped run" begin
        # ‖g_{k+1}‖ = ‖g_k‖(1 − λ*θ_k) on a binding iteration. This ties Δ, ‖g‖
        # and the step together, so it fails on a wrong step, a wrong radius, or
        # a trajectory misaligned by one.
        #
        # It is ASYMPTOTIC, and the distinction matters. The exact local
        # statement uses the curvature at the iterate, φ''(y) = ε(3y² − 1), and
        # λ* = 2ε is its limit at y = 1. Measured on this run at ε = 0.005:
        #
        #   k    y_k       g'/g        1 − λ*θ     1 − φ''(y_k)θ
        #   0    1.200   0.983447     0.990000     0.983400
        #   9    1.008   0.490887     0.500000     0.487789
        #  23    1.000   0.4999995    0.5000000    0.4999993
        #
        # so the λ* form carries a 6.6e-3 relative error at k = 0 and only
        # reaches 1e-6 by k ≈ 23, while the φ'' form is a hundred times closer
        # early on. Asserting the λ* form at 1e-6 over every binding iteration
        # therefore fails 24 times out of 29, on a correct run. Assert it where
        # it is claimed, in the tail, and separately assert that it is genuinely
        # converging rather than merely small.
        ε = 0.005; λ = 2ε
        st = run_fw(mk_capped(50.0), ε)
        m  = clean_prefix(st)
        g  = st.solver_specific[:grad_trajectory]
        θ  = theta_trajectory(st)
        a  = st.solver_specific[:active_trajectory]
        d  = st.solver_specific[:dist_trajectory]

        binding = [k for k in 1:m if a[k] && λ * θ[k] < 1]
        @test length(binding) >= 10

        relerr(k, c) = abs(g[k + 1] / g[k] - (1 - c * θ[k])) / abs(1 - c * θ[k])

        # (a) the local identity, with the curvature at the iterate. The residual
        #     is the second-order term of φ' over the step, so it is O((hθ)²)
        #     rather than a fixed size. Assert that scaling law: it tightens
        #     automatically as the step shortens, which a fixed tolerance does
        #     not, and it is what the linearisation actually claims.
        local_checked = 0
        for k in binding
            h = ε * (3 * (1 + d[k])^2 - 1)
            hθ = h * θ[k]
            hθ <= 0.2 || continue
            @test relerr(k, h) < hθ^2
            local_checked += 1
        end
        @test local_checked >= 3

        # (b) the asymptotic identity with λ*, in the tail where it is claimed.
        #     Measured relative error on this run, by iteration:
        #       k = 23  1.09e-6      k = 25  2.71e-7      k = 27  5.88e-8
        #       k = 24  5.43e-7      k = 26  1.38e-7      k = 28  3.68e-8
        #     so it crosses 1e-6 at k = 24 and halves each step thereafter,
        #     which is the y_k → 1 rate. Three iterations deep into that regime.
        tail = binding[max(1, end - 2):end]
        for k in tail
            @test relerr(k, λ) < 1e-6
        end

        # (c) and it is converging: the error falls by orders of magnitude, so a
        #     run that merely sat at a fixed small error would not pass.
        @test relerr(binding[end], λ) < 1e-3 * relerr(binding[1], λ)
    end

    @testset "the escape is permanent" begin
        # Once the step is Newton, ‖g‖ falls quadratically while Δ only shrinks
        # geometrically, so θ → ∞ and the constraint cannot bind again.
        st = run_fw(mk_rdfo(1.0), 0.05; x0 = [0.0, 2.0])
        a  = st.solver_specific[:active_trajectory]
        i  = findfirst(!, a)
        @test i !== nothing                       # it does go inactive
        @test !any(a[i:end])                      # and never binds again
        @test inactivity_index(st) isa Int
    end
end
