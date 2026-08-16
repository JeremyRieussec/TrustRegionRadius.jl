# The diagnostics layer: the thresholds the theory names, the branch a rule took,
# and the hypotheses the local results assume. Nothing here is about speed; it is
# about whether a number reported from a run means what the paper says it means.

@testset "diagnostics" begin
    rosen(x) = (1 - x[1])^2 + 100(x[2] - x[1]^2)^2
    quad(x)  = 0.5 * (x[1]^2 + 4 * x[2]^2)

    # ---------------------------------------------------------------------
    @testset "kappa_bar" begin
        # The default is :neighbourhood, κ̄ = 8/λ*_min, which is the convention
        # Part III is written in. Pin the default explicitly rather than only
        # pinning the two named conventions: the whole point of the default is
        # that a caller who names nothing agrees with the paper.
        @test kappa_bar(0.5) ≈ 16.0
        @test kappa_bar(0.5; convention = :neighbourhood) ≈ 16.0
        @test kappa_bar(0.5; convention = :eigenvalue) ≈ 8.0
        # and they still differ by exactly the removable factor of two
        @test kappa_bar(0.5; convention = :neighbourhood) ==
              2 * kappa_bar(0.5; convention = :eigenvalue)
        @test kappa_bar(0.5) == 2 * kappa_bar(0.5; convention = :eigenvalue)

        @test_throws ArgumentError kappa_bar(0.5; convention = :whatever)
        @test_throws ArgumentError kappa_bar(0.0)
        @test_throws ArgumentError kappa_bar(-1.0)

        # From a problem and a solution: f = ½(x₁² + 4x₂²) has ∇²f = diag(1,4),
        # so λ*_min = 1, and κ̄ = 8 in the default convention, 4 in the other.
        nlp = ADNLPModel(quad, [1.0, 1.0])
        @test kappa_bar(nlp, [0.0, 0.0]) ≈ 8.0 atol = 1e-8
        @test kappa_bar(nlp, [0.0, 0.0]; convention = :neighbourhood) ≈ 8.0 atol = 1e-8
        @test kappa_bar(nlp, [0.0, 0.0]; convention = :eigenvalue) ≈ 4.0 atol = 1e-8
        # the default must agree with the named convention on both methods
        @test kappa_bar(nlp, [0.0, 0.0]) ==
              kappa_bar(nlp, [0.0, 0.0]; convention = :neighbourhood)
    end

    # ---------------------------------------------------------------------
    @testset "last_branch reports the branch that fired" begin
        η1, η2 = 0.1, 0.9

        r = RDelta()
        @test last_branch(r) === :none
        update_radius!(r, 1.0, 0.95, true, η1, η2, 0.5, 1.0, 1.0)
        @test last_branch(r) === :expand
        update_radius!(r, 1.0, 0.5, true, η1, η2, 0.5, 1.0, 1.0)
        @test last_branch(r) === :shrink
        update_radius!(r, 1.0, 0.0, false, η1, η2, 0.5, 1.0, 1.0)
        @test last_branch(r) === :contract

        r = RStep()
        update_radius!(r, 1.0, 0.95, true, η1, η2, 0.5, 1.0, 1.0)
        @test last_branch(r) === :expand

        # RDFO: the middle branch is the criticality test Δ > ζ·crit firing.
        r = RDFO(ζ = 1.0)
        update_radius!(r, 10.0, 0.95, true, η1, η2, 1.0, 1.0, 1.0)
        @test last_branch(r) === :shrink
        update_radius!(r, 0.5, 0.95, true, η1, η2, 0.5, 1.0, 1.0)
        @test last_branch(r) === :expand

        # RGrad: the ½Δ guard decides between growing and holding.
        r = RGrad(μ = 1.0)
        update_radius!(r, 1.0, 0.95, true, η1, η2, 0.9, 1.0, 1.0)
        @test last_branch(r) === :expand
        update_radius!(r, 1.0, 0.95, true, η1, η2, 0.1, 1.0, 1.0)
        @test last_branch(r) === :hold

        # RGradCapped: the cap refusing a requested climb is its own branch, and
        # it is the mechanism by which μ_max < κ̄ traps a run.
        r = RGradCapped(μ = 1.0, μ_max = 1.0)
        update_radius!(r, 1.0, 0.95, true, η1, η2, 0.9, 1.0, 1.0)
        @test last_branch(r) === :expand_capped
        @test r.μ == 1.0
        r = RGradCapped(μ = 1.0, μ_max = 8.0)
        update_radius!(r, 1.0, 0.95, true, η1, η2, 0.9, 1.0, 1.0)
        @test last_branch(r) === :expand
        @test r.μ ≈ 2.0

        # RRTR branches on ρ̃, and on a rejected step on ρ̃ alone.
        r = RRTR()
        update_radius!(r, 1.0, 0.95, true, η1, η2, 0.5, 1.0, 1.0)
        @test last_branch(r) === :expand
        update_radius!(r, 1.0, 0.5, true, η1, η2, 0.5, 1.0, 1.0)
        @test last_branch(r) === :hold
        update_radius!(r, 1.0, 0.01, true, η1, η2, 0.5, 1.0, 1.0)
        @test last_branch(r) === :shrink
        update_radius!(r, 1.0, -1.0, false, η1, η2, 0.5, 1.0, 1.0)
        @test last_branch(r) === :contract

        # The Hei family has a continuous factor, so the branch is its position
        # relative to one.
        for r in (RAdaptiveStep(), RAdaptiveGrad(), RRTRGrad())
            update_radius!(r, 1.0, 0.95, true, η1, η2, 0.9, 1.0, 1.0)
            @test last_branch(r) in (:expand, :hold, :contract)
        end

        # reset clears it, along with the multiplier.
        r = RGrad(μ = 1.0)
        update_radius!(r, 1.0, 0.95, true, η1, η2, 0.9, 1.0, 1.0)
        reset_rule!(r)
        @test last_branch(r) === :none
        @test r.μ == 1.0
    end

    # ---------------------------------------------------------------------
    @testset "SubWorkspace keeps its four-buffer constructor" begin
        v = zeros(3)
        ws = SubWorkspace(copy(v), copy(v), copy(v), copy(v))
        @test ws.iters == 0
        ws2 = SubWorkspace(v)
        @test ws2.iters == 0
    end

    # ---------------------------------------------------------------------
    @testset "trace alignment" begin
        nlp = ADNLPModel(rosen, [-1.2, 1.0])
        stats = tr_solve(nlp; rule = RDelta(), model = ExactHessian(),
                         subsolver = SteihaugCG(),
                         params = TRParams(tol = 1e-8, max_iterations = 200),
                         trace = true)
        ss = stats.solver_specific
        Δ = ss[:delta_trajectory]; s = ss[:step_trajectory]

        # The state trajectories carry one entry more than the per-iteration
        # ones, and they agree at the HEAD: Δ[j] and s[j] both describe iteration
        # j-1. Aligning on the tail shifts every plot by one.
        @test length(Δ) == length(s) + 1
        for key in (:ratio_trajectory, :active_trajectory, :accepted_trajectory,
                    :rho_tilde_trajectory, :gamma_trajectory, :xi_trajectory,
                    :cos_cauchy_trajectory, :cg_iters_trajectory,
                    :branch_trajectory)
            @test length(ss[key]) == length(s)
        end
        @test length(ss[:grad_trajectory]) == length(Δ)
        @test length(theta_trajectory(stats)) == length(Δ)

        # A radius can never be shorter than the step taken inside it, and this
        # inequality is the one that fails if the alignment is off by one.
        g = ss[:grad_trajectory]
        @test all(s[j] <= Δ[j] * (1 + 1e-10) for j in eachindex(s))

        # Not requested, so not attached: absence is unambiguous.
        @test !haskey(ss, :dist_trajectory)
        @test !haskey(ss, :lambda_min_true_trajectory)

        θ = theta_trajectory(stats)
        @test all(isnan(θ[j]) || θ[j] ≈ Δ[j] / g[j] for j in eachindex(θ))
    end

    # ---------------------------------------------------------------------
    @testset "the recorded hypotheses are the right quantities" begin
        # On a quadratic with the exact Hessian, y_k = ∇²f·s_k identically, so
        # the Dennis-Moré residual is zero to rounding. If γ is ever computed
        # after `update_model!`, or from the wrong gradient, this fails.
        nlp = ADNLPModel(quad, [3.0, -2.0])
        stats = tr_solve(nlp; rule = RDelta(), model = ExactHessian(),
                         subsolver = SteihaugCG(),
                         params = TRParams(tol = 1e-10, max_iterations = 100),
                         trace = true)
        γ = stats.solver_specific[:gamma_trajectory]
        acc = stats.solver_specific[:accepted_trajectory]
        for j in eachindex(γ)
            acc[j] || continue
            @test isfinite(γ[j])
            @test γ[j] < 1e-8
        end

        # ξ is a ratio of norms: non-negative wherever it is defined.
        ξ = stats.solver_specific[:xi_trajectory]
        @test all(x -> isnan(x) || x >= 0, ξ)

        # cos(s, -g) lies in [-1, 1] and is 1 when CG truncates on its first
        # iteration, which a small cap forces.
        c = stats.solver_specific[:cos_cauchy_trajectory]
        @test all(x -> isnan(x) || -1 - 1e-9 <= x <= 1 + 1e-9, c)

        rep = hypotheses_report(stats)
        @test 0 <= rep.cauchy_fraction <= 1
        @test isnan(rep.gamma_final) || rep.gamma_final < 1e-8
    end

    @testset "a small cap degenerates to the Cauchy point" begin
        # RGradCapped with μ_max small enough that the first CG iterate leaves
        # the region: the step is then along -g and the model Hessian has not
        # influenced the direction.
        nlp = ADNLPModel(rosen, [-1.2, 1.0])
        stats = tr_solve(nlp; rule = RGradCapped(μ = 1e-6, μ_max = 1e-6),
                         model = ExactHessian(), subsolver = SteihaugCG(),
                         params = TRParams(tol = 1e-6, max_iterations = 50),
                         trace = true)
        ss = stats.solver_specific
        @test all(==(1), ss[:cg_iters_trajectory])
        @test all(x -> abs(x - 1) < 1e-10, ss[:cos_cauchy_trajectory])
        @test hypotheses_report(stats).cauchy_fraction ≈ 1.0
    end

    # ---------------------------------------------------------------------
    @testset "inactivity and the derived quantities" begin
        nlp = ADNLPModel(quad, [3.0, -2.0])
        stats = tr_solve(nlp; rule = RDelta(), model = ExactHessian(),
                         subsolver = SteihaugCG(),
                         params = TRParams(Δ0 = 10.0, tol = 1e-10),
                         trace = true)
        ki = inactivity_index(stats)
        @test ki isa Int                      # the region does stop binding
        @test 0 <= active_fraction(stats) <= 1

        counts = branch_counts(stats)
        @test sum(values(counts)) == length(stats.solver_specific[:step_trajectory])
        @test all(k -> k in (:contract, :shrink, :expand, :expand_capped, :hold),
                  keys(counts))

        # κ̄ measured against κ̄ predicted. The empirical constant is the largest
        # realised ‖s‖/‖g‖ on inactive iterations, and the theory's 4/λ*_min is
        # an upper bound on it asymptotically.
        #
        # :eigenvalue named explicitly, not left to the default. The sharp bound
        # is the attained one, 4/λ*_min; the default is now :neighbourhood, and
        # taking it here would double κ and make this assertion strictly weaker
        # without anything in the line saying so.
        κ = kappa_bar(nlp, [0.0, 0.0]; convention = :eigenvalue)
        κe = kappa_bar_empirical(stats)
        @test isnan(κe) || κe <= κ + 1e-6

        # Too few points after inactivity to fit an order: NaN, not a number.
        o = observed_order(stats; last = 5)
        @test isnan(o.order) || isfinite(o.order)
        @test o.npts >= 0
    end

    # ---------------------------------------------------------------------
    @testset "the deterministic trace carries nothing stochastic" begin
        nlp = ADNLPModel(quad, [3.0, -2.0])
        stats = tr_solve(nlp; params = TRParams(tol = 1e-10), trace = true)
        ss = stats.solver_specific
        for key in (:sigma_g2_trajectory, :sigma_f2_trajectory,
                    :true_grad_trajectory, :paired_decrease_trajectory,
                    :paired_variance_trajectory, :grad_sample_trajectory)
            @test !haskey(ss, key)
        end
    end

    # ---------------------------------------------------------------------
    @testset "paired differences and CertifiedDecrease" begin
        # A finite sum whose terms differ: F_i(x) = ½‖x‖² + i·x₁/M, so the
        # per-observation values are distinct and the paired difference has a
        # variance that is genuinely O(‖s‖²).
        M = 40
        Fi  = (i, x) -> 0.5 * (x[1]^2 + x[2]^2) + i * x[1] / M
        Gi! = (i, x, g) -> (g[1] = x[1] + i / M; g[2] = x[2]; nothing)
        Hi  = (i, x) -> [1.0 0.0; 0.0 1.0]
        prob = FiniteSum(2, M, Fi, Gi!, Hi)

        @test supports_paired(prob)
        v = obs_objective(prob, [1.0, 1.0], collect(1:M))
        @test length(v) == M
        @test v ≈ [Fi(i, [1.0, 1.0]) for i in 1:M]

        # Pairing is the whole point: on the same batch the variance of the
        # difference vanishes with the step, while two independent evaluations
        # keep the O(1) spread of F itself.
        x  = [1.0, 1.0]
        for h in (1e-1, 1e-2, 1e-3)
            xc = x .+ [h, 0.0]
            d  = obs_objective(prob, x, collect(1:M)) .-
                 obs_objective(prob, xc, collect(1:M))
            @test std(d) < 2h            # O(‖s‖), not O(1)
        end
        @test std(obs_objective(prob, x, collect(1:M))) > 0.1

        # The rule: N = ⌈z²σ̂²/δ̂²⌉ once a paired statistic has been recorded.
        r = CertifiedDecrease(p = 0.9, N_min = 2)
        stt = SamplingState(3, 1.0, 1.0, 0.0, 0.0)
        @test grad_sample_size(r, stt) >= 2          # nothing recorded yet
        record_paired!(r, 1.0, 4.0, 10)
        stt2 = SamplingState(4, 1.0, 1.0, 0.0, 0.0)
        @test grad_sample_size(r, stt2) == ceil(Int, r.z^2 * 4.0)
        # Halving the decrease quadruples the demand: the ‖g‖⁻² scaling.
        record_paired!(r, 0.5, 4.0, 10)
        stt3 = SamplingState(5, 1.0, 1.0, 0.0, 0.0)
        @test grad_sample_size(r, stt3) == ceil(Int, r.z^2 * 4.0 / 0.25)
        # A failure to certify grows the batch rather than shrinking it.
        record_paired!(r, -1.0, 4.0, 10)
        stt4 = SamplingState(6, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, NaN, 16)
        @test grad_sample_size(r, stt4) > 16
        # Cached within an iteration: grad and obj must agree.
        @test grad_sample_size(r, stt4) == obj_sample_size(r, stt4)

        @test needs_paired(r)
        @test !needs_paired(NormTest())
        @test record_paired!(NormTest(), 1.0, 1.0, 8) === nothing

        @test_throws ArgumentError CertifiedDecrease(p = 0.4)
        @test_throws ArgumentError CertifiedDecrease(growth = 1.0)
    end

    @testset "a sampled run records the stochastic half" begin
        M = 60
        Fi  = (i, x) -> 0.5 * (x[1]^2 + x[2]^2) + i * x[1] / M
        Gi! = (i, x, g) -> (g[1] = x[1] + i / M; g[2] = x[2]; nothing)
        Hi  = (i, x) -> [1.0 0.0; 0.0 1.0]
        prob = FiniteSum(2, M, Fi, Gi!, Hi)
        nlp = FiniteSumNLP(prob, CertifiedDecrease(p = 0.9, N_min = 4);
                           x0 = [2.0, 2.0], seed = 1)
        stats = tr_solve(nlp; rule = RDelta(),
                         params = TRParams(tol = 1e-6, max_iterations = 40),
                         trace = true)
        ss = stats.solver_specific
        @test haskey(ss, :paired_decrease_trajectory)
        @test haskey(ss, :sigma_g2_trajectory)
        @test length(ss[:paired_decrease_trajectory]) ==
              length(ss[:step_trajectory])
        # The deterministic half is still there and still aligned.
        @test length(ss[:delta_trajectory]) == length(ss[:step_trajectory]) + 1
    end

    # ---------------------------------------------------------------------
    @testset "reference solution and true curvature are opt-in" begin
        nlp = ADNLPModel(quad, [3.0, -2.0])
        stats = tr_solve(nlp; rule = RDelta(), model = ExactHessian(),
                         params = TRParams(tol = 1e-10),
                         x_ref = [0.0, 0.0], true_curvature = true, trace = true)
        ss = stats.solver_specific
        @test haskey(ss, :dist_trajectory)
        @test haskey(ss, :lambda_min_true_trajectory)
        d = ss[:dist_trajectory]
        @test length(d) == length(ss[:delta_trajectory])
        @test d[1] ≈ norm([3.0, -2.0])
        @test d[end] < 1e-6
        # ∇²f ≡ diag(1,4) everywhere, so the true leftmost eigenvalue is 1.
        @test all(x -> isnan(x) || isapprox(x, 1.0; atol = 1e-6),
                  ss[:lambda_min_true_trajectory])

        @test_throws ArgumentError DeterministicTRSolver(nlp; x_ref = [0.0, 0.0, 0.0])
    end
end
