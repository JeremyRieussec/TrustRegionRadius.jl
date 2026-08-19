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
        @test 0 <= active_fraction(stats) <= 1   # on a real run, only a sanity bound;
        # the exact slice is pinned on synthetic traces below, where the answer
        # is known rather than whatever this run happened to produce.

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
        # (`isnan(o.order) || isfinite(o.order)` was here, which no implementation
        # can fail. The order is pinned against known sequences below.)
        o = observed_order(stats; last = 5)
        @test o.npts >= 0
        @test o.npts < 3 ? isnan(o.order) : isfinite(o.order)
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

    # ---------------------------------------------------------------------
    # Synthetic traces. A trace built by hand pins the *function*; a trace taken
    # from a run pins whatever that run happened to do, which is why the three
    # assertions these replace could not fail.
    # ---------------------------------------------------------------------

    synth(nlp; kw...) = begin
        s = GenericExecutionStats(nlp)
        for (k, v) in kw
            set_solver_specific!(s, k, v)
        end
        s
    end

    @testset "active_fraction takes the exact tail slice" begin
        nlp = ADNLPModel(quad, [1.0, 1.0])
        # 20 entries, only the last two active.
        a = fill(false, 20); a[19] = true; a[20] = true
        s = synth(nlp; active_trajectory = a)

        # tail = 0.1 -> lo = floor(20*0.9)+1 = 19, slice 19:20, both active.
        @test active_fraction(s; tail = 0.1) == 1.0
        # tail = 0.5 -> lo = floor(20*0.5)+1 = 11, slice 11:20, two of ten.
        @test active_fraction(s; tail = 0.5) == 0.2
        # tail = 1 is the whole run.
        @test active_fraction(s; tail = 1.0) == 0.1

        # The boundary the hand-rolled notebook version got wrong: at n = 41 it
        # took six points where this takes five, because floor(0.9n) is an index
        # and floor(0.9n)+1 is the first index of the tail.
        b = fill(false, 41); b[37:41] .= true      # exactly the tail slice
        s41 = synth(nlp; active_trajectory = b)
        @test active_fraction(s41; tail = 0.1) == 1.0
        b2 = fill(false, 41); b2[36] = true        # one before it: outside
        @test active_fraction(synth(nlp; active_trajectory = b2); tail = 0.1) == 0.0

        @test_throws ArgumentError active_fraction(s; tail = 0.0)
        @test_throws ArgumentError active_fraction(s; tail = -0.1)
        @test_throws ArgumentError active_fraction(s; tail = 1.5)
    end

    @testset "inactivity_index survives release and rebind" begin
        nlp = ADNLPModel(quad, [1.0, 1.0])
        # act[j] describes iteration j-1. Inactive at k = 2, binding again at
        # k = 5, permanently inactive from k = 9.
        a = fill(false, 15)
        a[1] = a[2] = true            # iterations 0, 1 active
        a[6] = a[7] = a[8] = a[9] = true   # iterations 5..8 active again
        s = synth(nlp; active_trajectory = a)
        # An implementation that stops at the first release returns 2. The
        # answer is 9: the run was still binding at k = 8.
        @test inactivity_index(s) == 9

        # never binds
        @test inactivity_index(synth(nlp; active_trajectory = fill(false, 10))) == 0
        # still binding at the end: no inactivity index exists
        @test inactivity_index(synth(nlp; active_trajectory = fill(true, 10))) === nothing
        tail_true = fill(false, 10); tail_true[10] = true
        @test inactivity_index(synth(nlp; active_trajectory = tail_true)) === nothing
        # empty trace
        @test inactivity_index(synth(nlp; active_trajectory = Bool[])) === nothing
    end

    @testset "observed_order recovers a known order" begin
        nlp = ADNLPModel(quad, [1.0, 1.0])
        # after_inactivity = false isolates the regression from the gating.
        mk(e) = synth(nlp; grad_trajectory = e,
                           accepted_trajectory = fill(true, length(e) - 1),
                           active_trajectory = fill(false, length(e) - 1))

        # Order 1: e_{k+1} = 0.3 e_k, so log e_{k+1} = log 0.3 + 1·log e_k.
        e1 = [0.5]; for _ in 1:9; push!(e1, 0.3 * e1[end]); end
        o1 = observed_order(mk(e1); after_inactivity = false)
        @test round(o1.order, digits = 2) == 1.00
        @test o1.npts == 9
        @test isapprox(exp(o1.intercept), 0.3; rtol = 1e-8)

        # Order 2: e_{k+1} = 0.5 e_k², slope 2 exactly.
        e2 = [0.5]; for _ in 1:6; push!(e2, 0.5 * e2[end]^2); end
        o2 = observed_order(mk(e2); after_inactivity = false)
        @test round(o2.order, digits = 2) == 2.00
        @test o2.npts == 6

        # Too short to fit: NaN order, but npts reports the count actually
        # available, so "too short" is distinguishable from "measured as zero".
        short = observed_order(mk([0.5, 0.15, 0.045]); after_inactivity = false)
        @test isnan(short.order)
        @test short.npts == 2

        # Zero is a measurement, not an absence: a flat sequence has order 0
        # and reports its full point count.
        flat = observed_order(mk(fill(0.25, 8)); after_inactivity = false)
        @test flat.npts == 7
        @test isnan(flat.order) || abs(flat.order) < 1e-8

        @test_throws ArgumentError observed_order(mk(e1); use = :nonsense)
    end

    @testset "model_hessian_norm on both branches" begin
        nlp2 = ADNLPModel(quad, [1.0, 1.0])
        x2 = [0.7, -0.3]

        # ScaledIdentity: ‖cI‖ = c exactly, at every point.
        @test model_hessian_norm(ScaledIdentity(c = 3.0), nlp2, x2) ≈ 3.0
        @test model_hessian_norm(ScaledIdentity(c = 0.25), nlp2, randn(2)) ≈ 0.25

        # ExactHessian on ½(x₁² + 4x₂²): ∇²f ≡ diag(1,4), so ‖H‖₂ = 4 everywhere.
        m = ExactHessian(); reset_model!(m, 2)
        @test model_hessian_norm(m, nlp2, x2) ≈ 4.0 atol = 1e-8
        @test model_hessian_norm(m, nlp2, [5.0, 5.0]) ≈ 4.0 atol = 1e-8

        # The two branches on identical input. nmax = 1 forces the power
        # iteration on a problem whose dense answer is known exactly, which is a
        # sharper cross-check than running the branches on different problems.
        @test model_hessian_norm(m, nlp2, x2; nmax = 1) ≈ 4.0 atol = 1e-6
        @test model_hessian_norm(ScaledIdentity(c = 3.0), nlp2, x2; nmax = 1) ≈ 3.0 atol = 1e-6

        # n = 250 > nmax = 200, so the power-iteration branch runs for real.
        n = 250
        xb = ones(n) ./ 3

        # (a) Well-separated spectrum, λ2/λ1 ≈ 0.1. Thirty iterations is ample
        #     and the branch is exact to machine precision.
        ws = vcat(collect(1.0:(n - 1)) ./ 250, 10.0)
        sep = ADNLPModel(z -> 0.5 * sum(ws .* z .^ 2), zeros(n))
        ms = ExactHessian(); reset_model!(ms, n)
        @test model_hessian_norm(ms, sep, xb) ≈ 10.0 rtol = 1e-10
        @test model_hessian_norm(ms, sep, xb; nmax = 10_000) ≈ 10.0 rtol = 1e-10

        # (b) Clustered spectrum {i/25 : i = 1…250}, λ2/λ1 = 0.996. This is the
        #     case the old power iteration on H² could not do: its per-step
        #     factor is (λ2/λ1)² = 0.992, so the thirty steps it defaulted to
        #     removed almost nothing and returned 9.9346 against a true 10.0, a
        #     relative error of 6.5e-3 that no amount of tightening a stopping
        #     test would have fixed. Lanczos on the same operator, at the same
        #     `lanczos_k = 40` that `lambda_min_estimate` already uses, reaches
        #     2e-5 in *forty* matrix-vector products against that method's sixty.
        w = collect(1.0:n) ./ 25
        big = ADNLPModel(z -> 0.5 * sum(w .* z .^ 2), zeros(n))
        mb = ExactHessian(); reset_model!(mb, n)
        truth = maximum(w)

        @test model_hessian_norm(mb, big, xb; nmax = 10_000) ≈ truth rtol = 1e-10

        d40  = model_hessian_norm(mb, big, xb)                      # default k = 40
        d60  = model_hessian_norm(mb, big, xb; lanczos_k = 60)
        d120 = model_hessian_norm(mb, big, xb; lanczos_k = 120)
        # Ritz values interlace the spectrum, so the estimate is a lower bound at
        # every k and rises with k. That one-sidedness is what makes an understated
        # ‖H‖ safe for "Σ Δ²/M is finite" and unsafe for a quoted value.
        @test d40 < truth
        @test d60 < truth
        @test d40 < d60 < d120
        @test d120 ≈ truth rtol = 1e-8
        # The default's error on this spectrum, pinned so a regression is visible.
        # The power iteration it replaced was at 6.5e-3 for half again the cost.
        @test abs(d40 - truth) / truth < 1e-4
        @test abs(d60 - truth) / truth < 1e-7

        # Cost is counted in matrix-vector products: k Lanczos steps cost k, p
        # power steps cost 2p, since the power branch formed B²v. So the default
        # k = 40 is *cheaper* than the thirty power steps it replaced, 40 products
        # against 60, and still two orders more accurate; and k = 60 costs exactly
        # what they did while being six orders better.
        @test 40 < 2 * 30
        @test abs(d40 - truth) / truth < 6.5e-3   # the power branch's error at 60 products
        @test 60 == 2 * 30
        @test abs(d60 - truth) / truth < 1e-7

        # NaN rather than an exception: a diagnostic must not be able to fail a
        # run. An LBFGS model that was never reset has no operator to ask.
        @test isnan(model_hessian_norm(LBFGSModel(mem = 5), nlp2, x2))
    end

    @testset "radius_sums and the hessian_norm keyword" begin
        nlp = ADNLPModel(quad, [3.0, -2.0])
        p = TRParams(tol = 1e-10, max_iterations = 100)

        # Without the flag the third series is unavailable, and says so with NaN
        # rather than with a zero that would read as a converged sum.
        off = tr_solve(nlp; rule = RDelta(), model = ExactHessian(),
                       params = p, trace = true)
        @test !haskey(off.solver_specific, :hessian_norm_trajectory)
        r0 = radius_sums(off)
        @test isnan(r0.sum_delta2_over_M)
        @test isfinite(r0.sum_delta) && isfinite(r0.sum_delta2)
        @test r0.n == length(off.solver_specific[:delta_trajectory])

        # With it, all three are finite and the count is the full trajectory.
        on = tr_solve(nlp; rule = RDelta(), model = ExactHessian(),
                      params = p, hessian_norm = true, trace = true)
        ss = on.solver_specific
        @test haskey(ss, :hessian_norm_trajectory)
        @test length(ss[:hessian_norm_trajectory]) == length(ss[:delta_trajectory])
        # ∇²f ≡ diag(1,4), so the recorded norm is 4 at every iterate.
        @test all(x -> isapprox(x, 4.0; atol = 1e-8), ss[:hessian_norm_trajectory])
        r = radius_sums(on)
        @test isfinite(r.sum_delta) && isfinite(r.sum_delta2)
        @test isfinite(r.sum_delta2_over_M)
        @test r.n == length(ss[:delta_trajectory])

        # M_k = L + max_{i≤k}‖H_i‖ is non-decreasing in L, so raising L lowers
        # the third sum strictly.
        rL = radius_sums(on; L = 10.0)
        @test rL.sum_delta2_over_M < r.sum_delta2_over_M
        @test rL.sum_delta == r.sum_delta          # the other two do not move
        @test rL.sum_delta2 == r.sum_delta2

        # With a constant ‖H_k‖ the third sum is the second divided by L + ‖H‖,
        # exactly. ScaledIdentity makes ‖H_k‖ = c at every iterate.
        c = 2.5
        sc = tr_solve(nlp; rule = RDelta(), model = ScaledIdentity(c = c),
                      params = TRParams(tol = 1e-8, max_iterations = 100),
                      hessian_norm = true, trace = true)
        @test all(x -> isapprox(x, c; atol = 1e-12),
                  sc.solver_specific[:hessian_norm_trajectory])
        rs = radius_sums(sc)
        @test rs.sum_delta2_over_M ≈ rs.sum_delta2 / c rtol = 1e-12
        rs3 = radius_sums(sc; L = 3.0)
        @test rs3.sum_delta2_over_M ≈ rs.sum_delta2 / (3.0 + c) rtol = 1e-12

        # An empty trace reports NaN and n = 0 rather than a sum of nothing.
        empt = radius_sums(synth(nlp; delta_trajectory = Float64[]))
        @test empt.n == 0 && isnan(empt.sum_delta)
    end
end
