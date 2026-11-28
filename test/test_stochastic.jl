# The sampling layer: problems, sampling rules, and the SampledNLP oracle.
#
# The property that matters most is the one checked last: with common random
# numbers the noise in f̂(x) − f̂(x+s) shrinks with the step, and without them it
# does not. Every mechanism stalls at small radii in the second case, for reasons
# that have nothing to do with the mechanism.

@testset "stochastic" begin
    quad(n = 4) = ADNLPModel(x -> 0.5 * sum(collect(1.0:n) .* (x .- 1.0).^2),
                             zeros(n), name = "quad")

    @testset "PerturbedSum has the base model as its exact mean" begin
        base = quad(); p = PerturbedSum(base, 200; σg = 1.0, seed = 3)
        x = [0.3, -0.7, 1.4, 0.2]
        full = collect(1:200)
        @test batch_obj(p, x, full) ≈ obj(base, x)          rtol = 1e-10
        g = zeros(4); batch_grad!(p, x, full, g)
        @test g ≈ grad(base, x)                              rtol = 1e-10
        @test true_objective(p, x) ≈ obj(base, x)
        @test true_gradient(p, x)  ≈ grad(base, x)
        # A subsample is biased away from it, by an amount that shrinks with N.
        errs = Float64[]
        for N in (4, 64, 1024)
            rng = MersenneTwister(1)
            e = mean(1:40) do _
                b = rand(rng, 1:200, N); gg = zeros(4)
                batch_grad!(p, x, b, gg); norm(gg - grad(base, x))
            end
            push!(errs, e)
        end
        @test errs[1] > errs[2] > errs[3]                    # Monte Carlo rate
        @test errs[3] < errs[1] / 4
    end

    @testset "sampling rules" begin
        st = SamplingState(3, 0.1, 2.0, 4.0, 9.0)            # σg = 2, σf = 3

        @test grad_sample_size(FixedSample(32), st) == 32
        @test obj_sample_size(FixedSample(32; N_obj = 8), st) == 8
        @test !couples_to_radius(FixedSample(32))

        # N ≥ (σ/(κΔ))² = (2/0.1)² = 400, and (3/0.01)² = 90 000
        rp = RadiusProportional(κ_g = 1.0, κ_f = 1.0, N_max = 10^7)
        @test grad_sample_size(rp, st) == 400
        @test obj_sample_size(rp, st)  == 90_000
        @test couples_to_radius(rp)

        # N ~ Δ^{-2}: halving the radius quadruples the sample size.
        st2 = SamplingState(3, 0.05, 2.0, 4.0, 9.0)
        @test grad_sample_size(rp, st2) == 4 * grad_sample_size(rp, st)

        # Norm test: σ²/(θ²‖g‖²) = 4/(0.25·4) = 4, and it never reads Δ.
        nt = NormTest(θ = 0.5)
        @test grad_sample_size(nt, st) == 4
        @test grad_sample_size(nt, SamplingState(3, 1e-8, 2.0, 4.0, 9.0)) ==
              grad_sample_size(nt, st)
        @test !couples_to_radius(nt)

        @test grad_sample_size(GeometricSample(N₀ = 8, rate = 2.0), st) == 64
        @test_throws ArgumentError RadiusProportional(κ_g = 0.0)
        @test_throws ArgumentError NormTest(θ = -1.0)
        @test_throws ArgumentError GeometricSample(rate = 0.5)
    end

    @testset "SampledNLP satisfies the NLP interface" begin
        base = quad(); p = PerturbedSum(base, 500; σg = 1.0, seed = 5)
        m = FiniteSumNLP(p, FixedSample(500))
        x = [0.3, -0.7, 1.4, 0.2]
        resample!(m, 1, 1.0, 1.0)
        @test length(m.batch_g) == 500
        @test obj(m, x) isa Real
        g = zeros(4); grad!(m, x, g); @test length(g) == 4
        @test size(Matrix(hess(m, x))) == (4, 4)
        Hv = zeros(4); hprod!(m, x, ones(4), Hv); @test length(Hv) == 4
        @test samples_used(m).total == 1000                   # 500 grad + 500 obj
    end

    @testset "the solver counts samples, not just iterations" begin
        base = quad(); p = PerturbedSum(base, 2_000; σg = 0.5, seed = 7)
        m = FiniteSumNLP(p, FixedSample(32))
        st = tr_solve(m; rule = RDelta(), model = ExactHessian(),
                      subsolver = ExactMS(), trace = true,
                      params = TRParams(tol = 1e-4, max_iterations = 60))
        ss = st.solver_specific
        @test haskey(ss, :grad_sample_trajectory)
        @test haskey(ss, :samples_total)
        @test length(ss[:grad_sample_trajectory]) == st.iter
        @test all(==(32), ss[:grad_sample_trajectory])
        @test ss[:samples_total] == 64 * st.iter              # grad + obj

        # A deterministic run is untouched: no sampling keys, no resampling.
        st2 = tr_solve(quad(); rule = RDelta(), trace = true,
                       params = TRParams(tol = 1e-8))
        @test !haskey(st2.solver_specific, :samples_total)
    end

    @testset "a stalled sampled run leaves the two trajectories in step" begin
        # The worse half of the :stalled off-by-one. `resample!` runs at the top of
        # the iteration and pushes to Ng_hist, so on a stalled exit that was not
        # traced :grad_sample_trajectory had k entries while :ratio_trajectory and
        # :sigma_g2_trajectory had k-1, and the two sampled series disagreed with
        # each other. Anything pairing N[i] against ρ[i] read the wrong iteration.
        base = quad(); p = PerturbedSum(base, 2_000; σg = 1.0, seed = 11)
        m = FiniteSumNLP(p, RadiusProportional(κ_g = 1.0, κ_f = 1.0); seed = 1)
        st = tr_solve(m; rule = RDelta(), trace = true,
                      params = TRParams(Δ0 = 1e-18, Δmax = 1e-18, tol = 1e-16,
                                        max_iterations = 20))
        ss = st.solver_specific
        @test st.status === :stalled
        @test st.iter >= 1
        for key in (:ratio_trajectory, :grad_sample_trajectory, :sigma_g2_trajectory,
                    :sigma_f2_trajectory, :step_trajectory, :accepted_trajectory)
            @test length(ss[key]) == st.iter
        end
        for key in (:delta_trajectory, :grad_trajectory, :obj_trajectory,
                    :true_grad_trajectory)
            @test length(ss[key]) == st.iter + 1
        end
    end

    @testset "radius-proportional sampling grows N as Δ falls" begin
        base = quad(); p = PerturbedSum(base, 20_000; σg = 1.0, seed = 11)
        # No N_max: on a finite sum the cap is M, imposed by the problem
        # (check_population_cap). M = 20_000 here, so the old N_max was redundant.
        m = FiniteSumNLP(p, RadiusProportional(κ_g = 1.0, κ_f = 1.0))
        # true_stop = :test, and it is load-bearing rather than incidental. Under
        # :none this run stops after 3 iterations at ‖ĝ‖ = 3e-16 on a batch of
        # N = 2 — ExactMS on a 2-term batch lands exactly on that batch's
        # minimiser — while the true ‖∇f‖ there is 1.79. There is no N-versus-Δ
        # trajectory to measure because there is no trajectory. :test costs no
        # evaluations and refuses the stop until the batch can resolve tol.
        #
        # on_refusal = :continue keeps every sample size the rule's own. Under the
        # default the first refusal moves the run to the full batch for good, and we
        # measured it stopping at iteration 4 with four sizes to correlate.
        st = tr_solve(m; rule = RDFO(ζ = 1.0), model = ExactHessian(),
                      subsolver = ExactMS(), trace = true,
                      params = TRParams(tol = 1e-6, max_iterations = 40,
                                        true_stop = :test, on_refusal = :continue))
        ss = st.solver_specific
        N = ss[:grad_sample_trajectory]; Δ = ss[:delta_trajectory]
        @test length(N) >= 5
        # N_k and Δ_k move in opposite directions: the mechanism pays for its own
        # radius policy. Checked on the run's own trajectory rather than asserted.
        @test cor(log.(max.(N, 1)), log.(max.(Δ[1:length(N)], 1e-300))) < -0.5
    end

    @testset "true_stop refuses a stop the batch cannot support" begin
        # The regression this exists for: ‖ĝ_k‖ ≤ tol on a tiny batch is not
        # evidence of anything. With an exact model and an exact subsolver the
        # step lands on the batch minimiser, so ‖ĝ‖ is zero to rounding at N = 2
        # whatever ∇f is doing, and the run certifies :first_order 1.79 away from
        # criticality.
        base = quad(); p = PerturbedSum(base, 20_000; σg = 1.0, seed = 11)
        mk() = FiniteSumNLP(p, RadiusProportional(κ_g = 1.0, κ_f = 1.0))
        run(mode) = tr_solve(mk(); rule = RDFO(ζ = 1.0), model = ExactHessian(),
                             subsolver = ExactMS(), trace = true,
                             params = TRParams(tol = 1e-6, max_iterations = 60,
                                               true_stop = mode))

        unguarded = run(:none)
        @test unguarded.status === :first_order
        @test unguarded.dual_feas <= 1e-6                       # the batch is happy
        @test norm(true_gradient(p, unguarded.solution)) > 1.0  # and it is wrong

        # Every guard rejects it and reaches the actual minimiser. That holds on
        # this problem, where the two terms of the batch cancel without vanishing.
        # The next testset is the case where the bound alone does not reject, and
        # where reading it only on the largest batch has to.
        for mode in (:test, :full, :both)
            st = run(mode)
            @test st.status === :first_order
            @test norm(true_gradient(p, st.solution)) <= 1e-6
            @test st.iter > unguarded.iter
        end
    end

    @testset "stop_N_min keeps :test and :both off a batch of vanished scores" begin
        # :test reads the plug-in variance of the batch that produced ĝ. When every
        # observation of that batch is fitted with a margin, each score vanishes on
        # its own, the variance vanishes with the gradient, and
        # ‖ĝ‖ + z·σ_g/√N ≤ tol holds whatever ∇f is doing. :both accepts on the
        # same bound and never takes its pass. stop_N_min is the guard. The bound
        # is read only on a batch of at least that many observations, by default
        # the largest batch, which here is the population. :full is the control,
        # because its verdict is the exact gradient.
        #
        # No threshold here is tuned. tol is the tolerance of the runs, and 1e-2
        # is four orders of magnitude above it and below every true gradient we
        # measured at a wrong stop.
        tol = 1e-6
        p = LogisticRegression(K = 5, M = 2_000, seed = 2)
        run(mode, seed; kw...) = begin
            m = FiniteSumNLP(p, RadiusProportional(); x0 = zeros(5), seed = seed)
            st = tr_solve(m; rule = RDelta(), model = BHHHModel(ridge = 1e-8),
                          subsolver = SteihaugCG(), trace = true,
                          params = TRParams(; tol = tol, max_iterations = 1_000,
                                            true_stop = mode, kw...))
            (m, st)
        end
        truth(st) = norm(true_gradient(p, st.solution))
        z = TRParams().stop_z

        # The bound without the guard, oracle seed 1. With stop_N_min = 1 both
        # modes stop at iteration 33 on a batch of two with σ_g² = 8.1e-14, where
        # the true ‖∇f‖ is 0.42. The last batch has two observations, the bound
        # holds on it, and no confirmation pass was ever paid for.
        wrong_iter = 0
        for mode in (:test, :both)
            m, st = run(mode, 1; stop_N_min = 1)
            @test st.status === :first_order
            @test truth(st) > 1e-2
            @test st.solver_specific[:grad_sample_trajectory][end] == 2
            @test st.dual_feas + z * grad_standard_error(m) <= tol
            @test samples_used(m).confirm == 0
            wrong_iter = st.iter
        end

        # The default, over six oracle seeds on the same problem. No mode reports
        # a :first_order that the true gradient does not support, and every mode
        # reports some, so the implication is not vacuous. We measured six of the
        # six for each mode. A run that stalls does so on the population, where
        # the stall is a statement about f and not about one batch.
        modes = (:test, :both, :full)
        res = [NamedTuple{modes}(map(mode -> run(mode, seed), modes)) for seed in 1:6]
        for r in res, mode in modes
            st = r[mode][2]
            @test st.status !== :first_order || truth(st) <= tol
            @test st.status !== :stalled ||
                  st.solver_specific[:grad_sample_trajectory][end] == population(p)
        end
        for mode in modes
            @test count(r -> r[mode][2].status === :first_order, res) >= 1
        end
        # On seed 1 the guarded runs go past the wrong stop. Until a run is moved
        # to the population, the three modes and the unguarded runs see the same
        # draws. The unguarded runs end on a batch of two, so nothing moved them,
        # and the unguarded :both run never paid for a pass, so no candidate came
        # earlier. The candidate at the wrong stop is therefore the first refusal in
        # every mode, and the run is on the population from the next iteration. We
        # measured the stop at iteration 68, with ‖∇f‖ = 5.4e-7.
        for mode in modes
            @test res[1][mode][2].iter > wrong_iter
            @test res[1][mode][2].solver_specific[:sample_floor_from] == wrong_iter + 1
        end
        # The cost. :test stays free. :both never pays more than :full, because
        # the two see the same iterates and :both skips a pass whenever the bound
        # settles the candidate.
        for r in res
            @test samples_used(r.test[1]).confirm == 0
            @test samples_used(r.both[1]).confirm <= samples_used(r.full[1]).confirm
        end
    end

    @testset "the :test bound on a batch fitted with a margin" begin
        # The same failure with no trajectory behind it. We draw two observations
        # and take the minimum-norm x that puts each of them 40 units on the right
        # side of its label. Their scores are then zero to rounding, so the batch
        # gradient and the plug-in variance are both zero, while the other 1 998
        # observations leave the true gradient at 0.40.
        tol = 1e-6
        p = LogisticRegression(K = 5, M = 2_000, seed = 2)
        m = FiniteSumNLP(p, FixedSample(2); x0 = zeros(5), seed = 1, replace = false)
        resample!(m, 1, 1.0, 1.0)
        b  = m.batch_g
        Xb = p.X[b, :]
        x  = Xb' * ((Xb * Xb') \ (40.0 .* (2 .* p.y[b] .- 1)))
        g  = zeros(5); grad!(m, x, g); update_variances!(m, x, g)

        @test norm(g) <= tol                            # the batch is satisfied
        @test grad_standard_error(m) <= tol             # and reports no noise
        @test norm(true_gradient(p, x)) > 1e-2          # the population disagrees

        confirm(mode; kw...) = TrustRegionRadius._confirm_stop(
            m, x, norm(g), TRParams(; tol = tol, true_stop = mode, kw...))

        # Without the guard the bound accepts, and :both then skips its pass.
        @test confirm(:test; stop_N_min = 1)
        @test confirm(:both; stop_N_min = 1)
        @test samples_used(m).confirm == 0
        # The minimum is compared with the batch size. Two observations satisfy a
        # minimum of two and not a minimum of three.
        @test confirm(:test; stop_N_min = 2)
        @test !confirm(:test; stop_N_min = 3)
        # The default refuses. :test stays free, :both pays for its pass and gets
        # the answer of :full.
        @test !confirm(:test)
        @test samples_used(m).confirm == 0
        @test !confirm(:both)
        @test samples_used(m).confirm == population(p)
        @test !confirm(:full)
        @test samples_used(m).confirm == 2 * population(p)
    end

    @testset "stop_N_min is a parameter with a default, capped by the oracle" begin
        # The default is the largest batch the oracle allows.
        @test TRParams().stop_N_min == typemax(Int)
        @test TRParams(stop_N_min = 5).stop_N_min == 5
        @test :stop_N_min in propertynames(TRParams())
        @test_throws ArgumentError TRParams(stop_N_min = 0)
        @test occursin("N_min = cap", sprint(show, TRParams(true_stop = :test)))
        @test occursin("N_min = 30", sprint(show, TRParams(true_stop = :test, stop_N_min = 30)))

        # The minimum never asks for more than the largest batch the oracle
        # allows. A population of ten is read at ten observations, where the batch
        # is the population, and not at nine. So is a population of 2 000 under the
        # default. An explicit minimum reads the bound on smaller batches.
        can(M, N; kw...) = begin
            q = LogisticRegression(K = 3, M = M, seed = 1)
            o = FiniteSumNLP(q, FixedSample(N); x0 = zeros(3), seed = 1)
            resample!(o, 1, 1.0, 1.0)
            TrustRegionRadius._batch_can_conclude(o, TRParams(; kw...))
        end
        @test can(10, 10)
        @test !can(10, 9)
        @test can(2_000, 2_000)
        @test !can(2_000, 1_999)
        @test can(2_000, 30; stop_N_min = 30)
        @test !can(2_000, 29; stop_N_min = 30)

        # What a refused stop does is a parameter too.
        @test TRParams().on_refusal === :cap
        @test TRParams(on_refusal = :continue).on_refusal === :continue
        @test :on_refusal in propertynames(TRParams())
        @test_throws ArgumentError TRParams(on_refusal = :ramp)
        @test occursin("on_refusal = :cap", sprint(show, TRParams(true_stop = :full)))
        @test !occursin("on_refusal", sprint(show, TRParams()))

        # The price of the guard. With σg = 0 every term of the sum has the
        # gradient of the base model, so at its minimiser a batch of eight has a
        # zero gradient and a zero variance, and this time the stop is real. :test
        # withholds it, a minimum of eight lets it through, and :both pays for one
        # pass to accept it.
        q = PerturbedSum(quad(), 2_000; σg = 0.0, seed = 3)
        o = FiniteSumNLP(q, FixedSample(8); seed = 1)
        resample!(o, 1, 1.0, 1.0)
        x = ones(4); g = zeros(4); grad!(o, x, g); update_variances!(o, x, g)
        @test norm(true_gradient(q, x)) <= 1e-6
        ok(mode; kw...) = TrustRegionRadius._confirm_stop(
            o, x, norm(g), TRParams(; tol = 1e-6, true_stop = mode, kw...))
        @test !ok(:test)
        @test ok(:test; stop_N_min = 8)
        @test samples_used(o).confirm == 0
        @test ok(:both)
        @test samples_used(o).confirm == population(q)
    end

    @testset "what each true_stop mode costs" begin
        base = quad(); p = PerturbedSum(base, 20_000; σg = 1.0, seed = 11)
        mk() = FiniteSumNLP(p, RadiusProportional(κ_g = 1.0, κ_f = 1.0))
        cost(mode) = begin
            m = mk()
            tr_solve(m; rule = RDFO(ζ = 1.0), model = ExactHessian(),
                     subsolver = ExactMS(),
                     params = TRParams(tol = 1e-6, max_iterations = 60,
                                       true_stop = mode))
            samples_used(m)
        end
        none, test, full, both = cost(:none), cost(:test), cost(:full), cost(:both)

        # :test is free — it reads σ_g and N_k, which the oracle already has.
        @test test.confirm == 0
        @test none.confirm == 0
        # :full and :both pay in whole population passes, and :both pays less
        # because the cheap bound settles the well-resolved candidates.
        @test full.confirm > 0
        @test both.confirm > 0
        @test both.confirm < full.confirm
        @test full.confirm % population(p) == 0
        # confirmation is separated from optimisation work but included in total
        @test both.total == both.grad + both.obj + both.confirm
    end

    @testset "a confirming mode concludes on the largest batch" begin
        # Under a confirming mode the bound is read on the largest batch, a refused
        # stop moves the run there for good, and so does a stall on a smaller batch.
        # A stall on the largest batch ends the run, after its batch has been
        # offered to the stopping test. Nothing of this happens under :none.
        p = LogisticRegression(K = 5, M = 2_000, seed = 2)
        M = population(p)
        kw = (rule = RDelta(), model = BHHHModel(ridge = 1e-8), subsolver = SteihaugCG())
        per_iter = (:ratio_trajectory, :grad_sample_trajectory, :sigma_g2_trajectory,
                    :sigma_f2_trajectory, :step_trajectory, :accepted_trajectory)
        per_state = (:delta_trajectory, :grad_trajectory, :obj_trajectory,
                     :true_grad_trajectory)
        aligned(st) = all(length(st.solver_specific[k]) == st.iter for k in per_iter) &&
                      all(length(st.solver_specific[k]) == st.iter + 1 for k in per_state)

        # A stall below the largest batch, then a stall on it. The run starts at the
        # solution with a radius of 1e-17, so every step falls below the stall
        # threshold. Under :none it ends on its first batch of eight, where ‖ĝ‖ is
        # 0.19. A confirming mode moves to the population instead, stalls there, and
        # that batch passes the stopping test.
        xs = tr_solve(FullBatchNLP(p; x0 = zeros(5)); kw...,
                      params = TRParams(tol = 1e-9, max_iterations = 1_000)).solution
        # `local` throughout: these closures share names with the testset's own
        # variables, and would otherwise assign to them.
        stall_run(mode) = begin
            local m = FiniteSumNLP(p, FixedSample(8); x0 = xs, seed = 1)
            local calls = Ref(0)
            local st = tr_solve(m; kw..., trace = true,
                                callback = (args...) -> (calls[] += 1),
                                params = TRParams(tol = 1e-6, Δ0 = 1e-17,
                                                  max_iterations = 50, true_stop = mode))
            (m, st, calls[])
        end
        m, st, calls = stall_run(:none)
        @test st.status === :stalled
        @test m.Ng_hist == [8]
        @test !haskey(st.solver_specific, :sample_floor_from)
        for (mode, passes) in ((:test, 0), (:both, 0), (:full, 1))
            m, st, calls = stall_run(mode)
            @test st.status === :first_order
            @test m.Ng_hist == [8, M]
            @test st.solver_specific[:sample_floor_from] == 2
            @test norm(true_gradient(p, st.solution)) <= 1e-6
            @test samples_used(m).confirm == passes * M
            # The first stall is a rejected iteration like any other. It is traced,
            # recorded and handed to the callback. The second ends the run.
            @test calls == 1
            @test aligned(st)
        end

        # :sample_floor_from names an iteration drawn under the floor. A run that
        # ends before any such draw keeps 0, and the oracle has dropped the floor.
        m1 = FiniteSumNLP(p, FixedSample(8); x0 = xs, seed = 1)
        s1 = tr_solve(m1; kw..., trace = true,
                      params = TRParams(tol = 1e-6, Δ0 = 1e-17, max_iterations = 1,
                                        true_stop = :test))
        @test s1.status === :max_iter && m1.Ng_hist == [8]
        @test s1.solver_specific[:sample_floor_from] == 0
        @test m1.N_floor == 0

        # A stalled batch is judged only exactly. Under :none this probit run stalls
        # at iteration 15 on a batch of two whose scores have vanished, at a point
        # where the true gradient is 2.28. With stop_N_min = 1 the bound reads every
        # batch but that one, so :test and :both move to the population and stop
        # there. We measured the stop at iteration 103 in both modes.
        q2 = Probit(K = 5, M = 2_000, seed = 2)
        probit_run(mode) = begin
            local mq = FiniteSumNLP(q2, RadiusProportional(); x0 = zeros(5), seed = 2)
            local sq = tr_solve(mq; kw..., trace = true,
                                params = TRParams(tol = 1e-6, max_iterations = 1_000,
                                                  true_stop = mode, stop_N_min = 1))
            (mq, sq)
        end
        mq2, sq2 = probit_run(:none)
        @test sq2.status === :stalled && mq2.Ng_hist[end] == 2
        @test norm(true_gradient(q2, sq2.solution)) > 1e-2
        for mode in (:test, :both)
            mq2, sq2 = probit_run(mode)
            @test sq2.status !== :first_order || norm(true_gradient(q2, sq2.solution)) <= 1e-6
            @test mq2.Ng_hist[end] == population(q2)
        end

        # Only the candidate that follows a stall is judged as a stalled batch. With
        # a budget of 1 000 the largest batch is a sample, and on_refusal = :continue
        # leaves the sizes to the rule, so that only a stall can raise the floor.
        # The step stalls at iteration 15 on a batch of two, as under :none above,
        # and the batch of 1 000 is drawn from iteration 16. The later candidates
        # are ordinary batches, which the bound of :test reads. We measured the stop
        # at iteration 99, with ‖∇f‖ = 0.029. Were every later candidate judged as a
        # stalled batch, :test could not stop on a sample, and we measured the run
        # reaching max_iterations.
        mk = FiniteSumNLP(q2, RadiusProportional(); x0 = zeros(5), seed = 2, budget = 1_000)
        sk = tr_solve(mk; kw..., trace = true,
                      params = TRParams(tol = 0.1, max_iterations = 200, true_stop = :test,
                                        on_refusal = :continue))
        fk = sk.solver_specific[:sample_floor_from]
        @test fk == 16 && mk.Ng_hist[fk - 1] == 2
        @test all(==(1_000), mk.Ng_hist[fk:end])
        @test sk.status === :first_order
        @test sk.iter > fk
        @test norm(true_gradient(q2, sk.solution)) <= 0.1

        # A stall on the largest batch stays final. This is the configuration of the
        # testset on stalled trajectories above. The rule asks for the population at
        # once and ‖ĝ‖ is far above tol, so the stopping test cannot accept.
        q = PerturbedSum(quad(), 2_000; σg = 1.0, seed = 11)
        for mode in (:test, :both, :full)
            mq = FiniteSumNLP(q, RadiusProportional(κ_g = 1.0, κ_f = 1.0); seed = 1)
            st = tr_solve(mq; rule = RDelta(), trace = true,
                          params = TRParams(Δ0 = 1e-18, Δmax = 1e-18, tol = 1e-16,
                                            max_iterations = 20, true_stop = mode))
            @test st.status === :stalled
            @test mq.Ng_hist == [population(q)]
            @test st.solver_specific[:sample_floor_from] == 0
            @test samples_used(mq).confirm == 0
        end

        # A refused stop moves the run to the largest batch, unless on_refusal says
        # otherwise. The configuration is the tutorial's scoring example, where a
        # batch of eight offers candidates that the full pass refuses. We measured a
        # stop at iteration 3 by default, and 199 refusals ending at max_iter with
        # on_refusal = :continue.
        r = PerturbedSum(quad(), 2_000; σg = 1.0, seed = 3)
        scored(; okw...) = begin
            local mr = FiniteSumNLP(r, FixedSample(8); seed = 1)
            local st = tr_solve(mr; rule = RDelta(), model = ExactHessian(),
                                subsolver = ExactMS(), trace = true,
                                params = TRParams(; tol = 1e-2, max_iterations = 200,
                                                  true_stop = :full, okw...))
            (mr, st)
        end
        mr, st = scored()
        from = st.solver_specific[:sample_floor_from]
        @test st.status === :first_order
        @test norm(true_gradient(r, st.solution)) <= 1e-2
        @test from > 1
        @test all(==(8), mr.Ng_hist[1:from-1])
        @test all(==(population(r)), mr.Ng_hist[from:end])
        # The floor is not a cap hit, and the oracle drops it when the run ends. It
        # keeps the iteration the floor took effect until the next reset.
        @test st.solver_specific[:sample_cap_hits] == 0
        @test mr.N_floor == 0 && mr.floor_from == from
        mc, sc = scored(on_refusal = :continue)
        @test sc.solver_specific[:sample_floor_from] == 0
        @test all(==(8), mc.Ng_hist)
        @test samples_used(mc).confirm > 0
        @test samples_used(mc).confirm % population(r) == 0

        # The floor belongs to one run. reset_sampling!, which every solve calls
        # first, clears it, so a second solve on the same oracle repeats the first.
        used = samples_used(mr)
        reset_sampling!(mr)
        @test mr.N_floor == 0 && mr.floor_from == 0
        st2 = tr_solve(mr; rule = RDelta(), model = ExactHessian(), subsolver = ExactMS(),
                       trace = true,
                       params = TRParams(tol = 1e-2, max_iterations = 200, true_stop = :full))
        @test st2.status === st.status && st2.iter == st.iter
        @test st2.solution == st.solution
        @test samples_used(mr) == used
        resample!(mr, 1, 1.0, 1.0)
        @test mr.Ng == 8                      # the rule's size: the floor is gone

        # The floor may be raised more than once, and :sample_floor_from keeps the
        # first time. With a budget of 500 the largest batch is a sample, the exact
        # pass refuses candidates there too, and we measured 19 refusals in 20
        # iterations, with the batch of 500 from iteration 3.
        mb = FiniteSumNLP(r, FixedSample(8); seed = 1, budget = 500)
        sb = tr_solve(mb; rule = RDelta(), model = ExactHessian(), subsolver = ExactMS(),
                      trace = true,
                      params = TRParams(tol = 1e-2, max_iterations = 20, true_stop = :full))
        @test samples_used(mb).confirm > population(r)      # more than one refusal
        @test sb.solver_specific[:sample_floor_from] == findfirst(==(500), mb.Ng_hist)
        @test mb.Ng_hist[end] == 500

        # The floor itself is never counted as a cap hit, whatever the rule asks for.
        of = FiniteSumNLP(p, FixedSample(8); x0 = zeros(5), seed = 1)
        TrustRegionRadius._escalate!(of)
        resample!(of, 1, 1.0, 1.0)
        @test of.Ng == M && of.Nf == M
        @test of.capped == 0
        @test of.floor_from == 1

        # On a finite sum without a budget the largest batch is the population.
        # Under :nested with replacement a batch of M draws is not, so a confirming
        # mode refuses that oracle. A budget below M makes the largest batch a
        # sample, and the bound then treats it as one.
        nested(; okw...) = FiniteSumNLP(p, RadiusProportional(); x0 = zeros(5), seed = 1,
                                        scheme = :nested, okw...)
        for mode in (:test, :both, :full)
            par = TRParams(true_stop = mode)
            @test_throws ArgumentError FiniteSumTRSolver(nested(); params = par)
            @test_throws ArgumentError tr_solve(nested(); kw..., params = par)
            @test FiniteSumTRSolver(nested(replace = false); params = par) isa FiniteSumTRSolver
            @test FiniteSumTRSolver(nested(budget = 500); params = par) isa FiniteSumTRSolver
        end
        @test FiniteSumTRSolver(nested(); params = TRParams()) isa FiniteSumTRSolver
        # The oracle handed to solve! is checked, not only the one the solver was
        # built on.
        built = FiniteSumTRSolver(nested(replace = false); params = TRParams(true_stop = :test))
        other = nested()
        @test_throws ArgumentError solve!(built, other, GenericExecutionStats(other))

        # FullBatch draws the population from the first iteration, so no floor
        # changes a size, and every mode reproduces the deterministic solver.
        det = tr_solve(FullBatchNLP(p; x0 = zeros(5)); kw...,
                       params = TRParams(tol = 1e-6, max_iterations = 1_000))
        for mode in (:none, :test, :both, :full)
            st = tr_solve(FiniteSumNLP(p, FullBatch(); x0 = zeros(5), seed = 1); kw...,
                          params = TRParams(tol = 1e-6, max_iterations = 1_000,
                                            true_stop = mode))
            @test st.status === det.status
            @test st.iter == det.iter
            @test st.solution == det.solution
        end

        # On an expectation the rule's own N_max is part of the largest batch. The
        # floor stops there, and the bound is read there. Taking the budget alone,
        # 10^6, would draw a million observations per batch.
        pe = PerturbedExpectation(quad(); σg = 1.0)
        for seed in 1:3
            me = ExpectationNLP(pe, RadiusProportional(κ_g = 1.0, κ_f = 1.0, N_max = 1_000);
                                seed = seed)
            st = tr_solve(me; rule = RDFO(ζ = 1.0), model = ExactHessian(),
                          subsolver = ExactMS(), trace = true,
                          params = TRParams(tol = 0.3, max_iterations = 100,
                                            true_stop = :test))
            @test st.status === :first_order
            @test norm(true_gradient(pe, st.solution)) <= 0.3
            @test st.solver_specific[:sample_floor_from] > 0
            @test maximum(me.Ng_hist) == 1_000
        end
        o = ExpectationNLP(pe, RadiusProportional(κ_g = 1.0, κ_f = 1.0, N_max = 10); seed = 1)
        resample!(o, 1, 1e-6, 1.0)
        @test o.Ng == 10
        @test TrustRegionRadius._batch_can_conclude(o, TRParams())

        # The bound waits for a variance estimated on the batch in hand. Before the
        # first resample the oracle holds the placeholder σ_g² = 1 that its
        # constructor wrote, and here the initial batch of eight is already the
        # largest one.
        o8 = ExpectationNLP(pe, RadiusProportional(N_max = 8); seed = 1)
        @test o8.Ng == 8
        @test !TrustRegionRadius._batch_can_conclude(o8, TRParams())
        resample!(o8, 1, 1e-6, 1.0)                 # the rule asks for its N_max of 8
        @test o8.Ng == 8
        @test TrustRegionRadius._batch_can_conclude(o8, TRParams())
        # On a population of eight the initial batch is the population, where the
        # standard error is zero whatever the placeholder says.
        tiny = FiniteSumNLP(LogisticRegression(K = 3, M = 8, seed = 1), FixedSample(8);
                            x0 = zeros(3), seed = 1)
        @test TrustRegionRadius._batch_can_conclude(tiny, TRParams())

        # The expectation loop moves on after a stall as the finite-sum loop does.
        # The run starts at the solution with a radius of 1e-17, stalls on its batch
        # of eight, moves to the budget of 1 000 and stalls there. That batch is a
        # sample, so :test does not accept it and the run ends :stalled, while :both
        # and :full accept it on the exact gradient.
        exp_run(mode) = begin
            local me = ExpectationNLP(pe, FixedSample(8); x0 = ones(4), seed = 1,
                                      budget = 1_000)
            local calls = Ref(0)
            local st = tr_solve(me; rule = RDelta(), model = ExactHessian(),
                                subsolver = ExactMS(), trace = true,
                                callback = (args...) -> (calls[] += 1),
                                params = TRParams(tol = 0.3, Δ0 = 1e-17,
                                                  max_iterations = 50, true_stop = mode))
            (me, st, calls[])
        end
        me, st, calls = exp_run(:none)
        @test st.status === :stalled && me.Ng_hist == [8]
        for (mode, status) in ((:test, :stalled), (:both, :first_order),
                               (:full, :first_order))
            me, st, calls = exp_run(mode)
            @test st.status === status
            @test me.Ng_hist == [8, 1_000]
            @test st.solver_specific[:sample_floor_from] == 2
            @test calls == 1
            @test aligned(st)
        end

        # The expectation loop, too, judges only the candidate that follows a stall
        # as a stalled batch. This quadratic has its minimiser at 0.1·ones(4). The run
        # starts from 0 with a radius of 1e-15, so a step stalls when its batch has
        # ‖ĝ‖ ≤ 0.22, while ‖∇f‖ = 0.55 there. With seed 1057 the batch of two at
        # iteration 1 has ‖ĝ‖ = 0.12 and its step stalls, and the step on the batch of
        # 1 000 at iteration 2 does not. The radius then doubles at every step, and we
        # measured the stop at iteration 49, on an ordinary batch that the bound of
        # :test reads, with ‖∇f‖ = 0.022. Were every later candidate judged as a
        # stalled batch, :test could not stop on this sample, and we measured the run
        # reaching max_iterations.
        pz = PerturbedExpectation(ADNLPModel(x -> 0.5 * sum(collect(1.0:4) .* (x .- 0.1).^2),
                                             zeros(4)); σg = 0.4)
        mz = ExpectationNLP(pz, FixedSample(2); x0 = zeros(4), seed = 1057, budget = 1_000)
        sz = tr_solve(mz; rule = RDelta(), model = ExactHessian(), subsolver = ExactMS(),
                      trace = true,
                      params = TRParams(tol = 0.1, Δ0 = 1e-15, max_iterations = 200,
                                        true_stop = :test))
        bz = sz.solver_specific[:branch_trajectory]
        @test bz[1] === :none && bz[2] !== :none      # a stall at 1, none at 2
        @test mz.Ng_hist[1:2] == [2, 1_000]
        @test sz.solver_specific[:sample_floor_from] == 2
        @test sz.status === :first_order
        @test sz.iter > 2
        @test norm(true_gradient(pz, sz.solution)) <= 0.1

        # With no budget and no N_max there is no largest batch to move to, and a
        # confirming mode is refused, by the constructor and by solve!.
        unbounded() = ExpectationNLP(pe, RadiusProportional(); seed = 1,
                                     budget = typemax(Int))
        for mode in (:test, :both, :full)
            @test_throws ArgumentError ExpectationTRSolver(unbounded();
                                                           params = TRParams(true_stop = mode))
        end
        @test ExpectationTRSolver(unbounded(); params = TRParams()) isa ExpectationTRSolver
        bounded = ExpectationNLP(pe, RadiusProportional(N_max = 1_000); seed = 1,
                                 budget = typemax(Int))
        eb = ExpectationTRSolver(bounded; params = TRParams(true_stop = :test))
        other_e = unbounded()
        @test_throws ArgumentError solve!(eb, other_e, GenericExecutionStats(other_e))
    end

    @testset "the finite population correction is what makes :test usable" begin
        # Without it the standard error stays positive at N = M, so no tolerance
        # below z·σ_g/√M is ever certifiable and :both degenerates into :full.
        base = quad(); p = PerturbedSum(base, 500; σg = 1.0, seed = 17)
        m = FiniteSumNLP(p, FixedSample(500))          # N = M: the batch IS the truth
        resample!(m, 1, 1.0, 1.0)
        x = [0.3, -0.7, 1.4, 0.2]; g = zeros(4)
        grad!(m, x, g); update_variances!(m, x, g)
        @test m.Ng == 500
        @test grad_standard_error(m) == 0.0

        # and strictly positive, shrinking in N, below the population
        small = FiniteSumNLP(p, FixedSample(50); replace = false)
        resample!(small, 1, 1.0, 1.0)
        grad!(small, x, g); update_variances!(small, x, g)
        @test grad_standard_error(small) > 0
    end

    @testset "common random numbers keep ρ̂ usable at small radii" begin
        # With a shared batch, f̂(x) − f̂(x+s) is an average of F_i(x) − F_i(x+s)
        # and shrinks with ‖s‖. With independent batches it does not, so ρ̂ is
        # noise once the step is smaller than the sampling error — and every
        # mechanism then stalls for a reason external to it.
        base = quad(); p = PerturbedSum(base, 5_000; σg = 1.0, seed = 13)
        rng = MersenneTwister(2)
        x = [0.5, 0.5, 0.5, 0.5]; d = [1.0, 0.0, 0.0, 0.0]

        for α in (1e-1, 1e-3)
            s = α .* d
            crn = Float64[]; ind = Float64[]
            for _ in 1:200
                b1 = rand(rng, 1:5_000, 64); b2 = rand(rng, 1:5_000, 64)
                truth = true_objective(p, x) - true_objective(p, x .+ s)
                push!(crn, batch_obj(p, x, b1) - batch_obj(p, x .+ s, b1) - truth)
                push!(ind, batch_obj(p, x, b1) - batch_obj(p, x .+ s, b2) - truth)
            end
            # Shared batch: error scales with ‖s‖. Independent: it does not.
            @test std(crn) < std(ind)
            α < 1e-2 && @test std(crn) < 0.1 * std(ind)
        end
    end
end
