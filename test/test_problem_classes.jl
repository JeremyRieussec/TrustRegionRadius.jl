# The problem-class hierarchy: the assertions that make the separation worth
# having.
#
# Three regimes, three solvers, and a set of compatibility checks that fire at
# construction rather than producing a run that looks plausible:
#
#   * BHHH over anything that is not a LikelihoodProblem
#   * Gauss-Newton over anything that is not an NLSProblem
#   * a sampling rule handed to the deterministic solver
#   * a sampled oracle handed to the deterministic solver
#   * FullBatch on an expectation
#   * a user N_max on a finite sum
#   * true_stop where there is no truth

@testset "problem classes" begin

    base = ADNLPModel(z -> 0.5 * sum(collect(1.0:4) .* (z .- 1.0).^2), zeros(4))
    fsum = PerturbedSum(base, 500; σg = 1.0, seed = 1)
    expp = PerturbedExpectation(base; σg = 1.0)
    logit = LogisticRegression(K = 4, M = 400, seed = 1)
    ls    = linear_least_squares(n = 3, M = 200, seed = 1)

    @testset "the hierarchy is what it claims" begin
        @test PerturbedSum          <: FiniteSumProblem
        @test PerturbedExpectation  <: ExpectationProblem
        @test FiniteSum             <: FiniteSumProblem
        @test ScoredProblem         <: FiniteSumProblem
        @test LikelihoodProblem     <: ScoredProblem
        @test NLSProblem            <: LikelihoodProblem
        @test LogisticRegression    <: LikelihoodProblem
        @test MLPClassifier         <: LikelihoodProblem
        @test LeastSquares          <: NLSProblem
        # An expectation is not a finite sum with a large M, and vice versa.
        @test !(ExpectationProblem <: FiniteSumProblem)
        @test !(FiniteSumProblem <: ExpectationProblem)
        @test ExpectationProblem <: SampledProblem
        @test FiniteSumProblem   <: SampledProblem
    end

    @testset "class traits" begin
        @test problem_class(fsum)  === :finite_sum
        @test problem_class(expp)  === :expectation
        @test problem_class(base)  === :deterministic
        @test problem_class(FullBatchNLP(logit)) === :deterministic

        @test population(fsum) == 500
        @test population(logit) == 400
        @test population(expp) == typemax(Int)     # unbounded, not an error
        @test n_terms(logit) == population(logit)

        @test !has_scores(fsum) && has_scores(logit) && has_scores(ls)
        @test has_truth(fsum) && has_truth(logit)  # one full pass
        @test has_truth(expp)                      # PerturbedExpectation supplies it
        @test full_batch(logit) == collect(1:400)
    end

    @testset "the expectation is genuinely not a finite sum" begin
        # PerturbedSum centres its M perturbations, so at N = M the estimate is
        # exact. PerturbedExpectation draws fresh, so the sample mean is O(σ/√N)
        # at every N — there is no sample size at which it becomes exact, which is
        # the substantive difference the two types carry.
        x = [0.3, -0.7, 1.4, 0.2]
        @test batch_obj(fsum, x, full_batch(fsum)) ≈ true_objective(fsum, x) rtol = 1e-12

        rng = MersenneTwister(3)
        errs = Float64[]
        for N in (64, 4096)
            d = draw_batch(expp, rng, N)
            g = zeros(4); batch_grad!(expp, x, d, g)
            push!(errs, norm(g - true_gradient(expp, x)))
        end
        @test errs[1] > errs[2] > 0                # shrinks, never reaches zero
        @test !applicable(full_batch, expp)
    end

    @testset "model Hessians are checked against the problem class" begin
        # BHHH needs a likelihood. Over a plain NLP, or over an unscored finite
        # sum, the arithmetic still runs and produces a positive semidefinite
        # matrix — which is exactly the failure worth preventing, because nothing
        # in ρ, ‖g‖ or the radius trace reveals it.
        @test required_problem(BHHHModel())        === LikelihoodProblem
        @test required_problem(BHHH2Model())       === LikelihoodProblem
        @test required_problem(GaussNewtonModel()) === NLSProblem
        @test required_problem(ExactHessian())     === AbstractProblem
        @test required_problem(LBFGSModel())       === AbstractProblem

        @test_throws ArgumentError DeterministicTRSolver(base; model = BHHHModel())
        @test_throws ArgumentError tr_solve(FiniteSumNLP(fsum, FixedSample(32));
                                            model = BHHHModel())
        # a likelihood: fine, sampled or not
        @test DeterministicTRSolver(FullBatchNLP(logit); model = BHHHModel()) isa
              DeterministicTRSolver
        @test FiniteSumTRSolver(FiniteSumNLP(logit, FixedSample(32); x0 = zeros(4));
                                model = BHHHModel()) isa FiniteSumTRSolver

        # Gauss-Newton needs the Jacobian, so a likelihood is not enough.
        @test_throws ArgumentError DeterministicTRSolver(FullBatchNLP(logit);
                                                         model = GaussNewtonModel())
        @test DeterministicTRSolver(FullBatchNLP(ls); model = GaussNewtonModel()) isa
              DeterministicTRSolver
        # and BHHH *is* enough for least squares: it is a Gaussian likelihood
        @test DeterministicTRSolver(FullBatchNLP(ls); model = BHHHModel()) isa
              DeterministicTRSolver
    end

    @testset "the deterministic solver refuses a sampled oracle" begin
        fs = FiniteSumNLP(fsum, FixedSample(32))
        ex = ExpectationNLP(expp, FixedSample(32))
        @test_throws ArgumentError DeterministicTRSolver(fs)
        @test_throws ArgumentError DeterministicTRSolver(ex)
        # and takes the full-batch view of the same problem happily
        @test DeterministicTRSolver(FullBatchNLP(fsum)) isa DeterministicTRSolver
        # ... which is also what tr_solve picks
        @test tr_solve(FullBatchNLP(fsum); params = TRParams(max_iterations = 2)) isa TRResult
    end

    @testset "tr_solve dispatches on the class" begin
        p = TRParams(max_iterations = 3, tol = 1e-6)
        @test tr_solve(base; params = p).status in (:first_order, :max_iter)
        @test tr_solve(FiniteSumNLP(fsum, FixedSample(32)); params = p,
                       subsolver = ExactMS()).status in (:first_order, :max_iter, :stalled)
        @test tr_solve(ExpectationNLP(expp, FixedSample(32)); params = p,
                       subsolver = ExactMS()).status in (:first_order, :max_iter, :stalled)
    end

    @testset "N_max belongs to the expectation, not to the finite sum" begin
        # On a finite sum the cap is M, imposed by the problem. A user N_max there
        # is either redundant or a deliberate sub-population budget, and those are
        # different intentions that should not share a keyword.
        @test user_cap(RadiusProportional()) === nothing
        @test user_cap(RadiusProportional(N_max = 500)) == 500
        @test sample_cap(RadiusProportional()) == typemax(Int)
        @test sample_cap(RadiusProportional(N_max = 500)) == 500

        @test_throws ArgumentError FiniteSumNLP(fsum, RadiusProportional(N_max = 500))
        @test_throws ArgumentError FiniteSumNLP(fsum, NormTest(N_max = 100))
        @test FiniteSumNLP(fsum, RadiusProportional()) isa FiniteSumNLP
        # ... and on an expectation it is exactly right
        @test ExpectationNLP(expp, RadiusProportional(N_max = 500)) isa ExpectationNLP

        # A deliberate sub-population budget goes to the oracle, where it reads as
        # an experimental choice rather than a property of the rule.
        m = FiniteSumNLP(fsum, RadiusProportional(); budget = 100)
        @test population_cap(m) == 100
        resample!(m, 1, 1e-10, 1.0)              # a demand far above the budget
        @test m.Ng == 100 && m.capped >= 1
        # and with no budget the cap is M
        m2 = FiniteSumNLP(fsum, RadiusProportional())
        @test population_cap(m2) == population(fsum)
        resample!(m2, 1, 1e-10, 1.0)
        @test m2.Ng == population(fsum)
    end

    @testset "a full-size draw IS the population, not a bootstrap" begin
        # The bug this pins: draw_batch defaulted to replace = true, so N = M gave
        # rand(1:M, M) — a bootstrap resample missing ~37% of the terms. FullBatch
        # was therefore noisy, :full_batch_trajectory claimed exactness it did not
        # have, and the equivalence against DeterministicTRSolver was false while
        # every status/solution comparison still passed by luck.
        rng = MersenneTwister(7)
        M = population(fsum)
        for rep in (true, false)
            b = draw_batch(fsum, rng, M; replace = rep)
            @test sort(b) == collect(1:M)             # every term exactly once
            @test length(unique(b)) == M
        end
        # ... and over-full requests saturate rather than duplicate
        @test sort(draw_batch(fsum, rng, M + 50)) == collect(1:M)
        # below M, a with-replacement draw is still a genuine resample
        small = draw_batch(fsum, rng, 32; replace = true)
        @test length(small) == 32 && all(i -> 1 <= i <= M, small)

        # the batch estimate at N = M must equal the truth exactly
        m = FiniteSumNLP(fsum, FullBatch())
        resample!(m, 1, 1.0, 1.0)
        xq = [0.3, -0.7, 1.4, 0.2]
        gb = zeros(4); grad!(m, xq, gb)
        @test gb ≈ true_gradient(fsum, xq) rtol = 1e-12
        @test obj(m, xq) ≈ true_objective(fsum, xq) rtol = 1e-12
    end

    @testset "FullBatch is finite-sum only" begin
        @test requires_finite_population(FullBatch())
        @test !requires_finite_population(FixedSample(8))
        @test_throws ArgumentError ExpectationNLP(expp, FullBatch())
        @test_throws ArgumentError grad_sample_size(
            FullBatch(), SamplingState(1, 1.0, 1.0, 1.0, 1.0))   # N_pop unbounded

        m = FiniteSumNLP(fsum, FullBatch())
        resample!(m, 1, 1.0, 1.0)
        @test m.Ng == population(fsum) == m.Nf
        @test m.full_hist == [true]
    end

    @testset "FullBatch reproduces the deterministic solver exactly" begin
        # The only place the sampled and exact code paths can be compared iterate
        # for iterate: at N_k = M the finite-sum iteration IS the deterministic
        # one, so this pins the resampling and re-evaluation logic against a
        # reference that has none.
        p = TRParams(tol = 1e-8, max_iterations = 200)
        det = tr_solve(FullBatchNLP(fsum); rule = RDelta(), subsolver = ExactMS(),
                       params = p, trace = true)
        fs  = tr_solve(FiniteSumNLP(fsum, FullBatch()); rule = RDelta(),
                       subsolver = ExactMS(), params = p, trace = true)
        @test det.status === fs.status
        @test det.iter == fs.iter
        @test det.solution ≈ fs.solution rtol = 1e-12
        @test all(fs.solver_specific[:full_batch_trajectory])
        # Every trajectory, not just the endpoints: a noisy "full" batch can still
        # land near the same solution while taking a different path, so comparing
        # only status and solution is what let the bootstrap bug hide.
        for key in (:delta_trajectory, :obj_trajectory,
                    :ratio_trajectory, :step_trajectory)
            @test det.solver_specific[key] ≈ fs.solver_specific[key] rtol = 1e-12
        end
        @test det.solver_specific[:active_trajectory] ==
              fs.solver_specific[:active_trajectory]
        @test det.solver_specific[:accepted_trajectory] ==
              fs.solver_specific[:accepted_trajectory]
        # and the sampled run reports the truth, because the batch IS the truth.
        #
        # An ABSOLUTE tolerance, deliberately. These are two different reductions of
        # the same mathematical quantity — true_gradient evaluates the base model,
        # dual_feas averages M perturbed terms — so they agree only to rounding, and
        # at a converged point both are ~1e-15, where a *relative* comparison is
        # meaningless: rtol = 1e-10 on quantities of magnitude 4e-15 demands
        # agreement to 4e-25 and no implementation can pass it. The absolute
        # agreement, ~2e-17, is the equivalence actually holding.
        @test isapprox(fs.solver_specific[:true_grad_trajectory][end], fs.dual_feas;
                       atol = 1e-12, rtol = 1e-6)
    end

    @testset "true_stop needs a truth, and a regime where it means something" begin
        # Deterministic: the stopping test already IS the true one.
        @test_throws ArgumentError DeterministicTRSolver(base;
            params = TRParams(true_stop = true))
        # Finite sum: always available, at one full pass.
        @test FiniteSumTRSolver(FiniteSumNLP(fsum, FixedSample(16));
                                params = TRParams(true_stop = true)) isa FiniteSumTRSolver
        # Expectation with a closed-form mean: available.
        @test ExpectationTRSolver(ExpectationNLP(expp, FixedSample(16));
                                  params = TRParams(true_stop = true)) isa ExpectationTRSolver
    end

    @testset "the finite-sum trace records which iterations were exact" begin
        m = FiniteSumNLP(logit, GeometricSample(N₀ = 8, rate = 2.0); x0 = zeros(4))
        st = tr_solve(m; rule = RDelta(), model = BHHHModel(ridge = 1e-8),
                      trace = true, params = TRParams(tol = 1e-8, max_iterations = 25))
        ss = st.solver_specific
        @test haskey(ss, :full_batch_trajectory)
        @test ss[:population] == 400
        N = ss[:grad_sample_trajectory]
        @test ss[:full_batch_trajectory] == [n == 400 for n in N]
        @test any(ss[:full_batch_trajectory])       # 8·2^k reaches 400 quickly
        @test all(<=(400), N)                       # never asks for more than M
        @test ss[:problem_class] === :finite_sum
    end

    @testset "the expectation trace has no full-batch key" begin
        m = ExpectationNLP(expp, FixedSample(64); budget = 10_000)
        st = tr_solve(m; rule = RDelta(), subsolver = ExactMS(), trace = true,
                      params = TRParams(tol = 1e-4, max_iterations = 20))
        ss = st.solver_specific
        @test !haskey(ss, :full_batch_trajectory)
        @test !haskey(ss, :population)
        @test ss[:problem_class] === :expectation
        @test haskey(ss, :true_grad_trajectory)     # PerturbedExpectation has truth
        @test ss[:samples_total] == 128 * st.iter
    end
end
