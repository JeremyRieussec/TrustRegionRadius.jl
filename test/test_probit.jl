# Binary probit, checked against its definition and not against itself:
#
#   derivatives   against central differences
#   the kernels   against the asymptotic series and Laplace's continued fraction, from
#                 the origin to where Φ itself is not representable
#   the identity  exactly, conditional on x, and then as a rate in M
#   `noise > 0`   breaks the identity at β* and at the minimiser
#
# Two assertions come with a control. The conditional identity is only evidence if
# it fails away from the generating parameters, and a confirmed stop is only
# evidence if the unconfirmed run stops somewhere else.

@testset "probit" begin

    @testset "interface and derivatives" begin
        p = Probit(K = 5, M = 400, seed = 3)
        @test p isa LikelihoodProblem
        @test population(p) == 400
        @test has_scores(p)
        @test supports_paired(p)
        @test full_batch(p) == collect(1:400)

        all_b = collect(1:p.M)
        # At β = 0 both outcomes have probability 1/2 whatever the data, so
        # f = log 2. A check on the loss that needs no reference value.
        @test abs(batch_obj(p, zeros(p.n), all_b) - log(2)) < 1e-12

        β = 0.3 .* randn(MersenneTwister(11), p.n)
        g = zeros(p.n); batch_grad!(p, β, all_b, g)
        ε = 1e-6

        fd = zeros(p.n)
        for j in 1:p.n
            bp = copy(β); bp[j] += ε
            bm = copy(β); bm[j] -= ε
            fd[j] = (batch_obj(p, bp, all_b) - batch_obj(p, bm, all_b)) / (2ε)
        end
        @test norm(g - fd) / norm(fd) < 1e-7

        # The gradient is the mean of the score columns: what lets BHHH be built
        # from the array the gradient came from. Probit takes `batch_grad!` from
        # ScoredProblem, so the line holds by construction today. It is kept for the
        # day the problem gets a `batch_grad!` of its own, as MLPClassifier has.
        S = scores(p, β, all_b)
        @test size(S) == (p.n, p.M)
        @test norm(vec(sum(S; dims = 2)) ./ p.M - g) < 1e-13

        H = batch_hess(p, β, all_b)
        Hfd = zeros(p.n, p.n); gp = zeros(p.n); gm = zeros(p.n)
        for j in 1:p.n
            bp = copy(β); bp[j] += ε
            bm = copy(β); bm[j] -= ε
            batch_grad!(p, bp, all_b, gp); batch_grad!(p, bm, all_b, gm)
            Hfd[:, j] = (gp - gm) ./ (2ε)
        end
        @test norm(H - Hfd) / norm(Hfd) < 1e-6

        # −ln Φ is strictly convex: the weight of a per-observation Hessian lies
        # strictly inside (0, 1). Checked on single observations with x = 1, where
        # the value is the weight itself and no Gram product can supply the sign.
        single(y) = Probit(reshape([1.0], 1, 1), [y])
        @test all(0 < batch_hess(single(y), [t], [1])[1] < 1
                  for y in (0.0, 1.0), t in (-3.0, -0.5, 0.5, 3.0))

        # A batch is a list of observation indices, in any order and with repeats.
        b = [7, 3, 7, 399, 12]
        @test scores(p, β, b) == S[:, b]
        @test loss_terms(p, β, b) == loss_terms(p, β, all_b)[b]
        @test batch_hess(p, β, b) ≈ sum(batch_hess(p, β, [i]) for i in b) ./ length(b) rtol = 1e-12
    end

    @testset "construction" begin
        @test_throws DimensionMismatch Probit(randn(4, 2), [0.0, 1.0, 1.0])
        @test_throws ArgumentError Probit(randn(3, 2), [0.0, 2.0, 1.0])
        @test_throws ArgumentError Probit(K = 0)
        @test_throws ArgumentError Probit(M = 0)
        @test_throws ArgumentError Probit(noise = -1.0)
        @test_throws DimensionMismatch Probit(K = 3, β_star = [1.0, 2.0])

        # Observed data: integer outcomes are accepted, and there is no truth to report.
        q = Probit(randn(MersenneTwister(1), 6, 2), [0, 1, 1, 0, 1, 0])
        @test β_true(q) === nothing
        @test q.noise == 0.0
        @test population(q) == 6

        @test all(==(1.0), Probit(K = 3, M = 50, seed = 4).X[:, 1])
        @test !all(==(1.0), Probit(K = 3, M = 50, seed = 4, intercept = false).X[:, 1])
        @test β_true(Probit(K = 2, M = 10, β_star = [0.5, -0.5])) == [0.5, -0.5]
        @test β_true(Probit(K = 4, M = 10, seed = 1, signal = 2.5)) ==
              2.5 .* β_true(Probit(K = 4, M = 10, seed = 1))

        # One seed gives the probit and the logit the same design and the same β*,
        # so the two links can be compared on it.
        p = Probit(K = 5, M = 400, seed = 3)
        l = LogisticRegression(K = 5, M = 400, seed = 3)
        @test p.X == l.X
        @test β_true(p) == β_true(l)
    end

    @testset "the tails are computed, not underflowed" begin
        # One observation with x = 1, so the index is β itself.
        lo = Probit(reshape([1.0], 1, 1), [0.0])        # y = 0: the loss is −ln Φ(−β)
        hi = Probit(reshape([1.0], 1, 1), [1.0])        # y = 1: the loss is −ln Φ(β)
        t = 40.0

        # Φ(−40) is below the smallest Float64. The probability is not representable,
        # and the loss still is.
        ℓ = loss_terms(lo, [t], [1])[1]
        @test isfinite(ℓ)
        @test exp(-ℓ) == 0.0

        # The reference is the asymptotic series of the Mills ratio,
        #   Φ(−t) = φ(t)/t · (1 − 1/t² + 3/t⁴ − 15/t⁶ + 105/t⁸ − 945/t¹⁰ + …),
        # whose first omitted term, 10395/t¹², is 6e-16 at t = 40.
        series = 1 - 1/t^2 + 3/t^4 - 15/t^6 + 105/t^8 - 945/t^10
        ref = t^2 / 2 + log(t) + log(2π) / 2 - log(series)
        @test abs(ℓ - ref) < 1e-10

        # The score and the curvature against central differences, at the same point.
        ε = 1e-4
        s = scores(lo, [t], [1])[1]
        fd = (loss_terms(lo, [t + ε], [1])[1] - loss_terms(lo, [t - ε], [1])[1]) / (2ε)
        @test abs(s - fd) < 1e-8 * abs(s)
        h = batch_hess(lo, [t], [1])[1]
        hfd = (scores(lo, [t + ε], [1])[1] - scores(lo, [t - ε], [1])[1]) / (2ε)
        @test abs(h - hfd) < 1e-8
        @test 0 < h < 1

        # The other tail: the observation is fitted and contributes nothing.
        @test abs(loss_terms(hi, [t], [1])[1]) < 1e-300
        @test abs(scores(hi, [t], [1])[1]) < 1e-300
        @test abs(batch_hess(hi, [t], [1])[1]) < 1e-300

        # Much further out the curvature tends to 1 on the left and is 0 on the right.
        # The reference on the left is the series 1 − 1/t² + 6/t⁴, whose next term is
        # of size 50/t⁶, 5e-17 at t = 1e3. An interval such as [0, 1] is no reference
        # here: a clamp satisfies it whatever the value.
        for far in (1e3, 1e6, 1e8, 1e12)
            @test isfinite(loss_terms(lo, [far], [1])[1])
            @test isfinite(scores(lo, [far], [1])[1])
            @test abs(batch_hess(lo, [far], [1])[1] - (1 - 1/far^2 + 6/far^4)) < 1e-14
            @test isfinite(loss_terms(hi, [far], [1])[1])
            @test isfinite(scores(hi, [far], [1])[1])
            @test batch_hess(hi, [far], [1])[1] == 0
        end

        # A run started where Φ underflows for a large part of the sample still
        # reaches the estimate found from the origin.
        p = Probit(K = 6, M = 3_000, seed = 13)
        x0 = fill(30.0, 6)
        signed = [(p.y[i] == 1 ? 1.0 : -1.0) * dot(view(p.X, i, :), x0) for i in 1:p.M]
        @test count(<(-38.5), signed) > 100
        @test isfinite(true_objective(p, x0))
        @test all(isfinite, true_gradient(p, x0))
        solve(start) = tr_solve(LikelihoodNLP(p; x0 = start); rule = RDelta(),
                                model = ExactHessian(), subsolver = SteihaugCG(),
                                params = TRParams(tol = 1e-8, max_iterations = 500))
        near = solve(zeros(6)); far = solve(x0)
        @test near.status === :first_order
        @test far.status === :first_order
        @test norm(far.solution - near.solution) < 1e-6
    end

    @testset "the kernels between the origin and the far tails" begin
        # The loss, the score and the curvature of one observation, at indices where
        # neither the derivative checks above nor the far-tail checks look.
        lo = Probit(reshape([1.0], 1, 1), [0.0])
        hi = Probit(reshape([1.0], 1, 1), [1.0])

        # Laplace's continued fraction for the Mills ratio,
        #   (1 − Φ(u))/φ(u) = 1/(u + 1/(u + 2/(u + 3/(u + …)))),
        # evaluated from the tail. No special function is involved.
        function ratio(u)
            r = u
            for k in 400:-1:1
                r = u + k / r
            end
            return 1 / r
        end

        for u in (4.0, 6.0, 8.5, 12.0, 20.0, 30.0, 37.0)
            R = ratio(u); λ = 1 / R
            q = exp(-u^2 / 2) / sqrt(2π) * R            # 1 − Φ(u)
            μ = exp(-u^2 / 2) / sqrt(2π) / (1 - q)      # φ(u)/Φ(u)
            @test loss_terms(lo, [u], [1])[1] ≈ u^2 / 2 + log(2π) / 2 - log(R) rtol = 1e-10
            @test scores(lo, [u], [1])[1]     ≈ λ            rtol = 1e-10
            @test batch_hess(lo, [u], [1])[1] ≈ λ * (λ - u)  rtol = 1e-10
            @test loss_terms(hi, [u], [1])[1] ≈ -log1p(-q)   rtol = 1e-10
            @test scores(hi, [u], [1])[1]     ≈ -μ           rtol = 1e-10
            @test batch_hess(hi, [u], [1])[1] ≈ μ * (μ + u)  rtol = 1e-10
        end
    end

    @testset "the curvature in the left tail, against a high-precision reference" begin
        # With u = −t, Laplace's continued fraction gives λ(t) + t = 1/d with
        # d = u + 2/(u + 3/(u + …)), and λ = u + 1/d. Evaluated in BigFloat to a depth
        # far beyond what convergence needs, it shares no arithmetic with the code.
        function reference(u)
            b = big(u); d = b
            for k in 400:-1:2
                d = b + k / d
            end
            return (b + 1 / d) / d
        end

        worst = 0.0
        for u in exp10.(range(1, 12; length = 600))
            r = reference(u)
            w = TrustRegionRadius._probit_curvature(-u)
            worst = max(worst, Float64(abs(w - r) / r))
        end
        @test worst < 8eps()

        # Control. λ(λ + t) formed as written subtracts two numbers of size u whose
        # difference is 1/u. It returns 0 at u = 1e8, where the curvature is 1.
        direct(t) = (λ = TrustRegionRadius._mills(t); clamp(λ * (λ + t), 0.0, 1.0))
        @test abs(direct(-1e8) - reference(1e8)) > 0.5
        @test abs(TrustRegionRadius._probit_curvature(-1e8) - reference(1e8)) < 8eps()

        # The two branches meet at t = −10 without a step.
        below = TrustRegionRadius._probit_curvature(prevfloat(-10.0))
        above = TrustRegionRadius._probit_curvature(nextfloat(-10.0))
        @test abs(below - above) < 1e-12
    end

    @testset "the identity is exact conditional on x" begin
        rng = MersenneTwister(5)
        m = 40
        X = randn(rng, m, 3)
        # Each row twice, once with each outcome: rows i and m + i share xᵢ.
        p = Probit(vcat(X, X), vcat(ones(m), zeros(m)))
        β = [0.4, -0.7, 0.3]

        ℓ = loss_terms(p, β, 1:2m)
        P1 = exp.(-ℓ[1:m]); P0 = exp.(-ℓ[m+1:2m])
        # The two outcomes exhaust the probability.
        @test maximum(abs.(P1 .+ P0 .- 1)) < 1e-14

        # Under the probabilities of β, the expected score and the gap between the
        # expected outer product and the expected Hessian, both taken at `at`.
        function conditional(at)
            S = scores(p, at, 1:2m)
            mean_score = 0.0; gap = 0.0
            for i in 1:m
                s1 = S[:, i]; s0 = S[:, m + i]
                mean_score = max(mean_score, norm(P1[i] .* s1 .+ P0[i] .* s0))
                V = P1[i] .* (s1 * s1') .+ P0[i] .* (s0 * s0')
                H = P1[i] .* batch_hess(p, at, [i]) .+ P0[i] .* batch_hess(p, at, [m + i])
                gap = max(gap, norm(V - H) / norm(H))
            end
            return (mean_score = mean_score, gap = gap)
        end

        at_truth = conditional(β)
        @test at_truth.mean_score < 1e-14
        @test at_truth.gap < 1e-12

        # Control. Scores and Hessians taken at another point do not satisfy it.
        elsewhere = conditional(β .+ 0.5)
        @test elsewhere.mean_score > 0.1
        @test elsewhere.gap > 0.1

        # The probit's observed information depends on the outcome. The logit's does not.
        @test norm(batch_hess(p, β, [1]) - batch_hess(p, β, [m + 1])) >
              1e-3 * norm(batch_hess(p, β, [1]))
        l = LogisticRegression(vcat(X, X), vcat(ones(m), zeros(m)))
        @test batch_hess(l, β, [1]) == batch_hess(l, β, [m + 1])
    end

    @testset "the identity as a rate in M" begin
        K = 5
        Ms = [500, 2_000, 8_000, 32_000]
        # The rate is a statement about the mean error. One sample at each M follows
        # it only roughly, so each entry is a mean over eight samples.
        seeds = 1:8
        at_true = Float64[]; at_wrong = Float64[]
        for M in Ms
            t = 0.0; w = 0.0
            for seed in seeds
                p = Probit(K = K, M = M, seed = seed)
                β = β_true(p)
                t += information_identity_error(p, β).B_err
                w += information_identity_error(p, β .+ 1.5).B_err
            end
            push!(at_true, t / length(seeds)); push!(at_wrong, w / length(seeds))
        end
        # At β*: monotone decay at the rate M^{-1/2}, so err·√M stays bounded.
        @test issorted(at_true; rev = true)
        @test at_true[end] < at_true[1] / 4
        scaled = at_true .* sqrt.(Ms)
        @test maximum(scaled) / minimum(scaled) < 3

        # Away from β*: no decay.
        @test all(>(0.5), at_wrong)
        @test at_wrong[end] > at_wrong[1] / 2

        # W and B agree at a stationary point and differ by exactly ḡḡᵀ.
        p = Probit(K = K, M = 8_000, seed = 1)
        e = information_identity_error(p, β_true(p))
        @test e.BW_gap < 0.05
        @test e.BW_gap ≈ e.grad_norm^2 / norm(batch_hess(p, β_true(p), 1:p.M))

        # The origin is an exception to "away from β*". Every index is zero there, so
        # each score outer product equals the per-observation Hessian, (2/π) xxᵀ for
        # the probit and ¼ xxᵀ for the logit, and B is the exact Hessian whatever the
        # data, misspecified or not.
        for q in (Probit(K = K, M = 2_000, seed = 1),
                  Probit(K = K, M = 2_000, seed = 1, noise = 1.5),
                  LogisticRegression(K = K, M = 2_000, seed = 1))
            @test information_identity_error(q, zeros(K)).B_err < 1e-14
        end
        q = Probit(K = K, M = 2_000, seed = 1)
        @test batch_hess(q, zeros(K), 1:q.M) ≈ (2 / π) .* (q.X' * q.X) ./ q.M rtol = 1e-13
    end

    @testset "noise breaks the identity" begin
        err(M, σ) = (p = Probit(K = 6, M = M, seed = 3, noise = σ);
                     information_identity_error(p, β_true(p)).B_err)
        # Random coefficients make the generating process heteroscedastic. The
        # identity then fails at β* itself, and a larger sample does not repair it.
        @test err(4_000, 1.5) > err(4_000, 0.0)
        @test err(32_000, 1.5) > err(4_000, 1.5) / 2
        # The correctly specified problem is the one whose error shrinks.
        @test err(32_000, 0.0) < err(4_000, 0.0)
        @test Probit(K = 3, M = 20, noise = 0.5).noise == 0.5

        # β* is then far from the minimiser of f, so the error at β* mixes two
        # things. At the minimiser itself the identity still fails: the error
        # levels off as M grows, where the correctly specified one keeps shrinking.
        function at_minimiser(M, σ)
            p = Probit(K = 5, M = M, seed = 3, noise = σ)
            st = tr_solve(LikelihoodNLP(p; x0 = zeros(p.n)); rule = RDelta(),
                          model = ExactHessian(), subsolver = SteihaugCG(),
                          params = TRParams(tol = 1e-8, max_iterations = 500))
            @test st.status === :first_order
            return information_identity_error(p, st.solution).B_err
        end
        small_ok, large_ok   = at_minimiser(8_000, 0.0), at_minimiser(512_000, 0.0)
        small_mis, large_mis = at_minimiser(8_000, 1.5), at_minimiser(512_000, 1.5)
        @test large_ok < small_ok / 2
        @test large_mis > small_mis / 2
        @test large_mis > 3 * large_ok

        # With K = 1 and an intercept every xₙ is 1, the added variance is the same
        # for all observations, and the noise only rescales the coefficient: the
        # model is correctly specified at β*/√(1 + σ²), and the error there shrinks.
        rescaled(M) = (p = Probit(K = 1, M = M, seed = 1, noise = 1.5);
                       information_identity_error(p, β_true(p) ./ sqrt(1 + 1.5^2)).B_err)
        @test rescaled(128_000) < 0.01
        @test rescaled(128_000) < rescaled(8_000)

        # The scale of the noise, in closed form. With one coefficient and x = 1 the
        # latent index is β* + σε + e, of variance 1 + σ², so the frequency of y = 1
        # is Φ(β*/√(1 + σ²)). Its standard error at this M is 0.001.
        unit = Probit(reshape([1.0], 1, 1), [1.0])
        Φ(z) = exp(-loss_terms(unit, [z], [1])[1])
        p = Probit(K = 1, M = 200_000, seed = 0, β_star = [1.0], noise = 1.5)
        @test abs(sum(p.y) / p.M - Φ(1 / sqrt(1 + 1.5^2))) < 0.01
    end

    @testset "every model reaches the same estimate" begin
        p = Probit(K = 6, M = 3_000, seed = 13)
        ref = nothing
        for m in (ExactHessian(), BHHHModel(ridge = 1e-10), BHHH2Model(ridge = 1e-10))
            nlp = LikelihoodNLP(p; x0 = zeros(6))
            st = tr_solve(nlp; rule = RDelta(), model = m, subsolver = SteihaugCG(),
                          params = TRParams(tol = 1e-8, max_iterations = 500))
            @test st.status === :first_order
            ref === nothing ? (ref = st.solution) : @test(st.solution ≈ ref, rtol = 1e-4)
        end
        # The estimate is near the generating parameters and is not them, and it
        # fits the sample at least as well as they do.
        @test 0 < norm(ref .- β_true(p)) < 0.5
        @test true_objective(p, ref) <= true_objective(p, β_true(p))

        # A supplied β_star is the source of the data, and not only a stored field.
        bs = [0.5, -0.5, 1.0]
        q = Probit(K = 3, M = 20_000, seed = 7, β_star = bs)
        est = tr_solve(LikelihoodNLP(q; x0 = zeros(3)); rule = RDelta(),
                       model = ExactHessian(), subsolver = SteihaugCG(),
                       params = TRParams(tol = 1e-8, max_iterations = 500)).solution
        @test norm(est - bs) < 0.1
    end

    @testset "a sampled run, and what its stopping test is worth" begin
        p = Probit(K = 5, M = 2_000, seed = 2)
        sampled(stop, seed) =
            tr_solve(FiniteSumNLP(p, RadiusProportional(); x0 = zeros(5), seed = seed);
                     rule = RDelta(), model = BHHHModel(ridge = 1e-8),
                     subsolver = SteihaugCG(), trace = true,
                     params = TRParams(tol = 1e-6, max_iterations = 1_000,
                                       true_stop = stop))
        truth(st) = norm(true_gradient(p, st.solution))
        # The estimate, from a deterministic run the sampled solver takes no part in.
        best = tr_solve(LikelihoodNLP(p; x0 = zeros(5)); rule = RDelta(),
                        model = ExactHessian(), subsolver = SteihaugCG(),
                        params = TRParams(tol = 1e-8, max_iterations = 500)).solution

        # Control. φ/Φ decays like a Gaussian, so on a batch of two the gradient
        # vanishes to rounding once both observations are fitted with a margin, and
        # an unconfirmed run then reports :first_order at a point that is not
        # critical. Whether a run does so depends on its draws: of the six oracle
        # seeds below, four do.
        loose = [sampled(:none, s) for s in 1:6]
        @test count(st -> st.status === :first_order && truth(st) > 1e-2, loose) >= 1

        # Confirmed on the full batch, a run that reports :first_order is at the
        # estimate. We measured all six doing so. Oracle seed 2 stalls at iteration
        # 15 on a batch of two before any candidate stop, under :none as well, and
        # under :full it continues on the full batch from iteration 16. A
        # confirmed run stalls only on the full batch, and a stall that does not end
        # the run keeps the trajectories in step.
        tight = [sampled(:full, s) for s in 1:6]
        @test count(st -> st.status === :first_order, tight) >= 1
        for st in tight
            ss = st.solver_specific
            @test st.status !== :first_order || norm(st.solution - best) < 1e-4
            @test st.status !== :stalled || ss[:grad_sample_trajectory][end] == population(p)
            @test length(ss[:ratio_trajectory]) == length(ss[:grad_sample_trajectory]) == st.iter
            @test length(ss[:delta_trajectory]) == length(ss[:true_grad_trajectory]) == st.iter + 1
        end
    end
end
