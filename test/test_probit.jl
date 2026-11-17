# Binary probit, checked against its definition and not against itself:
#
#   derivatives   against central differences
#   the tails     against the asymptotic series, where Φ itself is not representable
#   the identity  exactly, conditional on x, and then as a rate in M
#   `noise > 0`   breaks the identity at β* itself
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
        # from the array the gradient came from.
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
        @test norm(H - H') < 1e-14
        # −ln Φ is convex, so every per-observation Hessian is positive semidefinite.
        @test minimum(eigvals(Symmetric(H))) > 0
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

        # Much further out, where λ + t cancels: finite, and the curvature stays in [0, 1].
        for far in (1e3, 1e6)
            for q in (lo, hi)
                @test isfinite(loss_terms(q, [far], [1])[1])
                @test isfinite(scores(q, [far], [1])[1])
                @test 0 <= batch_hess(q, [far], [1])[1] <= 1
            end
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
        at_true = Float64[]; at_wrong = Float64[]
        for M in Ms
            p = Probit(K = K, M = M, seed = 1)
            β = β_true(p)
            push!(at_true,  information_identity_error(p, β).B_err)
            push!(at_wrong, information_identity_error(p, β .+ 1.5).B_err)
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
        @test isapprox(e.W_err, e.B_err; atol = 0.05)
    end

    @testset "noise breaks the identity at β*" begin
        err(M, σ) = (p = Probit(K = 6, M = M, seed = 3, noise = σ);
                     information_identity_error(p, β_true(p)).B_err)
        # Random coefficients make the generating process heteroscedastic. The
        # identity then fails at β* itself, and a larger sample does not repair it.
        @test err(4_000, 1.5) > err(4_000, 0.0)
        @test err(32_000, 1.5) > err(4_000, 1.5) / 2
        # The correctly specified problem is the one whose error shrinks.
        @test err(32_000, 0.0) < err(4_000, 0.0)
        @test Probit(K = 3, M = 20, noise = 0.5).noise == 0.5
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
    end

    @testset "a sampled run, and what its stopping test is worth" begin
        p = Probit(K = 5, M = 2_000, seed = 2)
        solve(stop) = tr_solve(FiniteSumNLP(p, RadiusProportional(); x0 = zeros(5), seed = 1);
                               rule = RDelta(), model = BHHHModel(ridge = 1e-8),
                               subsolver = SteihaugCG(),
                               params = TRParams(tol = 1e-6, max_iterations = 1_000,
                                                 true_stop = stop))

        # Control. φ/Φ decays like a Gaussian, so on a batch of two the gradient
        # vanishes to rounding once both observations are fitted with a margin. The
        # unconfirmed run reports :first_order at a point that is not critical.
        loose = solve(:none)
        @test loose.status === :first_order
        @test norm(true_gradient(p, loose.solution)) > 1e-2

        # Confirmed on the full batch, a :first_order status means what it says.
        tight = solve(:full)
        @test tight.status === :first_order
        @test norm(true_gradient(p, tight.solution)) <= 1e-6
    end
end
