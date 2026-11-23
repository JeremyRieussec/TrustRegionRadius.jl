# Linear regression with a known error scale, checked against its definition and
# not against itself:
#
#   derivatives      against central differences, and the Hessian against XᵀX/(Mσ²)
#   Gauss-Newton     is the exact Hessian, to the bit
#   the BHHH error   in closed form at any β, exactly zero conditional on x, then as
#                    a rate in M with its constant
#   `noise > 0`      breaks the identity at β* and leaves β* the minimiser
#   a wrong σ        multiplies B by a known factor, at β* itself
#
# Every reference is built from X, y and σ, never from the functions under test.
# An assertion that could pass for the wrong reason comes with a control.

@testset "linear regression" begin

    # (1/N) Σ (rₙ² − 1) xₙxₙᵀ/σ² with rₙ = (xₙᵀβ − yₙ)/σ, from the data alone.
    gap_reference(X, y, σ, β) =
        (X' * (((((X * β .- y) ./ σ) .^ 2) .- 1) .* X)) ./ (size(X, 1) * σ^2)
    # √E[B_err² | X] at β* under Gaussian errors of the right variance.
    rms_at_truth(X) =
        sqrt(2 * sum(sum(abs2, view(X, i, :))^2 for i in 1:size(X, 1))) / norm(X' * X)

    @testset "interface and derivatives" begin
        σ = 0.7
        p = LinearRegression(K = 5, M = 400, seed = 3, σ = σ)
        @test p isa NLSProblem
        @test p isa LikelihoodProblem
        @test population(p) == 400
        @test has_scores(p)
        @test supports_paired(p)
        @test full_batch(p) == collect(1:400)

        all_b = collect(1:p.M)
        # At β = 0 the loss is mean(y²)/(2σ²): a check that needs no reference value.
        @test abs(batch_obj(p, zeros(p.n), all_b) - sum(abs2, p.y) / (2σ^2 * p.M)) < 1e-12

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

        S = scores(p, β, all_b)
        @test S isa Matrix{Float64}
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
        # The closed form, written from the design and σ.
        Hdef = p.X' * p.X ./ (p.M * σ^2)
        @test norm(H - Hdef) / norm(Hdef) < 1e-13
        @test norm(H - H') < 1e-14
        @test minimum(eigvals(Symmetric(H))) > 0

        # The Hessian involves neither β nor y.
        @test batch_hess(p, zeros(p.n), all_b) == H
        @test batch_hess(LinearRegression(p.X, reverse(p.y); σ = σ), β, all_b) == H
        # Control. On the same design the logit Hessian moves with β, and the
        # probit Hessian moves with the outcomes.
        l = LogisticRegression(K = 5, M = 400, seed = 3)
        @test batch_hess(l, zeros(5), all_b) != batch_hess(l, β, all_b)
        b = Probit(K = 5, M = 400, seed = 3)
        @test batch_hess(Probit(b.X, 1 .- b.y), β, all_b) != batch_hess(b, β, all_b)

        # The loss is a negative log density less ½ln(2πσ²): exp(−F) integrates
        # to √(2π)σ over y.
        ys = range(-12.0, 12.0; length = 20_001)
        grid = LinearRegression(reshape(ones(length(ys)), :, 1), collect(ys); σ = σ)
        mass = step(ys) * sum(exp.(-loss_terms(grid, [0.4], 1:length(ys))))
        @test abs(mass - sqrt(2π) * σ) < 1e-10

        # The objective is an exact quadratic about the least-squares estimate.
        βhat = p.X \ p.y
        @test abs(true_objective(p, β) - true_objective(p, βhat) -
                  0.5 * dot(β - βhat, H * (β - βhat))) < 1e-12
        @test norm(true_gradient(p, βhat)) < 1e-12
    end

    @testset "construction" begin
        @test_throws DimensionMismatch LinearRegression(randn(4, 2), [0.0, 1.0, 1.0]; σ = 1.0)
        @test_throws ArgumentError LinearRegression(randn(3, 2), [1.0, NaN, 0.0]; σ = 1.0)
        @test_throws ArgumentError LinearRegression(randn(3, 2), [1.0, Inf, 0.0]; σ = 1.0)
        @test_throws ArgumentError LinearRegression(randn(3, 2), randn(3); σ = 1.0, noise = -1.0)
        @test_throws DimensionMismatch LinearRegression(randn(3, 2), randn(3); σ = 1.0, β_star = [1.0])
        # A scale that is not a positive finite number is refused by both forms.
        for bad in (0.0, -1.0, Inf, NaN)
            @test_throws ArgumentError LinearRegression(randn(3, 2), randn(3); σ = bad)
            @test_throws ArgumentError LinearRegression(σ = bad)
        end
        @test_throws ArgumentError LinearRegression(K = 0)
        @test_throws ArgumentError LinearRegression(M = 0)
        @test_throws ArgumentError LinearRegression(noise = -1.0)
        @test_throws DimensionMismatch LinearRegression(K = 3, β_star = [1.0, 2.0])
        # Two refusals that a later line would also make, under another name. They are
        # made here, and say what is wrong.
        @test_throws "need 0 < σ < Inf" LinearRegression(σ = Inf)
        @test_throws "β_star must have length K = 3" LinearRegression(K = 3, β_star = [1.0, 2.0])
        # The keyword form does not build outcomes the data form would refuse.
        @test_throws ArgumentError LinearRegression(noise = Inf, M = 20)
        @test_throws ArgumentError LinearRegression(signal = NaN, M = 20)
        @test_throws ArgumentError LinearRegression(K = 2, M = 20, β_star = [NaN, 1.0])
        # Observed data come with no σ, and the constructor does not guess one.
        @test_throws UndefKeywordError LinearRegression(randn(3, 2), randn(3))
        @test LinearRegression(K = 3, M = 20).σ == 1.0

        # Observed data: integer outcomes are accepted, and there is no truth to report.
        Xq = randn(MersenneTwister(1), 6, 2)
        q = LinearRegression(Xq, [0, 1, 1, 0, 1, 0]; σ = 2.0)
        @test β_true(q) === nothing
        @test q.σ == 2.0
        @test q.noise == 0.0
        @test population(q) == 6
        # What the data form is given, it keeps: a copy of the data, and the truth and
        # the generating noise when the caller knows them.
        yq = [0.5, -1.0, 2.0, 0.0, 1.5, -0.5]
        q = LinearRegression(Xq, yq; σ = 2.0, β_star = [0.5, -0.5], noise = 0.3)
        @test (q.n, q.M, q.σ, q.noise) == (2, 6, 2.0, 0.3)
        @test q.X == Xq && q.y == yq
        @test q.X !== Xq && q.y !== yq
        @test β_true(q) == [0.5, -0.5]

        # The generator, rebuilt here from the seed in the documented order: β*, the
        # design, the coefficient perturbations when there is noise, the errors. The
        # three settings move signal, noise, σ and the intercept away from their defaults.
        for (K, M, seed, signal, noise, σ, intercept) in
                ((4, 60, 5, 2.5, 0.7, 0.3, true), (3, 40, 9, 0.5, 1.2, 2.0, false),
                 (5, 30, 2, 1.0, 0.0, 1.0, true))
            g = LinearRegression(K = K, M = M, seed = seed, signal = signal, noise = noise,
                                 σ = σ, intercept = intercept)
            rng = MersenneTwister(seed)
            β = signal .* randn(rng, K)
            X = randn(rng, M, K)
            intercept && (X[:, 1] .= 1.0)
            z = X * β
            noise > 0 && (z .+= noise .* vec(sum(X .* randn(rng, M, K); dims = 2)))
            y = z .+ σ .* randn(rng, M)
            @test β_true(g) == β
            @test g.X == X
            @test g.y == y
            @test (g.σ, g.noise) == (σ, noise)
        end

        # A supplied β_star is the source of the data and not a label: the estimate
        # lands on it, on the design the probit draws for the same arguments.
        bs = [0.5, -0.5, 1.0]
        s  = LinearRegression(K = 3, M = 20_000, seed = 7, β_star = bs)
        sp = Probit(K = 3, M = 20_000, seed = 7, β_star = bs)
        @test s.X == sp.X
        @test sp.y == Float64.(s.y .> 0)
        @test norm(s.X \ s.y - bs) < 0.05
        # Control. A vector 0.1 away is told apart at this sample size.
        @test norm(s.X \ s.y - [0.5, -0.5, 0.9]) > 0.05

        @test all(==(1.0), LinearRegression(K = 3, M = 50, seed = 4).X[:, 1])
        @test !all(==(1.0), LinearRegression(K = 3, M = 50, seed = 4, intercept = false).X[:, 1])
        @test β_true(LinearRegression(K = 2, M = 10, β_star = [0.5, -0.5])) == [0.5, -0.5]
        # `signal` scales the β* that is drawn, and leaves a supplied one alone.
        @test β_true(LinearRegression(K = 2, M = 10, β_star = [0.5, -0.5], signal = 3.0)) == [0.5, -0.5]
        r = LinearRegression(K = 3, M = 20, σ = 0.25, noise = 0.5)
        @test r.σ == 0.25 && r.noise == 0.5
        # Fewer observations than coefficients is legal, and the Hessian is singular.
        @test rank(batch_hess(LinearRegression(K = 5, M = 3, seed = 1), zeros(5), 1:3)) == 3

        # One seed gives the regression, the probit and the logit the same design
        # and the same β*. At σ = 1 the probit outcome is the sign of y.
        for noise in (0.0, 0.8)
            p  = LinearRegression(K = 5, M = 400, seed = 3, noise = noise)
            pr = Probit(K = 5, M = 400, seed = 3, noise = noise)
            l  = LogisticRegression(K = 5, M = 400, seed = 3)
            @test p.X == pr.X == l.X
            @test β_true(p) == β_true(pr) == β_true(l)
            @test pr.y == Float64.(p.y .> 0)
        end
        # σ rescales the errors and nothing else.
        a = LinearRegression(K = 5, M = 400, seed = 3, σ = 0.25)
        b = LinearRegression(K = 5, M = 400, seed = 3)
        @test a.X == b.X && β_true(a) == β_true(b)
        @test norm((a.y - a.X * β_true(a)) ./ 0.25 - (b.y - b.X * β_true(b))) < 1e-12
        # Control. With another σ the outcome is no longer the probit's latent variable.
        @test Probit(K = 5, M = 400, seed = 3).y != Float64.(a.y .> 0)
    end

    @testset "σ = 1 is linear_least_squares, with the Hessian in closed form" begin
        σ = 0.7
        p = LinearRegression(K = 5, M = 400, seed = 3, σ = σ)
        β = 0.3 .* randn(MersenneTwister(11), 5); b = [3, 9, 9, 250, 400]
        ls = linear_least_squares(p.X, p.y)
        p1 = LinearRegression(p.X, p.y; σ = 1.0)
        for batch in (b, 1:p.M)
            @test loss_terms(p1, β, batch) == loss_terms(ls, β, batch)
            @test scores(p1, β, batch) == scores(ls, β, batch)
            @test residuals(p1, β, batch) == residuals(ls, β, batch)
            @test jacobian(p1, β, batch) == jacobian(ls, β, batch)
        end
        # What changes is the Hessian: closed form in place of central differences.
        H1 = batch_hess(p1, β, 1:p.M)
        @test norm(H1 - batch_hess(ls, β, 1:p.M)) / norm(H1) < 1e-8
        @test H1 == (p.X' * p.X) ./ p.M
        # σ divides the objective and the scores by σ², the outer product by σ⁴.
        # That is why the identity can hold for one σ only.
        @test abs(batch_obj(p, β, b) * σ^2 - batch_obj(ls, β, b)) < 1e-13 * batch_obj(ls, β, b)
        Sp = scores(p, β, b); Sl = scores(ls, β, b)
        @test norm(Sp .* σ^2 - Sl) < 1e-13 * norm(Sl)
        @test norm((Sp * Sp') .* σ^4 - Sl * Sl') < 1e-13 * norm(Sl * Sl')
    end

    @testset "Gauss-Newton is the exact Hessian, to the bit" begin
        p = LinearRegression(K = 5, M = 400, seed = 3, σ = 0.7)
        nlp = FullBatchNLP(p)
        for x in (zeros(5), β_true(p), 10 .* randn(MersenneTwister(4), 5))
            @test Matrix(dense_hessian(GaussNewtonModel(), nlp, x)) == batch_hess(p, x, 1:p.M)
            @test Matrix(dense_hessian(ExactHessian(), nlp, x)) == batch_hess(p, x, 1:p.M)
            # Nothing is discarded, so the measured Gauss-Newton error is zero and
            # not merely small, whatever the size of the residuals.
            e = gauss_newton_error(p, x)
            @test e.GN_err == 0.0
            @test e.residual ≈ norm((p.X * x .- p.y) ./ p.σ) / sqrt(p.M)
            @test !e.rank_def
        end
        # Control. On a fit that is not linear in its parameters the same measure
        # is not zero.
        nl = exponential_fit(n_terms_model = 1, M = 400, misfit = 0.5, seed = 2)
        @test gauss_newton_error(nl, [0.9, 0.6]).GN_err > 0.05
        # Control. BHHH is an outer product too, and it is not the Hessian.
        H = batch_hess(p, β_true(p), 1:p.M)
        @test norm(Matrix(dense_hessian(BHHHModel(), nlp, β_true(p))) - H) / norm(H) > 0.01
    end

    @testset "the BHHH error in closed form" begin
        # B − H = (1/N) Σ (rₙ² − 1) xₙxₙᵀ/σ², at any β, on any data, on any batch.
        # The rounding error of the left side is of the size of eps·‖B‖, and ‖B‖/‖H‖
        # grows like the squared residual, so the miss is measured against ‖B‖ + ‖H‖.
        # The third setting has a small σ, where ‖B‖ is some 50 000 times ‖H‖.
        worst = 0.0; control = Inf
        for (seed, σ, noise) in ((3, 0.7, 0.0), (8, 2.0, 0.6), (3, 0.05, 0.0))
            p = LinearRegression(K = 5, M = 400, seed = seed, σ = σ, noise = noise)
            for b in (collect(1:p.M), [4, 4, 19, 77, 200, 311, 400])
                for β in (β_true(p), zeros(5), 3 .* randn(MersenneTwister(1), 5))
                    S = scores(p, β, b); B = S * S' ./ length(b); H = batch_hess(p, β, b)
                    G = gap_reference(p.X[b, :], p.y[b], σ, β)
                    worst = max(worst, norm((B - H) - G) / (norm(B) + norm(H)))
                    # Control. The reference written with another σ does not match.
                    Gw = gap_reference(p.X[b, :], p.y[b], 1.1σ, β)
                    control = min(control, norm((B - H) - Gw) / (norm(B) + norm(H)))
                end
            end
        end
        @test worst < 1e-13
        @test control > 0.03

        # The origin is where a run starts by default. The two binary models have an
        # exact BHHH model there. The regression does not: its residual at zero is the
        # outcome, and the error is of the size of ‖β*‖²/σ².
        origin(make) = information_identity_error(make(K = 5, M = 2_000, seed = 1), zeros(5)).B_err
        @test origin(LogisticRegression) < 1e-12
        @test origin(Probit) < 1e-12
        @test origin(LinearRegression) > 1
    end

    @testset "the identity is exact conditional on x" begin
        rng = MersenneTwister(5)
        m = 40; σ = 0.7
        X = randn(rng, m, 3)
        β = [0.4, -0.7, 0.3]
        # Each row twice, with errors +σ and −σ: mean zero, variance σ², and
        # rₙ² = 1 at β. No normality is needed, only the first two moments.
        p = LinearRegression(vcat(X, X), vcat(X * β .+ σ, X * β .- σ); σ = σ)
        e = information_identity_error(p, β)
        @test e.B_err < 1e-13
        @test e.W_err < 1e-13
        @test e.grad_norm < 1e-13

        # Control. Away from β the gap is (1/m) Σ (xᵀδ)² xxᵀ/σ⁴: not small, and
        # positive definite, so BHHH overstates the curvature.
        δ = [0.5, 0.5, 0.5]
        e2 = information_identity_error(p, β .+ δ)
        G = (X' * (((X * δ) .^ 2) .* X)) ./ (m * σ^4); H = X' * X ./ (m * σ^2)
        @test e2.B_err > 0.1
        @test abs(e2.B_err - norm(G) / norm(H)) < 1e-12
        @test minimum(eigvals(Symmetric(G))) > 0
        @test abs(e2.grad_norm - norm(H * δ)) < 1e-12
        @test abs(e2.BW_gap - norm(H * δ)^2 / norm(H)) < 1e-12

        # A wrong σ. Posed with cσ the same data give B = H/c² at β, so the error
        # is |1/c² − 1| there, and an interior BHHH step beside β has ρ = 2 − c².
        for c in (2.0, 0.5)
            q = LinearRegression(p.X, p.y; σ = c * σ)
            @test abs(information_identity_error(q, β).B_err - abs(1 / c^2 - 1)) < 1e-12
        end
        for c in (0.5, 1.0, 2.0)
            q = LinearRegression(p.X, p.y; σ = c * σ)
            st = tr_solve(LikelihoodNLP(q; x0 = β .+ 1e-3); rule = RDelta(),
                          model = BHHHModel(), subsolver = ExactMS(), trace = true,
                          params = TRParams(tol = 1e-12, max_iterations = 1, Δ0 = 10.0))
            @test abs(st.solver_specific[:ratio_trajectory][1] - (2 - c^2)) < 1e-3
            # The step that the right σ takes with ρ = 1 is refused under 2σ.
            @test st.solver_specific[:accepted_trajectory][1] == (c <= 1)
        end
    end

    @testset "the identity as a rate in M" begin
        K = 5
        Ms = [500, 2_000, 8_000, 32_000]
        at_true = Float64[]; at_wrong = Float64[]; predicted = Float64[]; wrong_ref = Float64[]
        for M in Ms
            p = LinearRegression(K = K, M = M, seed = 1)
            β = β_true(p); δ = fill(1.5, K)
            push!(at_true,  information_identity_error(p, β).B_err)
            push!(at_wrong, information_identity_error(p, β .+ δ).B_err)
            push!(predicted, rms_at_truth(p.X))
            G = (p.X' * (((p.X * δ) .^ 2) .* p.X)) ./ M; H = p.X' * p.X ./ M
            push!(wrong_ref, norm(G) / norm(H))
        end
        # At β*: decay at the rate M^{-1/2}, so err·√M stays bounded.
        @test issorted(at_true; rev = true)
        @test at_true[end] < at_true[1] / 4
        scaled = at_true .* sqrt.(Ms)
        @test maximum(scaled) / minimum(scaled) < 3
        # The size is known from the design alone: E[B_err² | X] = 2Σ‖xₙ‖⁴/‖XᵀX‖².
        @test all(r -> 0.5 < r < 2, at_true ./ predicted)

        # Control, away from β*: no decay, and the limit is known as well.
        @test all(>(0.5), at_wrong)
        @test at_wrong[end] > at_wrong[1] / 2
        @test all(abs.(at_wrong .- wrong_ref) .< 0.05 .* wrong_ref)

        # W and B agree at a stationary point and differ by exactly ḡḡᵀ.
        p = LinearRegression(K = K, M = 8_000, seed = 1)
        e = information_identity_error(p, β_true(p))
        @test e.BW_gap < 0.05
        @test isapprox(e.W_err, e.B_err; atol = 0.05)

        # Over seeds, the root mean square of the error at β* is the prediction.
        errs = Float64[]; preds = Float64[]
        for seed in 1:40
            q = LinearRegression(K = K, M = 2_000, seed = seed)
            push!(errs, information_identity_error(q, β_true(q)).B_err)
            push!(preds, rms_at_truth(q.X))
        end
        @test 0.8 < sqrt(mean(abs2, errs)) / sqrt(mean(abs2, preds)) < 1.25
    end

    @testset "noise breaks the identity at β*, and β* stays the minimiser" begin
        function err(M, ν; σ = 1.0)
            p = LinearRegression(K = 6, M = M, seed = 3, noise = ν, σ = σ)
            w = [sum(abs2, view(p.X, i, :)) for i in 1:M]
            G = (p.X' * (w .* p.X)) .* (ν^2 / (M * σ^4)); H = p.X' * p.X ./ (M * σ^2)
            return (measured = information_identity_error(p, β_true(p)).B_err,
                    given_X = norm(G) / norm(H))
        end
        a0 = err(4_000, 0.0); a1 = err(4_000, 0.5); b0 = err(32_000, 0.0); b1 = err(32_000, 0.5)
        # Random coefficients make the errors heteroscedastic. The identity then
        # fails at β* itself, and a larger sample does not repair it.
        @test a1.measured > a0.measured
        @test b1.measured > a1.measured / 2
        # The correctly specified problem is the one whose error shrinks.
        @test b0.measured < a0.measured
        # The size of the break is known: (ν²/σ⁴)(1/M) Σ ‖xₙ‖² xₙxₙᵀ, relative to H.
        @test abs(a1.measured - a1.given_X) < 0.1 * a1.given_X
        @test abs(b1.measured - b1.given_X) < 0.1 * b1.given_X
        # The same at another σ, where the powers of σ in the reference are exercised:
        # the perturbation has standard deviation ν and not νσ, and the gap carries σ⁴.
        c1 = err(32_000, 0.5; σ = 0.5)
        @test abs(c1.measured - c1.given_X) < 0.1 * c1.given_X
        # Control. The size at σ = 1 is not the size at σ = 0.5.
        @test abs(c1.measured - b1.given_X) > 0.5 * b1.given_X

        # Most of the break is a wrong scale. Posed again with the scale of their own
        # residuals, the noisy data keep only what heteroscedasticity makes, which
        # given the design is (1/(M s⁴)) Σ (σ² + ν²‖xₙ‖² − s²) xₙxₙᵀ under the scale s.
        noisy = LinearRegression(K = 6, M = 256_000, seed = 3, noise = 0.5)
        βs = β_true(noisy); Xn = noisy.X; Mn = noisy.M
        s = sqrt(sum(abs2, noisy.y - Xn * (Xn \ noisy.y)) / Mn)
        reposed = LinearRegression(Xn, noisy.y; σ = s, β_star = βs, noise = 0.5)
        sq = [sum(abs2, view(Xn, i, :)) for i in 1:Mn]
        left  = (Xn' * ((1 .+ 0.5^2 .* sq .- s^2) .* Xn)) ./ (Mn * s^4)
        scale = norm(left) / norm(Xn' * Xn ./ (Mn * s^2))
        whole = information_identity_error(noisy, βs).B_err
        rest  = information_identity_error(reposed, βs).B_err
        @test rest < whole / 5
        @test abs(rest - scale) < 0.1 * scale
        # The remainder is what no single scale removes, and it is not nothing.
        @test rest > 0.1

        # With K = 1 and an intercept every xₙ is 1, so nothing is heteroscedastic.
        # The noise is then a wrong scale and nothing else: the error at β* is ν²/σ²,
        # and the same data posed with √(σ² + ν²) are a correctly specified regression.
        flat = LinearRegression(K = 1, M = 128_000, seed = 3, noise = 1.5)
        @test all(==(1.0), flat.X)
        @test abs(information_identity_error(flat, β_true(flat)).B_err - 1.5^2) < 0.03 * 1.5^2
        rescaled = LinearRegression(flat.X, flat.y; σ = sqrt(1 + 1.5^2),
                                    β_star = β_true(flat), noise = 1.5)
        @test information_identity_error(rescaled, β_true(rescaled)).B_err < 0.02

        # The knob does not touch the Hessian.
        @test batch_hess(LinearRegression(K = 6, M = 4_000, seed = 3, noise = 0.5), zeros(6), 1:4_000) ==
              batch_hess(LinearRegression(K = 6, M = 4_000, seed = 3), zeros(6), 1:4_000)

        # The errors keep mean zero given x, so β* remains the minimiser in the
        # population. Control: under the same knob the probit's β* does not.
        lr = LinearRegression(K = 5, M = 64_000, seed = 7, noise = 1.0)
        pb = Probit(K = 5, M = 64_000, seed = 7, noise = 1.0)
        @test norm(true_gradient(lr, β_true(lr))) < 0.05
        @test norm(true_gradient(pb, β_true(pb))) > 0.2
    end

    @testset "every model reaches the least-squares estimate" begin
        p = LinearRegression(K = 6, M = 3_000, seed = 13)
        βhat = p.X \ p.y                       # the reference is not another run
        iters = Dict{String, Int}(); first_ratio = Dict{String, Float64}()
        for (name, m) in (("exact", ExactHessian()), ("gauss-newton", GaussNewtonModel()),
                          ("bhhh", BHHHModel(ridge = 1e-10)), ("bhhh2", BHHH2Model(ridge = 1e-10)))
            st = tr_solve(LikelihoodNLP(p; x0 = zeros(6)); rule = RDelta(), model = m,
                          subsolver = SteihaugCG(), trace = true,
                          params = TRParams(tol = 1e-6, max_iterations = 500))
            @test st.status === :first_order
            @test norm(st.solution - βhat) < 1e-5
            ρ = st.solver_specific[:ratio_trajectory]
            iters[name] = st.iter; first_ratio[name] = ρ[1]
            if name in ("exact", "gauss-newton")
                # The model is the function, so ρ = 1 wherever the decrease is resolved.
                @test abs(ρ[1] - 1) < 1e-10
                @test abs(ρ[2] - 1) < 1e-10
            end
        end
        @test iters["gauss-newton"] == iters["exact"]
        # Control. Under BHHH the first ratio reads the identity error at β = 0,
        # and the run is longer.
        @test abs(first_ratio["bhhh"] - 1) > 0.5
        @test iters["bhhh"] > iters["exact"]
        # The estimate is near the generating parameters and is not them.
        @test 0 < norm(βhat .- β_true(p)) < 0.5
        @test true_objective(p, βhat) <= true_objective(p, β_true(p))
    end

    @testset "one BHHH step: ρ reads the error along the step" begin
        p = LinearRegression(K = 6, M = 3_000, seed = 13)
        x0 = zeros(6); ridge = 1e-10
        st = tr_solve(LikelihoodNLP(p; x0 = x0); rule = RDelta(), model = BHHHModel(ridge = ridge),
                      subsolver = SteihaugCG(), trace = true,
                      params = TRParams(tol = 1e-6, max_iterations = 1))
        @test st.solver_specific[:accepted_trajectory][1]
        s = st.solution .- x0
        # B, H, g and the predicted decrease, from X and y.
        r = p.X * x0 .- p.y
        B = (p.X' * ((r .^ 2) .* p.X)) ./ p.M + ridge * I
        H = p.X' * p.X ./ p.M
        g = p.X' * r ./ p.M
        pred = -dot(g, s) - 0.5 * dot(s, B * s)
        @test abs(st.solver_specific[:ratio_trajectory][1] -
                  (1 + dot(s, (B - H) * s) / (2 * pred))) < 1e-10
        @test abs(st.solver_specific[:gamma_trajectory][1] - norm((H - B) * s) / norm(s)) < 1e-8
    end

    @testset "a sampled run, and what its stopping test is worth" begin
        p = LinearRegression(K = 5, M = 2_000, seed = 2)
        solve(rule, model, stop; maxit = 200) =
            tr_solve(FiniteSumNLP(p, rule; x0 = zeros(5), seed = 1);
                     rule = RDelta(), model = model, subsolver = SteihaugCG(), trace = true,
                     params = TRParams(tol = 1e-6, max_iterations = maxit, true_stop = stop))
        gn() = GaussNewtonModel(ridge = 1e-8)

        # Control. Three observations and five parameters: the exact model puts the
        # step on a point that fits the batch, where the batch gradient is zero.
        loose = solve(FixedSample(3), gn(), :none)
        @test loose.status === :first_order
        @test norm(true_gradient(p, loose.solution)) > 1e-2

        # Confirmed on the full batch, that stop is refused for as long as the run lasts.
        @test solve(FixedSample(3), gn(), :full).status !== :first_order

        # The false stop needs the subproblem solved inside the region, so that the
        # step lands where the batch gradient vanishes. Truncated CG gets there on a
        # few observations and not on 64. An exact subproblem solve gets there on 64.
        @test solve(FixedSample(64), gn(), :none).status !== :first_order
        exact_sub = tr_solve(FiniteSumNLP(p, FixedSample(64); x0 = zeros(5), seed = 1);
                             rule = RDelta(), model = gn(), subsolver = ExactMS(),
                             params = TRParams(tol = 1e-6, max_iterations = 200, true_stop = :none))
        @test exact_sub.status === :first_order
        @test norm(true_gradient(p, exact_sub.solution)) > 1e-2

        # Under a rule that lets the batch grow, a confirmed stop means what it says.
        tight = solve(NormTest(θ = 0.5), gn(), :full)
        @test tight.status === :first_order
        @test norm(true_gradient(p, tight.solution)) <= 1e-6

        # The batch objective is itself a quadratic with the model's Hessian, so on
        # a shared batch the ratio of an exact model is 1 and carries no signal.
        blind = solve(FixedSample(64), ExactHessian(), :full; maxit = 100)
        @test maximum(abs.(blind.solver_specific[:ratio_trajectory] .- 1)) < 1e-10
        # Control. Under BHHH the same batches give a ratio that moves.
        seen = solve(FixedSample(64), BHHHModel(ridge = 1e-8), :full; maxit = 100)
        @test maximum(abs.(seen.solver_specific[:ratio_trajectory] .- 1)) > 0.5
    end
end
