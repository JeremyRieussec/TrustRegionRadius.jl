# Outer-product Hessians and the problems that justify them.
#
# The testset that matters most is the information identity: it holds at the true
# parameters of a correctly specified model and nowhere else, and everything BHHH
# claims rests on it. It is checked as a rate, not a threshold — the error must
# *decay like M^{-1/2}* at β* and must *not decay* away from it.

@testset "likelihood and outer-product Hessians" begin

    @testset "logistic regression: gradient and Hessian are correct" begin
        p = LogisticRegression(K = 4, M = 300, seed = 1)
        θ = 0.3 .* randn(MersenneTwister(9), 4)
        all_ = 1:p.M

        # analytic gradient against central differences
        g = zeros(4); batch_grad!(p, θ, all_, g)
        fd = similar(g); ε = 1e-6
        for j in 1:4
            tp = copy(θ); tp[j] += ε; tm = copy(θ); tm[j] -= ε
            fd[j] = (batch_obj(p, tp, all_) - batch_obj(p, tm, all_)) / (2ε)
        end
        @test g ≈ fd rtol = 1e-6

        # the score matrix averages to the gradient, by construction
        S = scores(p, θ, all_)
        @test size(S) == (4, p.M)
        @test vec(sum(S; dims = 2)) ./ p.M ≈ g rtol = 1e-12

        # Hessian against differences of the analytic gradient
        H = batch_hess(p, θ, all_)
        @test issymmetric(Symmetric(H))
        @test eigvals(Symmetric(H))[1] > 0          # strictly convex here
        gp = zeros(4); tp = copy(θ); tp[1] += ε
        batch_grad!(p, tp, all_, gp)
        @test (gp .- g) ./ ε ≈ H[:, 1] rtol = 1e-4
    end

    @testset "the information identity holds at β* and only there" begin
        K = 5
        Ms = [500, 2_000, 8_000, 32_000]
        at_true = Float64[]; at_wrong = Float64[]
        for M in Ms
            p = LogisticRegression(K = K, M = M, seed = 1)
            β = β_true(p)
            push!(at_true,  information_identity_error(p, β).B_err)
            push!(at_wrong, information_identity_error(p, β .+ 1.5).B_err)
        end
        # At β*: monotone decay, and the M^{-1/2} rate — err·√M stays bounded.
        @test issorted(at_true; rev = true)
        @test at_true[end] < at_true[1] / 4
        scaled = at_true .* sqrt.(Ms)
        @test maximum(scaled) / minimum(scaled) < 3        # flat to a small factor

        # Away from β*: no decay. This is the failure mode, not a weak bound.
        @test all(>(0.5), at_wrong)
        @test at_wrong[end] > at_wrong[1] / 2

        # W and B agree at a stationary point and differ by exactly ḡḡᵀ.
        p = LogisticRegression(K = K, M = 8_000, seed = 1)
        e = information_identity_error(p, β_true(p))
        @test e.BW_gap < 0.05
        @test isapprox(e.W_err, e.B_err; atol = 0.05)
    end

    @testset "B = W + ḡḡᵀ exactly" begin
        p = LogisticRegression(K = 4, M = 500, seed = 3)
        θ = β_true(p) .+ 0.8            # away from the optimum, where they differ
        nlp = LikelihoodNLP(p; x0 = θ)
        B = Matrix(dense_hessian(BHHHModel(), nlp, θ))
        W = Matrix(dense_hessian(BHHH2Model(), nlp, θ))
        g = zeros(4); batch_grad!(p, θ, 1:p.M, g)
        @test B ≈ W .+ g * g' rtol = 1e-10
        @test norm(B - W) > 1e-6                       # the gap is real here
    end

    @testset "outer-product models are positive semidefinite" begin
        # The guarantee, and the blind spot: B ⪰ 0 always, so no negative
        # curvature is ever reported, whatever the true Hessian does.
        p = LogisticRegression(K = 6, M = 400, seed = 5)
        nlp = LikelihoodNLP(p; x0 = zeros(6))
        for θ in ([zeros(6)], [5 .* randn(MersenneTwister(2), 6)])[1:2]
            for m in (BHHHModel(), BHHH2Model())
                B = Symmetric(Matrix(dense_hessian(m, nlp, θ[1])))
                @test minimum(eigvals(B)) >= -1e-10
            end
        end
        # Consequence: τ ≡ ‖g‖ over such a model, so SecondOrder certifies nothing.
        θ = zeros(6)
        λ = lambda_min_estimate(BHHHModel(), nlp, θ)
        @test λ >= -1e-10
        @test tau_criticality(1.23, λ) == 1.23
    end

    @testset "matrix-free B·v matches the dense product" begin
        p = LogisticRegression(K = 30, M = 200, seed = 7)
        nlp = LikelihoodNLP(p; x0 = zeros(30))
        θ = 0.4 .* randn(MersenneTwister(4), 30)
        v = randn(MersenneTwister(5), 30)
        for (m_dense, centred) in ((BHHHModel(dense_max = 10^6), false),
                                   (BHHH2Model(dense_max = 10^6), true))
            Bdense = Matrix(dense_hessian(m_dense, nlp, θ))
            S = score_matrix(nlp, θ)
            ḡ = vec(sum(S; dims = 2)) ./ size(S, 2)
            op = OuterProductOperator(S, ḡ, centred, 0.0)
            @test op * v ≈ Bdense * v rtol = 1e-10
            y = similar(v); mul!(y, op, v)
            @test y ≈ Bdense * v rtol = 1e-10
        end
    end

    @testset "ridge and rank" begin
        # N < n makes B singular by construction: a sum of N rank-one terms.
        p = LogisticRegression(K = 20, M = 1_000, seed = 11)
        nlp = LikelihoodNLP(p; x0 = zeros(20))
        S = scores(p, zeros(20), 1:5)                 # only 5 observations
        B = (S * S') ./ 5
        @test rank(B; atol = 1e-8) <= 5
        @test minimum(eigvals(Symmetric(B))) < 1e-8
        Br = B .+ 1e-6 .* I(20)
        @test minimum(eigvals(Symmetric(Br))) > 5e-7
    end

    @testset "solving with BHHH reaches the MLE" begin
        p = LogisticRegression(K = 6, M = 3_000, seed = 13)
        ref = nothing
        for m in (ExactHessian(), BHHHModel(ridge = 1e-10), BHHH2Model(ridge = 1e-10))
            nlp = LikelihoodNLP(p; x0 = zeros(6))
            st = tr_solve(nlp; rule = RDelta(), model = m, subsolver = SteihaugCG(),
                          params = TRParams(tol = 1e-8, max_iterations = 500))
            @test st.status === :first_order
            ref === nothing ? (ref = st.solution) : @test(st.solution ≈ ref, rtol = 1e-4)
        end
        # The MLE is near, but not equal to, the generating parameters.
        @test norm(ref .- β_true(p)) < 0.5
    end

    @testset "MLP: per-sample scores are the gradients of the terms" begin
        rng = MersenneTwister(1)
        X = randn(rng, 40, 6); y = rand(rng, 1:3, 40)
        p = MLPClassifier(X, y, 3; hidden = 4)
        @test p.n == 4 * 6 + 4 + 3 * 4 + 3
        θ = 0.5 .* randn(rng, p.n)

        S = scores(p, θ, 1:p.M)
        @test size(S) == (p.n, p.M)
        for i in (1, 17, 40)                     # column i is ∇ of term i
            fd = zeros(p.n); ε = 1e-6
            for j in 1:p.n
                tp = copy(θ); tp[j] += ε; tm = copy(θ); tm[j] -= ε
                fd[j] = (loss_terms(p, tp, [i])[1] - loss_terms(p, tm, [i])[1]) / (2ε)
            end
            @test S[:, i] ≈ fd rtol = 1e-5 atol = 1e-8
        end
        g = zeros(p.n); batch_grad!(p, θ, 1:p.M, g)
        @test g ≈ vec(sum(S; dims = 2)) ./ p.M rtol = 1e-12
        @test 0 <= accuracy(p, θ) <= 1
    end

    @testset "MLP: the identity fails and does not improve with N" begin
        rng = MersenneTwister(2)
        X = randn(rng, 600, 5); y = rand(rng, 1:3, 600)    # labels unrelated to X
        p = MLPClassifier(X, y, 3; hidden = 3)
        θ = init_params(p; seed = 1)
        e_small = information_identity_error(p, θ; batch = 1:100)
        e_large = information_identity_error(p, θ; batch = 1:600)
        # Misspecified: no decay. Contrast with the logistic testset above.
        @test e_large.B_err > 0.1
        @test e_large.B_err > e_small.B_err / 3
    end

    @testset "MLP: the exact Hessian refuses at scale" begin
        rng = MersenneTwister(3)
        X = randn(rng, 20, 200); y = rand(rng, 1:10, 20)
        p = MLPClassifier(X, y, 10; hidden = 30)           # n = 6340
        @test p.n > 2_000
        @test_throws ArgumentError batch_hess(p, init_params(p), 1:p.M)
        # BHHH does not: it needs the n × N score matrix, not an n × n array.
        @test size(scores(p, init_params(p), 1:p.M)) == (p.n, p.M)
    end

    @testset "score_matrix refuses a model that has no scores" begin
        nlp = ADNLPModel(x -> sum(abs2, x), ones(3))
        @test_throws ArgumentError score_matrix(nlp, ones(3))
    end
end
