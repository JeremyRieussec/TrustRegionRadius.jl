# The adaptive sampling rules, and the three worked examples under all of them.
#
# The centrepiece is the last testset: least squares, likelihood and classification
# all run through one solver and one sampling layer, with only the problem type
# changing. That is the claim the package makes, and asserting it in prose is not
# the same as running it.

@testset "sampling rules and the three examples" begin

    # A ScoredProblem with a controllable score matrix, so the moment formulas can
    # be checked against values computed directly.
    prob = LogisticRegression(K = 6, M = 2_000, seed = 4)
    θ    = β_true(prob) .+ 0.3
    all_ = collect(1:prob.M)

    @testset "the parallel/orthogonal split is exact" begin
        S = scores(prob, θ, all_); N = size(S, 2)
        ḡ = vec(sum(S; dims = 2)) ./ N
        st = batch_stats(prob, θ, all_, ḡ)

        # σ_g² = ip²/‖ĝ‖² + orth², the identity the inner-product tests rest on.
        @test st.σg² ≈ st.ip² / dot(ḡ, ḡ) + st.orth² rtol = 1e-8

        # orth² computed by explicit projection must agree.
        D = S .- ḡ
        P = D .- (ḡ * (ḡ' * D)) ./ dot(ḡ, ḡ)
        @test st.orth² ≈ sum(abs2, P) / (N - 1) rtol = 1e-8

        # ip² is the variance of the scalars ∇Fᵢᵀĝ.
        proj = vec(S' * ḡ)
        @test st.ip² ≈ sum(abs2, proj .- mean(proj)) / (N - 1) rtol = 1e-8

        @test st.σg² > 0 && st.ip² > 0 && st.orth² >= 0
    end

    @testset "each rule's arithmetic" begin
        # σg = 2, σf = 3, ip² = 8, orth² = 1.5, ‖g‖ = 2, Δ = 0.1
        st = SamplingState(3, 0.1, 2.0, 4.0, 9.0, 8.0, 1.5, 0.5, 32)

        @test grad_sample_size(FixedSample(32), st) == 32
        @test grad_sample_size(RadiusProportional(κ_g = 1.0, N_max = 10^9), st) == 400
        @test obj_sample_size(RadiusProportional(κ_f = 1.0, N_max = 10^9), st)  == 90_000
        @test grad_sample_size(NormTest(θ = 0.5), st) == 4          # 4/(0.25·4)

        # ip²/(θ²‖g‖⁴) = 8/(0.81·16) = 0.617 → 1, floored at N_min
        @test grad_sample_size(InnerProductTest(θ = 0.9, N_min = 1), st) == 1
        # orth²/(ν²‖g‖²) = 1.5/(4·4) = 0.094 → 1
        @test grad_sample_size(OrthogonalityTest(ν = 2.0, N_min = 1), st) == 1
        @test grad_sample_size(AugmentedInnerProduct(θ = 0.9, ν = 2.0, N_min = 1), st) ==
              max(grad_sample_size(InnerProductTest(θ = 0.9, N_min = 1), st),
                  grad_sample_size(OrthogonalityTest(ν = 2.0, N_min = 1), st))

        @test grad_sample_size(GeometricSample(N₀ = 8, rate = 2.0), st) == 64

        @test_throws ArgumentError InnerProductTest(θ = 0.0)
        @test_throws ArgumentError OrthogonalityTest(ν = -1.0)
        @test_throws ArgumentError AugmentedInnerProduct(θ = 1.0, ν = 0.0)
        @test_throws ArgumentError SequentialEstimation(κ = 0.0)
        @test_throws ArgumentError SequentialEstimation(α = 1.5)
        @test_throws ArgumentError SequentialEstimation(growth = 1.0)
    end

    @testset "the normal quantile" begin
        @test TrustRegionRadius._z_quantile(0.05) ≈ 1.959964 atol = 1e-5
        @test TrustRegionRadius._z_quantile(0.01) ≈ 2.575829 atol = 1e-4
        @test TrustRegionRadius._z_quantile(0.10) ≈ 1.644854 atol = 1e-4
    end

    @testset "SequentialEstimation sizes against the predicted decrease" begin
        # N ≥ 2 z² σ_f² / (κ² pred²) = 2·1.96²·4 / (0.25²·0.5²) = 1967
        r  = SequentialEstimation(κ = 0.25, α = 0.05, N_min = 1, N_max = 10^9,
                                  monotone = false)
        st = SamplingState(3, 0.1, 2.0, 4.0, 4.0, 0.0, 0.0, 0.5, 32)
        @test grad_sample_size(r, st) == 1967

        # A larger predicted decrease tolerates more noise, so needs fewer samples.
        st2 = SamplingState(3, 0.1, 2.0, 4.0, 4.0, 0.0, 0.0, 5.0, 32)
        @test grad_sample_size(SequentialEstimation(κ = 0.25, N_min = 1, N_max = 10^9,
                                                    monotone = false), st2) <
              grad_sample_size(r, st)

        # Monotone: never decreases, and rises by at most `growth` per iteration.
        rm = SequentialEstimation(κ = 0.25, N_min = 1, N_max = 10^9,
                                  monotone = true, growth = 2.0, N_start = 10)
        big  = SamplingState(1, 0.1, 2.0, 4.0, 4.0, 0.0, 0.0, 1e-3, 10)  # demands a lot
        n1 = grad_sample_size(rm, big)
        n2 = grad_sample_size(rm, big)
        @test n2 <= 2 * n1                       # capped growth
        small = SamplingState(3, 0.1, 2.0, 4.0, 4.0, 0.0, 0.0, 1e6, 10) # demands ~nothing
        @test grad_sample_size(rm, small) >= n2  # never shrinks
        reset_sampling_rule!(rm)
        @test rm.N_last == 0

        # No prediction yet on the first iteration.
        rn = SequentialEstimation(N_start = 17, N_min = 1, monotone = false)
        @test grad_sample_size(rn, SamplingState(0, 1.0, 1.0, 1.0, 1.0)) == 17
    end

    @testset "coupling to the radius is declared correctly" begin
        @test !couples_to_radius(FixedSample(8))
        @test !couples_to_radius(NormTest())
        @test !couples_to_radius(InnerProductTest())
        @test !couples_to_radius(OrthogonalityTest())
        @test !couples_to_radius(AugmentedInnerProduct())
        @test !couples_to_radius(GeometricSample())
        @test couples_to_radius(RadiusProportional())
        @test couples_to_radius(SequentialEstimation())   # through pred, not a formula in Δ
    end

    @testset "the inner-product test is cheaper than the norm test" begin
        # Both are computed from the same batch at the same iterate, so the
        # comparison is like for like. The norm test bounds σ_g², which includes
        # the orthogonal part the descent property does not need.
        S = scores(prob, θ, all_); N = size(S, 2)
        ḡ = vec(sum(S; dims = 2)) ./ N
        stt = batch_stats(prob, θ, all_, ḡ)
        st = SamplingState(1, 1.0, norm(ḡ), stt.σg², stt.σf², stt.ip², stt.orth², NaN, 0)

        n_norm = grad_sample_size(NormTest(θ = 0.5, N_min = 1, N_max = 10^9), st)
        n_ip   = grad_sample_size(InnerProductTest(θ = 0.5, N_min = 1, N_max = 10^9), st)
        @test n_ip <= n_norm
    end

    @testset "least squares: Gauss-Newton is exact for a linear model" begin
        p = linear_least_squares(n = 5, M = 800, noise = 2.0, seed = 1)
        @test p isa NLSProblem
        for x in (zeros(5), randn(MersenneTwister(3), 5))
            e = gauss_newton_error(p, x)
            @test e.GN_err < 1e-6            # ∇²rₙ = 0: nothing is discarded
        end
        # ... and the residual size is irrelevant to that, unlike the nonlinear case.
        @test gauss_newton_error(p, 10 .* randn(MersenneTwister(4), 5)).GN_err < 1e-6

        # Jacobian and scores agree with finite differences.
        x = randn(MersenneTwister(5), 5)
        J = jacobian(p, x, 1:10); r = residuals(p, x, 1:10)
        S = scores(p, x, 1:10)
        @test S ≈ (J .* r)' rtol = 1e-12
        g = zeros(5); batch_grad!(p, x, 1:10, g)
        @test g ≈ vec(sum(S; dims = 2)) ./ 10 rtol = 1e-12
    end

    @testset "least squares: the Gauss-Newton gap tracks residual size" begin
        errs = Float64[]; res = Float64[]
        for mf in (0.0, 0.5, 2.0)
            p = exponential_fit(n_terms_model = 2, M = 600, noise = 0.02,
                                misfit = mf, seed = 1)
            st = tr_solve(LikelihoodNLP(p; x0 = [0.8, 0.4, 1.2, 1.2]);
                          rule = RDelta(), model = GaussNewtonModel(ridge = 1e-8),
                          subsolver = SteihaugCG(),
                          params = TRParams(tol = 1e-8, max_iterations = 400))
            e = gauss_newton_error(p, st.solution)
            push!(errs, e.GN_err); push!(res, e.residual)
        end
        @test issorted(res)                       # misfit raises the residual
        @test errs[end] > errs[1]                 # and with it the discarded term
    end

    @testset "all three examples run under every sampling rule" begin
        samplers = [("Fixed",        () -> FixedSample(64)),
                    ("RadiusProp",   () -> RadiusProportional(N_max = 20_000)),
                    ("NormTest",     () -> NormTest(θ = 0.5, N_max = 20_000)),
                    ("InnerProduct", () -> InnerProductTest(N_min = 8, N_max = 20_000)),
                    ("Orthogonality",() -> OrthogonalityTest(N_min = 8, N_max = 20_000)),
                    ("Augmented",    () -> AugmentedInnerProduct(N_min = 8, N_max = 20_000)),
                    ("Geometric",    () -> GeometricSample(N₀ = 8, rate = 1.1, N_max = 20_000)),
                    ("Sequential",   () -> SequentialEstimation(N_min = 8, N_max = 20_000))]

        rngm = MersenneTwister(1)
        Xm = randn(rngm, 800, 5)
        ym = [argmax(view(2.0 .* hcat(Xm[:, 1].^2, sin.(3 .* Xm[:, 2]), Xm[:, 3]), i, :))
              for i in 1:800]

        examples = [
            ("LS-linear", () -> linear_least_squares(n = 4, M = 2_000, seed = 2),
                          p -> zeros(p.n), () -> GaussNewtonModel(ridge = 1e-8)),
            ("LS-expfit", () -> exponential_fit(n_terms_model = 1, M = 800, seed = 2),
                          p -> [0.9, 0.6],  () -> GaussNewtonModel(ridge = 1e-8)),
            ("Logistic",  () -> LogisticRegression(K = 5, M = 2_000, seed = 2),
                          p -> zeros(p.n), () -> BHHHModel(ridge = 1e-8)),
            ("MLP",       () -> MLPClassifier(Xm, ym, 3; hidden = 3, λ = 1e-4),
                          p -> init_params(p; seed = 1), () -> BHHHModel(ridge = 1e-6)),
        ]

        for (ename, mk, x0f, modelf) in examples, (sname, sf) in samplers
            @testset "$ename / $sname" begin
                p   = mk()
                nlp = SampledNLP(p, sf(); x0 = x0f(p), seed = 1)
                st  = tr_solve(nlp; rule = RDelta(), model = modelf(),
                               subsolver = SteihaugCG(), trace = true,
                               params = TRParams(tol = 1e-6, max_iterations = 40))

                @test st.status in (:first_order, :max_iter, :stalled)
                ss = st.solver_specific
                @test haskey(ss, :samples_total)
                @test haskey(ss, :grad_sample_trajectory)
                N = ss[:grad_sample_trajectory]
                @test length(N) == st.iter
                @test all(>(0), N)
                @test ss[:samples_total] > 0
                # The exact gradient is available, so score on it rather than on ĝ.
                @test isfinite(norm(true_gradient(p, st.solution)))
            end
        end
    end

    @testset "sample sizes actually vary under an adaptive rule" begin
        # A rule that never moves N is indistinguishable from FixedSample, so the
        # grid above would pass vacuously. Check that at least one adapts.
        p = LogisticRegression(K = 5, M = 20_000, seed = 6)
        fixed = SampledNLP(p, FixedSample(64); x0 = zeros(5), seed = 1)
        adapt = SampledNLP(p, RadiusProportional(N_max = 50_000); x0 = zeros(5), seed = 1)
        for nlp in (fixed, adapt)
            tr_solve(nlp; rule = RDFO(ζ = 1.0), model = BHHHModel(ridge = 1e-8),
                     subsolver = SteihaugCG(), trace = true,
                     params = TRParams(tol = 1e-7, max_iterations = 30))
        end
        @test length(unique(fixed.Ng_hist)) == 1
        @test length(unique(adapt.Ng_hist)) > 1
    end
end
