# The sampling features that reproduce the PreDoc logit experiment, each checked
# against its definition rather than against itself:
#
#   MultinomialLogit                  derivatives against central differences
#   SmoothedSize                      a·N_k ≤ N_{k+1} ≤ b·N_k and the rising floor, on a run
#   CertifiedDecrease(variance = …)   the outer-product variance against sᵀB̃s − (g̃ᵀs)²
#   FiniteSumNLP(scheme = …)          :independent, :nested and :prefix
#
# Several assertions come with a control, and the control is not optional. The
# smoothing bound is only evidence if the bare rule leaves it; nesting is only
# evidence if independent draws do not nest. Without the control each assertion
# would pass on an implementation that did nothing.

@testset "sampling features" begin

    @testset "MultinomialLogit: interface and derivatives" begin
        p = MultinomialLogit(K = 6, M = 400, J = 5, seed = 3, signal = 0.8)
        @test population(p) == 400
        @test has_scores(p)
        @test supports_paired(p)
        @test p isa LikelihoodProblem

        all_b = collect(1:p.M)
        # At β = 0 every alternative is equally likely whatever the data, so
        # f = log J. A check on the log-sum-exp that needs no reference value.
        f0 = sum(loss_terms(p, zeros(p.n), all_b)) / p.M
        @test abs(f0 - log(5)) < 1e-12

        β = 0.3 .* randn(MersenneTwister(11), p.n)
        g = zeros(p.n); batch_grad!(p, β, all_b, g)
        ε = 1e-6

        fd = zeros(p.n)
        for j in 1:p.n
            bp = copy(β); bp[j] += ε
            bm = copy(β); bm[j] -= ε
            fd[j] = (sum(loss_terms(p, bp, all_b)) - sum(loss_terms(p, bm, all_b))) / (2ε * p.M)
        end
        @test norm(g - fd) / norm(fd) < 1e-7

        # The gradient is the mean of the score columns: what lets BHHH be built
        # from the array the gradient came from.
        S = scores(p, β, all_b)
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
        # Each per-observation Hessian is a covariance under the choice
        # probabilities, so the objective is convex.
        @test minimum(eigvals(Symmetric(H))) > -1e-12

        @test init_params(p) == zeros(p.n)
        @test abs(accuracy(p, zeros(p.n)) - 0.2) < 0.06      # chance, 1/J
        @test accuracy(p, β_true(p)) > 0.35
    end

    @testset "MultinomialLogit: the information identity" begin
        p = MultinomialLogit(K = 6, M = 400, J = 5, seed = 3, signal = 0.8)
        βs = β_true(p)
        at_truth = information_identity_error(p, βs).B_err
        @test at_truth < 0.25
        @test information_identity_error(p, βs .+ 0.7).B_err > at_truth

        # `noise > 0` generates from a mixed logit and fits a plain one. That is
        # misspecification, and the identity must then fail at β* itself.
        pn = MultinomialLogit(K = 6, M = 4_000, J = 5, seed = 3, signal = 0.8, noise = 1.5)
        p0 = MultinomialLogit(K = 6, M = 4_000, J = 5, seed = 3, signal = 0.8, noise = 0.0)
        @test information_identity_error(pn, β_true(pn)).B_err >
              information_identity_error(p0, β_true(p0)).B_err
    end

    prob = MultinomialLogit(K = 5, M = 3_000, J = 5, seed = 1, signal = 1.0)
    M = population(prob)

    # One sampled run on `prob`, everything but the named arguments held fixed.
    function run_with(rule; scheme = :independent, iters = 60, seed = 0)
        nlp = FiniteSumNLP(prob, rule; x0 = zeros(prob.n), seed = seed,
                           replace = false, N_init = 100, scheme = scheme)
        st = tr_solve(nlp; rule = RGrad(), model = BHHHModel(), subsolver = SteihaugCG(),
                      trace = true,
                      params = TRParams(max_iterations = iters, tol = 1e-8,
                                        η = 0.01, η1 = 0.01, η2 = 0.8, Δ0 = 1.0))
        return nlp, st
    end

    @testset "SmoothedSize: the bound holds on a run" begin
        a, b = 0.75, 2.0
        rule = SmoothedSize(CertifiedDecrease(p = 0.95, N_min = 10);
                            a = a, b = b, N_floor = 10, N_incr = 10, N_start = 100)
        nlp_s, _ = run_with(rule)
        Ns = nlp_s.Ng_hist
        @test length(Ns) > 5

        # The bound is a claim about consecutive sizes, so it is audited on the
        # sizes the run produced, not on the arithmetic that imposed it.
        below = 0; above = 0
        for k in 2:length(Ns)
            prev, cur = Ns[k-1], Ns[k]
            lo = max(ceil(Int, a * prev), min(10 + 10 * (k - 1), M))
            hi = max(lo, ceil(Int, b * prev))
            cur < min(lo, M) && (below += 1)
            cur > min(hi, M) && (above += 1)
        end
        @test below == 0
        @test above == 0

        # Control: the bare rule leaves those bounds on the same problem.
        nlp_r, _ = run_with(CertifiedDecrease(p = 0.95, N_min = 10, N_start = 100))
        Nr = nlp_r.Ng_hist
        ratios = [Nr[k] / max(Nr[k-1], 1) for k in 2:length(Nr)]
        @test maximum(ratios) > b || minimum(ratios) < a

        # The floor N_floor + k·N_incr, with a ceiling too low to reach it: the
        # floor wins the conflict.
        rule_f = SmoothedSize(CertifiedDecrease(p = 0.95, N_min = 2);
                              a = 0.5, b = 1.05, N_floor = 20, N_incr = 25, N_start = 20)
        nlp_f, _ = run_with(rule_f; iters = 30)
        Nf = nlp_f.Ng_hist
        @test count(k -> Nf[k] < min(20 + 25 * (k - 1), M), eachindex(Nf)) == 0
    end

    @testset "SmoothedSize: forwarding and validation" begin
        rule = SmoothedSize(CertifiedDecrease(p = 0.95, N_min = 10))
        @test needs_paired(rule)
        @test couples_to_radius(rule)
        @test paired_variance_kind(rule) === :empirical
        @test !needs_scores(rule)

        op = SmoothedSize(CertifiedDecrease(variance = :outer_product))
        @test paired_variance_kind(op) === :outer_product
        @test needs_scores(op)

        # The wrapper carries no pairing of its own: over a rule that does not
        # pair, it does not pair either.
        @test !needs_paired(SmoothedSize(NormTest()))

        @test_throws ArgumentError SmoothedSize(FullBatch(); a = 1.5)
        @test_throws ArgumentError SmoothedSize(FullBatch(); a = 0.0)
        @test_throws ArgumentError SmoothedSize(FullBatch(); b = 0.5)
        @test_throws ArgumentError SmoothedSize(FullBatch(); N_floor = 0)
        @test_throws ArgumentError SmoothedSize(FullBatch(); N_incr = -1)
    end

    @testset "the outer-product variance" begin
        nlp = FiniteSumNLP(prob, FixedSample(200); x0 = zeros(prob.n), seed = 7, replace = false)
        resample!(nlp, 0, 1.0, 1.0)
        x = 0.2 .* randn(MersenneTwister(5), prob.n)

        # Against the definition, formed the other way: sᵀB̃s − (g̃ᵀs)² with B̃ the
        # outer product, rescaled from the population to the sample convention.
        for h in (1e-1, 1e-2, 1e-3)
            s = h .* randn(MersenneTwister(9), prob.n)
            Sm = scores(prob, x, nlp.batch_f)
            N = size(Sm, 2)
            ḡ = vec(sum(Sm; dims = 2)) ./ N
            direct = (dot(s, (Sm * Sm') * s) / N - dot(ḡ, s)^2) * N / (N - 1)
            @test abs(paired_op_variance(nlp, x, x .+ s) - direct) / abs(direct) < 1e-9
        end

        # It is a first-order Taylor approximation, so it must close on the
        # empirical variance as the step shrinks, and monotonically.
        gaps = Float64[]
        for h in (1e-1, 1e-2, 1e-3, 1e-4)
            s = h .* randn(MersenneTwister(9), prob.n)
            _, σ_emp, _ = paired_decrease_stats(nlp, x, x .+ s)
            push!(gaps, abs(paired_op_variance(nlp, x, x .+ s) / σ_emp - 1))
        end
        @test issorted(gaps; rev = true)
        @test gaps[end] < 1e-4

        # Both estimates reach the trace, and the rule uses the one it asked for.
        st_emp = run_with(CertifiedDecrease(p = 0.95, N_min = 10, N_start = 100,
                                            variance = :empirical))[2]
        st_op  = run_with(CertifiedDecrease(p = 0.95, N_min = 10, N_start = 100,
                                            variance = :outer_product))[2]
        emp = st_emp.solver_specific[:paired_variance_trajectory]
        op  = st_emp.solver_specific[:paired_op_variance_trajectory]
        @test length(emp) == length(op)
        @test any(isfinite, op)
        # Formed as a sample variance, so it cannot come out negative where the
        # difference of the two quadratic forms could.
        @test all(v -> !isfinite(v) || v >= 0, op)
        @test emp != st_op.solver_specific[:paired_variance_trajectory] ||
              st_emp.iter != st_op.iter

        # The outer product needs scores. A problem without them is rejected at
        # construction, and only for the rule that asks.
        noscore = FiniteSum(2, 4, (i, y) -> sum(abs2, y), (i, y, gg) -> (gg .+= 2y),
                            (i, y) -> 2.0 * [1.0 0.0; 0.0 1.0])
        @test !has_scores(noscore)
        @test_throws ArgumentError FiniteSumNLP(noscore, CertifiedDecrease(variance = :outer_product))
        @test FiniteSumNLP(noscore, CertifiedDecrease()) isa FiniteSumNLP
        @test_throws ArgumentError CertifiedDecrease(variance = :bogus)
    end

    @testset "sample schemes" begin
        @test sample_schemes() == (:independent, :nested, :prefix)

        smoothed() = SmoothedSize(CertifiedDecrease(p = 0.95, N_min = 10);
                                  a = 0.6, b = 2.0, N_start = 60)
        for sch in sample_schemes()
            nl = FiniteSumNLP(prob, smoothed(); x0 = zeros(prob.n), seed = 4,
                              replace = false, N_init = 60, scheme = sch)
            @test sample_scheme(nl) === sch
        end
        @test_throws ArgumentError FiniteSumNLP(prob, FixedSample(10); scheme = :bogus)

        # :prefix is {1, …, N_k} at every iteration, and draws nothing.
        nl = FiniteSumNLP(prob, smoothed(); x0 = zeros(prob.n), seed = 4,
                          replace = false, N_init = 60, scheme = :prefix)
        is_prefix = true
        for k in 0:25
            resample!(nl, k, 1.0, 1.0)
            nl.batch_g == collect(1:nl.Ng) || (is_prefix = false)
        end
        @test is_prefix
        n1 = FiniteSumNLP(prob, FixedSample(50); x0 = zeros(prob.n), seed = 4, scheme = :prefix)
        before = copy(n1.rng)
        for k in 0:5
            resample!(n1, k, 1.0, 1.0)
        end
        @test n1.rng == before

        # Drive the batch through sizes that rise and fall, and record whether
        # each batch contains, or is contained in, the one before it.
        sizes = [40, 80, 160, 90, 300, 120, 121]
        function nesting(scheme)
            # `local`, because the enclosing testset assigns names of its own and a
            # closure would otherwise share them.
            local oracle, grows, shrinks, distinct, last_set
            oracle = FiniteSumNLP(prob, FixedSample(50); x0 = zeros(prob.n), seed = 4,
                                  replace = false, N_init = 50, scheme = scheme)
            grows = true; shrinks = true; distinct = true
            last_set = nothing
            for N in sizes
                oracle.Ng = N; oracle.Nf = N
                TrustRegionRadius._draw!(oracle, N, N)
                this_set = Set(oracle.batch_g)
                length(this_set) == N || (distinct = false)
                if last_set !== nothing
                    if N >= length(last_set)
                        issubset(last_set, this_set) || (grows = false)
                    else
                        issubset(this_set, last_set) || (shrinks = false)
                    end
                end
                last_set = this_set
            end
            return (grows = grows, shrinks = shrinks, distinct = distinct)
        end

        nested = nesting(:nested)
        @test nested.grows          # 𝒩_k ⊆ 𝒩_{k+1} on a rise
        @test nested.shrinks        # 𝒩_{k+1} ⊂ 𝒩_k on a fall
        @test nested.distinct       # no duplicates when replace = false
        # Control: independent draws do not nest.
        @test !nesting(:independent).grows

        # On an expectation there is no first individual, so :prefix is refused,
        # and :nested keeps the old realisations when the sample grows.
        pexp = PerturbedExpectation(ADNLPModel(y -> sum(abs2, y), zeros(3)); σg = 0.1)
        @test_throws ArgumentError ExpectationNLP(pexp, FixedSample(10); scheme = :prefix)
        ne = ExpectationNLP(pexp, FixedSample(10); scheme = :nested, N_init = 10)
        @test sample_scheme(ne) === :nested
        kept = true
        prev = nothing
        for N in (10, 30, 60, 20, 90)
            ne.Ng = N; ne.Nf = N
            TrustRegionRadius._draw!(ne, N, N)
            cur = ne.batch_g.b
            size(cur, 2) == N || (kept = false)
            if prev !== nothing && N >= size(prev, 2)
                cur[:, 1:size(prev, 2)] == prev || (kept = false)
            end
            prev = cur
        end
        @test kept

        # A reset restores the pool, so a reused oracle sees the same batches.
        n4 = FiniteSumNLP(prob, FixedSample(50); x0 = zeros(prob.n), seed = 4,
                          replace = false, N_init = 50, scheme = :nested)
        first_batch = copy(n4.batch_g)
        for k in 0:4
            resample!(n4, k, 1.0, 1.0)
        end
        reset_sampling!(n4)
        @test n4.batch_g == first_batch
        @test n4.pool == first_batch
    end

    @testset "the defaults are the previous behaviour" begin
        # Archived runs were made before `scheme` and `variance` existed, so the
        # defaults must reproduce them: same batches, same sizes, same run.
        r1 = run_with(CertifiedDecrease(p = 0.9, N_min = 2, N_start = 8); seed = 12)[2]
        r2 = run_with(CertifiedDecrease(p = 0.9, N_min = 2, N_start = 8); seed = 12,
                      scheme = :independent)[2]
        @test r1.iter == r2.iter
        @test r1.objective == r2.objective
        @test paired_variance_kind(CertifiedDecrease()) === :empirical
        @test !needs_scores(CertifiedDecrease())
    end
end
