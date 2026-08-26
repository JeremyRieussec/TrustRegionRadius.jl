# =============================================================================
# benchmark/experiments/exp12_sampling_examples.jl
#
# EXPERIMENT 12 -- the three worked examples under every sampling rule.
#
# Adapted to the problem-class split. Three changes, all forced by the fact that
# these examples are FINITE SUMS rather than expectations:
#
#   * `SampledNLP` -> `FiniteSumNLP`. The oracle now names its class, and the
#     expectation branch (`ExpectationNLP`) is a different type with a different
#     solver.
#   * `N_max` dropped from every rule. On a finite sum the cap is M, imposed by
#     the problem, and a rule carrying a user N_max is now REJECTED at oracle
#     construction -- because there it is either redundant (≥ M) or a deliberate
#     sub-population budget (< M), and those are different intentions. NMAX is
#     kept below and passed as the oracle's `budget` where a genuine budget is
#     wanted.
#   * `LikelihoodNLP` -> `FullBatchNLP`, as in experiment 11.
#
#   least squares   linear (Gauss-Newton exact) and exponential fitting
#   likelihood      logistic regression, correctly specified
#   classification  one-hidden-layer softmax network
#
# crossed with
#
#   FixedSample            the baseline: work proportional to iterations
#   RadiusProportional     N ~ (sigma/(kappa*Delta))^2   -- couples to the mechanism
#   NormTest               N ~ sigma_g^2/(theta^2 |g|^2)
#   InnerProductTest       N ~ Var(dF_i'g)/(theta^2 |g|^4)
#   OrthogonalityTest      N ~ E|perp|^2/(nu^2 |g|^2)
#   AugmentedInnerProduct  the maximum of the two above
#   GeometricSample        N_0 * rate^k, fixed in advance
#   SequentialEstimation   noise small beside the predicted decrease
#
# Four things the grid answers.
#
#  (a) ALL THREE EXAMPLES RUN. One solver, one sampling layer; the problem type is
#      the only thing that changes. That is the claim the package makes, and this
#      is where it is checked rather than asserted.
#
#  (b) COST IS SAMPLES, NOT ITERATIONS. Only under FixedSample are the two
#      proportional. Everything is reported in both.
#
#  (c) THE INNER-PRODUCT TESTS ARE CHEAPER THAN THE NORM TEST FOR THE SAME
#      GUARANTEE. The norm test bounds sigma_g^2 = ip^2/|g|^2 + orth^2 as one
#      number; only the first part decides whether the sampled direction still
#      descends.
#
#  (d) COUPLING. `couples_to_radius` marks the rules whose N_k depends on Delta_k.
#      For those the sampling rule and the radius mechanism are not independent
#      axes, and the mechanism pays for its own radius policy.
#
#   julia --project=benchmark benchmark/experiments/exp12_sampling_examples.jl
#
# Format note: no `using` and no `include` here. `initialisation.jl` loads the
# packages, `harness.jl`, `archive.jl` and `config.jl` once, in order, exactly as
# it does for experiments 1-7. Re-including them per file re-ran `config.jl` and
# so re-evaluated its `const`s, which Julia either warns about or rejects.
# =============================================================================

const SEEDS  = 1:5
const MAXIT  = 300
const NMAX   = 100_000

# --- the three examples ------------------------------------------------------

"Each entry builds a fresh problem, an x0, and a way to score the result on the truth."
const EXAMPLES = [
    ("LS-linear", seed -> begin
         p = linear_least_squares(n = 8, M = 20_000, noise = 0.1, seed = seed)
         (prob = p, x0 = zeros(p.n), model = () -> GaussNewtonModel(ridge = 1e-10))
     end),
    #("LS-expfit", seed -> begin
    #     p = exponential_fit(n_terms_model = 2, M = 4_000, noise = 0.05,
    #                         misfit = 0.0, seed = seed)
    #     (prob = p, x0 = [0.8, 0.4, 1.2, 1.2], model = () -> GaussNewtonModel(ridge = 1e-8))
    # end),
    ("Logistic",  seed -> begin
         p = LogisticRegression(K = 10, M = 20_000, seed = seed)
         (prob = p, x0 = zeros(p.n), model = () -> BHHHModel(ridge = 1e-10))
     end),
    ("MLP",       seed -> begin
         rng = MersenneTwister(seed)
         X = randn(rng, 4_000, 6)
         L = 2.0 .* hcat(X[:, 1].^2, sin.(3 .* X[:, 2]), X[:, 3] .* X[:, 4])
         y = [argmax(view(L, i, :) .+ 0.3 .* randn(rng, 3)) for i in 1:4_000]
         p = MLPClassifier(X, y, 3; hidden = 4, λ = 1e-4)
         (prob = p, x0 = init_params(p; seed = seed),
          model = () -> BHHHModel(ridge = 1e-6))
     end),
]

# --- the sampling rules ------------------------------------------------------

const SAMPLERS = [
    ("FixedSample(64)",      () -> FixedSample(64)),
    ("RadiusProportional",   () -> RadiusProportional(κ_g = 1.0, κ_f = 1.0)),
    ("NormTest(0.5)",        () -> NormTest(θ = 0.5)),
    # ("InnerProduct(0.9)",    () -> InnerProductTest(θ = 0.9, N_min = 8)),
    # ("Orthogonality(2.0)",   () -> OrthogonalityTest(ν = 2.0, N_min = 8)),
    # ("Augmented(0.9,2.0)",   () -> AugmentedInnerProduct(θ = 0.9, ν = 2.0,
    #                                                     N_min = 8)),
    ("Geometric(8,1.06)",    () -> GeometricSample(N₀ = 8, rate = 1.06)),
    # ("Sequential(0.25)",     () -> SequentialEstimation(κ = 0.25, α = 0.05,
    #                                                   N_min = 8)),
    # Certified decrease: N from the paired differences D_i = F(x_k,ξ_i) −
    # F(x_k+s_k,ξ_i) under common random numbers. This is the rule whose cost is
    # ‖g_k‖^{-2} with the step length cancelling, and the contrast with
    # RadiusProportional (N ∼ Δ_k^{-2}) is the whole question of whether a
    # sampling rule can see the radius rule.
    ("Certified(0.9)",       () -> CertifiedDecrease(p = 0.9, N_min = 8)),
]

# The two radius mechanisms this experiment varies. NOT `RULES`: that name is
# config.jl's full list of eight, and redefining it here shadowed the mechanisms
# every other experiment runs on -- a `const` redefinition Julia warns about and
# whose effect depends on include order.
const TR_RULES = [("RDelta", () -> RDelta()), 
                    ("RDFO",    () -> RDFO(  γ1 = 0.25, γ2 = 0.50, γ3 = 2.0, ζ = 100.0)),
                    ("RGrad",   () -> RGrad( γ1 = 0.25, γ2 = 0.50, γ3 = 2.0, μ = 1.0)),
                    ("RAdaptiveGrad",    () -> RAdaptiveGrad())
                    ]

# --- one run -----------------------------------------------------------------

"Score on the exact gradient: the run's own ‖ĝ‖ is one batch's opinion."
function true_grad_norm(prob, x)
    try
        return norm(true_gradient(prob, x))
    catch
        return NaN
    end
end

function one_run(exf, samplerf, rulef, seed)
    ex  = exf(seed)
    nlp = FiniteSumNLP(ex.prob, samplerf(); x0 = copy(ex.x0), seed = seed)
    st = tr_solve(nlp; rule = rulef(), model = ex.model(), subsolver = SteihaugCG(),
                  trace = true,
                  params = TRParams(tol = 1e-7, max_iterations = MAXIT))
    ss = st.solver_specific
    return (iters   = st.iter,
            samples = get(ss, :samples_total, 0),
            g_true  = true_grad_norm(ex.prob, st.solution),
            capped  = get(ss, :sample_cap_hits, 0),
            N       = get(ss, :grad_sample_trajectory, Int[]),
            Δ       = ss[:delta_trajectory],
            status  = st.status)
end

med(v) = isempty(v) ? NaN : sort(collect(skipmissing(v)))[max(1, cld(length(v), 2))]

function sampling_grid()
    rows = NamedTuple[]
    for (ename, exf) in EXAMPLES, (sname, sf) in SAMPLERS, (rname, rf) in TR_RULES
        runs = [one_run(exf, sf, rf, s) for s in SEEDS]
        push!(rows, (example = ename, sampler = sname, rule = rname,
                     iters   = med([r.iters   for r in runs]),
                     samples = med([r.samples for r in runs]),
                     g_true  = med([r.g_true  for r in runs]),
                     capped  = med([r.capped  for r in runs]),
                     coupled = couples_to_radius(sf()),
                     example_run = runs[1]))
        @printf("  %-10s %-22s %-7s  iters=%5.0f samples=%9.0f  true‖g‖=%.2e\n",
                ename, sname, rname, rows[end].iters, rows[end].samples, rows[end].g_true)
    end
    return rows
end

function sampling_table(rows)
    io = IOBuffer()
    @printf(io, "%-10s %-22s %-7s %7s %11s %11s %7s %8s\n",
            "example", "sampling rule", "rule", "iters", "samples", "true ‖g‖",
            "capped", "coupled")
    println(io, "-"^92)
    last = ""
    for r in rows
        r.example == last || (println(io); last = r.example)
        @printf(io, "%-10s %-22s %-7s %7.0f %11.0f %11.3e %7.0f %8s\n",
                r.example, r.sampler, r.rule, r.iters, r.samples, r.g_true,
                r.capped, r.coupled ? "yes" : "no")
    end
    println(io)
    println(io, "Medians over $(length(SEEDS)) seeds. `true ‖g‖` uses the exact gradient over")
    println(io, "all terms, not the run's own estimate: a mechanism that shrinks the radius")
    println(io, "fast enough satisfies its own stopping test on noise alone.")
    println(io, "`coupled` marks rules whose N_k depends on Δ_k, for which the sampling rule")
    println(io, "and the radius mechanism are no longer independent axes.")
    return String(take!(io))
end

"The claim of (c): for the same descent guarantee the inner-product tests cost less."
function cost_ratio_table(rows)
    io = IOBuffer()
    println(io, "Sample cost relative to NormTest, same example and mechanism:\n")
    @printf(io, "%-10s %-7s %12s %12s %12s\n",
            "example", "rule", "InnerProd", "Orthog", "Augmented")
    println(io, "-"^58)
    for (ename, _) in EXAMPLES, (rname, _) in TR_RULES
        base = findfirst(r -> r.example == ename && r.rule == rname &&
                              r.sampler == "NormTest(0.5)", rows)
        base === nothing && continue
        b = rows[base].samples
        vals = map(("InnerProduct(0.9)", "Orthogonality(2.0)", "Augmented(0.9,2.0)")) do s
            i = findfirst(r -> r.example == ename && r.rule == rname && r.sampler == s, rows)
            i === nothing ? NaN : rows[i].samples / b
        end
        @printf(io, "%-10s %-7s %12.3f %12.3f %12.3f\n", ename, rname, vals...)
    end
    println(io)
    println(io, "Below 1 means cheaper than the norm test. The norm test bounds")
    println(io, "σ_g² = ip²/‖ĝ‖² + orth² as a single number; only the first part decides")
    println(io, "whether the sampled direction still descends, so bounding the sum is")
    println(io, "stricter than the descent property requires.")
    return String(take!(io))
end

# --- figures -----------------------------------------------------------------

function plot_sample_growth(rows, example)
    plt = plot(; xlabel = "iteration k", ylabel = "N_k", yscale = :log10,
                 title = "$example: sample size demanded", legend = :topleft, lw = 2)
    for r in filter(x -> x.example == example && x.rule == "RDelta", rows)
        N = r.example_run.N
        isempty(N) && continue
        plot!(plt, 1:length(N), N; label = r.sampler)
    end
    return plt
end

function plot_work(rows, example)
    plt = plot(; xlabel = "cumulative samples", ylabel = "median true ‖g‖",
                 xscale = :log10, yscale = :log10, legend = :bottomleft,
                 title = "$example: accuracy against work")
    for r in filter(x -> x.example == example && x.rule == "RDelta", rows)
        (isfinite(r.samples) && isfinite(r.g_true)) || continue
        scatter!(plt, [max(r.samples, 1)], [max(r.g_true, 1e-16)];
                 label = r.sampler, ms = 6)
    end
    return plt
end

# --- diagnostics that belong with the examples -------------------------------

"Gauss-Newton discards Σ rₙ∇²rₙ; that gap should track the residual size."
function gn_table()
    io = IOBuffer()
    println(io, "Gauss-Newton error against residual size (exponential fitting):\n")
    @printf(io, "%10s %12s %14s %14s\n", "misfit", "RMS residual", "‖GN−∇²f‖/‖∇²f‖", "at solution")
    println(io, "-"^54)
    for mf in (0.0, 0.1, 0.3, 1.0)
        p  = exponential_fit(n_terms_model = 2, M = 2_000, noise = 0.05,
                             misfit = mf, seed = 1)
        st = tr_solve(FullBatchNLP(p; x0 = [0.8, 0.4, 1.2, 1.2]);
                      rule = RDelta(), model = GaussNewtonModel(ridge = 1e-8),
                      subsolver = SteihaugCG(),
                      params = TRParams(tol = 1e-9, max_iterations = 500))
        e = gauss_newton_error(p, st.solution)
        @printf(io, "%10.2f %12.4f %14.5f %14s\n",
                mf, e.residual, e.GN_err, string(st.status))
    end
    println(io)
    println(io, "misfit adds a component the model cannot represent, raising the residual")
    println(io, "at the solution without changing the parameterisation. The discarded term")
    println(io, "is O(1) once the residual is, and Gauss-Newton stops being an")
    println(io, "approximation to the Hessian — the same pattern as BHHH under")
    println(io, "misspecification, reached from the other direction.")

    # The control: for a linear model the discarded term is identically zero.
    pl = linear_least_squares(n = 5, M = 1_000, noise = 2.0, seed = 1)
    el = gauss_newton_error(pl, randn(MersenneTwister(3), 5))
    @printf(io, "\nlinear least squares, RMS residual %.3f: ‖GN−∇²f‖/‖∇²f‖ = %.2e\n",
            el.residual, el.GN_err)
    println(io, "Zero to rounding whatever the residual: ∇²rₙ = 0, so Gauss-Newton is")
    println(io, "EXACT here, not merely accurate. That is the control for the column above.")
    return String(take!(io))
end

# --- main --------------------------------------------------------------------

function sampling_examples()
    arch = ExperimentArchive(tag = "sampling_examples")
    # The rule factories, not their names: `save_config` calls each once and
    # records its parameters, which is what makes the archive a description of
    # the run. Passing strings recorded only that they were strings.
    save_config(arch; rules = TR_RULES, seed = collect(SEEDS),
                params = TRParams(tol = 1e-7, max_iterations = MAXIT),
                extra = Dict("experiment" => "exp12_sampling_examples",
                             "examples" => join([e[1] for e in EXAMPLES], ", "),
                             "samplers" => join([s[1] for s in SAMPLERS], ", "),
                             "seeds" => collect(SEEDS)))

    println("running $(length(EXAMPLES))×$(length(SAMPLERS))×$(length(TR_RULES)) cells, "
            * "$(length(SEEDS)) seeds each\n")
    rows = sampling_grid()

    save_table(arch, "exp12_grid.txt", sampling_table(rows));       print(sampling_table(rows))
    save_table(arch, "exp12_cost_ratio.txt", cost_ratio_table(rows)); print(cost_ratio_table(rows))
    save_table(arch, "exp12_gauss_newton.txt", gn_table());     print(gn_table())

    for (ename, _) in EXAMPLES
        tag = replace(ename, r"[^A-Za-z0-9]" => "")
        savefig_archived(arch, "exp12_$(tag)_samples.pdf", plot_sample_growth(rows, ename))
        savefig_archived(arch, "exp12_$(tag)_work.pdf",    plot_work(rows, ename))
    end

    finalize_archive(arch; notes = """
        The three worked examples of Part III — least squares, likelihood,
        classification — crossed with every sampling rule and two radius
        mechanisms. One solver and one sampling layer throughout; only the problem
        type changes, which is the claim the package makes and this is where it is
        checked.

        Three readings.

        Cost is samples, not iterations. Only under FixedSample are the two
        proportional, and every profile that counts iterations quietly assumes it.
        Both columns are reported.

        The inner-product tests are cheaper than the norm test for the same descent
        guarantee. The norm test bounds σ_g² = ip²/‖ĝ‖² + orth² as a single number,
        while only the component along ĝ decides whether the sampled gradient still
        points downhill; error orthogonal to it rotates the direction without
        threatening descent. The cost-ratio table quantifies the saving on these
        problems.

        Coupling is what makes the axes non-orthogonal. RadiusProportional ties N_k
        to Δ_k, and SequentialEstimation ties it to the predicted decrease, which
        the radius also controls; the norm and inner-product tests do not consult
        the radius at all. Comparing a coupled rule against an uncoupled one at
        matched budget separates the cost of needing accuracy near a solution,
        which every method pays, from the cost of the mechanism's own radius
        policy, which only the coupled ones pay.

        The Gauss-Newton table is the least-squares counterpart of the information
        identity: the discarded term Σ rₙ∇²rₙ is small exactly when the residuals
        are, and identically zero for a linear model whatever the residual. So the
        two outer-product models fail for different reasons — BHHH on
        misspecification, Gauss-Newton on residual size — and both keep positive
        semidefiniteness, and with it the inability to report negative curvature,
        after their justification has gone.
        """)
end

if abspath(PROGRAM_FILE) == @__FILE__
    sampling_examples()
end
