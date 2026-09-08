# C6. true_gradient is evaluated on every iteration even when trace = false,
# because Julia evaluates arguments eagerly before sample_pre!/sample_post! can
# discard them (finitesum.jl:79,106,112).
using TrustRegionRadius, ADNLPModels, NLPModels, LinearAlgebra, Printf
sep(t) = (println(); println("="^72); println(t); println("="^72))

sep("C6. Cost of the eager true_gradient on an untraced finite-sum run")
base = ADNLPModel(z -> 0.5 * sum(collect(1.0:4) .* (z .- 1.0) .^ 2), zeros(4))

for M in (20_000, 200_000)
    prob = PerturbedSum(base, M; σg = 1.0, seed = 1)
    # tol tiny so the run always takes the full 40 iterations: a fixed workload
    p = TRParams(η = 0.1, η1 = 0.1, η2 = 0.9, tol = 1e-300, max_iterations = 40)

    mk() = FiniteSumNLP(prob, FixedSample(32); x0 = zeros(4), seed = 1)

    # warm up
    tr_solve(mk(); rule = RDelta(), model = ExactHessian(), subsolver = ExactMS(),
             params = p, trace = false)

    m1 = mk()
    t_untraced = @elapsed st1 = tr_solve(m1; rule = RDelta(), model = ExactHessian(),
                                         subsolver = ExactMS(), params = p, trace = false)
    m2 = mk()
    t_traced = @elapsed st2 = tr_solve(m2; rule = RDelta(), model = ExactHessian(),
                                       subsolver = ExactMS(), params = p, trace = true)

    # the cost of the calls themselves: one true_gradient per iteration, M terms each
    x = zeros(4)
    true_gradient(prob, x)                      # warm up before timing
    tg = minimum(@elapsed(for _ in 1:st1.iter; true_gradient(prob, x); end) for _ in 1:5)

    @printf("M = %6d  iter = %2d\n", M, st1.iter)
    @printf("  trace = false : %.4f s\n", t_untraced)
    @printf("  trace = true  : %.4f s\n", t_traced)
    @printf("  %d bare true_gradient calls: %.4f s  (%.0f%% of the untraced run)\n",
            st1.iter, tg, 100 * tg / t_untraced)
    @printf("  untraced run carries the full-pass cost: %s\n",
            tg > 0.15 * t_untraced ? "yes, materially" : "not materially")
    println()
end
println("The deterministic path never calls true_gradient, so no experiment in the")
println("plan that uses DeterministicTRSolver is affected.")
