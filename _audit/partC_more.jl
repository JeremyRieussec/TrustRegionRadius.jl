using TrustRegionRadius, ADNLPModels, NLPModels, LinearAlgebra, Printf, Random, SolverCore
import TrustRegionRadius: hessian_op, dense_hessian, reports_negative_curvature
sep(t) = (println(); println("="^72); println(t); println("="^72))

# ---------------------------------------------------------------- C4
sep("C4. Only DomainError becomes :exception (common.jl:473)")
struct InfHessian <: TrustRegionRadius.ModelHessian end
dense_hessian(::InfHessian, nlp, x) = fill(Inf, length(x), length(x))
hessian_op(::InfHessian, nlp, x) = Inf * I
reports_negative_curvature(::InfHessian) = true

nlp = ADNLPModel(x -> sum(abs2, x), [1.0, 1.0])
println("ExactMS with a non-finite model Hessian:")
try
    tr_solve(nlp; model = InfHessian(), subsolver = ExactMS(),
             params = TRParams(tol = 1e-8, max_iterations = 5))
    println("  completed without raising")
catch e
    @printf("  ESCAPED solve!: %s\n", typeof(e))
    @printf("  message: %s\n", first(sprint(showerror, e), 160))
end
println("common.jl:473 is `err isa DomainError || rethrow()`, so an ErrorException")
println("from subproblem.jl:343 is not converted to :exception; it propagates.")

# ---------------------------------------------------------------- C5
sep("C5. model_grad_evals is identically zero while SPDTarget calls grad")
q = ADNLPModel(p -> p[1]^4 - p[1]^3 + (0.25 - p[1]/2)*p[2]^2 + p[2]^4/4, [-0.5, 0.6])
solver = DeterministicTRSolver(q; rule = RDelta(),
                               model = SPDTarget(target = [0.75, 0.0]),
                               params = TRParams(tol = 1e-8, max_iterations = 200))
st = SolverCore.solve!(solver, q; trace = true)
@printf("SPDTarget run: status = %s, iter = %d\n", st.status, st.iter)
@printf("neval_grad reported        = %d\n", neval_grad(q))
@printf("model_grad_evals(solver)   = %d   <- the correction offered\n",
        model_grad_evals(solver))
@printf("gradients the ITERATION used, at most one per accepted step + 1 = %d\n",
        count(st.solver_specific[:accepted_trajectory]) + 1)
println("The difference is the model's own gradient calls, and the correction is 0,")
println("so `neval_grad - model_grad_evals` overstates the algorithm's gradient cost.")

# ---------------------------------------------------------------- C7
sep("C7. SR1 (and every model) updated on accepted steps only (common.jl:540,559)")
rosen = ADNLPModel(x -> (1 - x[1])^2 + 100(x[2] - x[1]^2)^2, [-1.2, 1.0])
s7 = tr_solve(rosen; rule = RDelta(), model = SR1Model(mem = 5),
              params = TRParams(tol = 1e-8, max_iterations = 300), trace = true)
acc = s7.solver_specific[:accepted_trajectory]
@printf("iterations = %d, accepted = %d, rejected = %d\n",
        length(acc), count(acc), count(!, acc))
@printf("update_model! calls = %d (accepted only); rejected steps discarded = %d\n",
        count(acc), count(!, acc))
println("`update_model!` sits inside `if st.accepted`. On a rejected step the code")
println("never evaluates grad at the trial point, so y = g_new - g_old does not")
println("exist there: the omission is structural, not a missing call.")

# ---------------------------------------------------------------- C8
sep("C8. lambda_min_estimate above n = 200 is a Lanczos Ritz value (bounds from above)")
println("signature: lambda_min_estimate(model, nlp, x; nmax = 200, lanczos_k = 40)")
println("occurrences of `nmax`/`lanczos_k` under src/Trust-region/: none (grep)")
println("=> neither is reachable through tr_solve or TRParams.")
println()
Random.seed!(1)
n = 400
w = collect(range(-2.0, 8.0; length = n))          # true lambda_min = -2 exactly
big = ADNLPModel(z -> 0.5 * sum(w .* z .^ 2), zeros(n))
m = ExactHessian(); reset_model!(m, n)
xb = ones(n) ./ 3
est = lambda_min_estimate(m, big, xb)
den = lambda_min_estimate(m, big, xb; nmax = 10_000)
@printf("n = %d, true lambda_min = %.6f\n", n, minimum(w))
@printf("Lanczos (n > nmax, 40 steps) = %.6f\n", est)
@printf("dense branch (nmax forced)   = %.6f\n", den)
@printf("estimate is ABOVE the truth by %.6f  (optimistic: %s)\n",
        est - minimum(w), est > minimum(w))
