using TrustRegionRadius, ADNLPModels, NLPModels, LinearAlgebra, Printf, SolverCore
import TrustRegionRadius: hessian_op, dense_hessian, model_hprod!, update_model!,
                          reset_model!, reports_negative_curvature

sep(t) = (println(); println("="^72); println(t); println("="^72))

# ---------------------------------------------------------------- C1
sep("C1. NaN curvature classified as positive (subproblem.jl:211 `dBd <= 0`)")
println("NaN <= 0 evaluates to: ", NaN <= 0)

struct NaNHessian <: TrustRegionRadius.ModelHessian end
hessian_op(::NaNHessian, nlp, x) = NaNOp(length(x))
struct NaNOp; n::Int; end
Base.:*(::NaNOp, v::AbstractVector) = fill(NaN, length(v))
LinearAlgebra.mul!(y::AbstractVector, ::NaNOp, v::AbstractVector) = (fill!(y, NaN); y)
Base.eltype(::NaNOp) = Float64
Base.size(o::NaNOp) = (o.n, o.n)
dense_hessian(::NaNHessian, nlp, x) = fill(NaN, length(x), length(x))
reports_negative_curvature(::NaNHessian) = true

nlp = ADNLPModel(x -> (1 - x[1])^2 + 100(x[2] - x[1]^2)^2, [-1.2, 1.0])
s  = zeros(2); Hs = zeros(2); ws = SubWorkspace(zeros(2))
g  = grad(nlp, [-1.2, 1.0])
act = solve_subproblem!(SteihaugCG(), NaNHessian(), nlp, [-1.2, 1.0], g, 1.0, s, Hs, ws)
@printf("solve_subproblem! -> active = %s, s = %s, cg_iters = %d\n", act, s, ws.iters)
@printf("step contains NaN: %s   flagged interior (active == false): %s\n",
        any(isnan, s), act == false)

st = tr_solve(nlp; model = NaNHessian(), subsolver = SteihaugCG(),
              params = TRParams(tol = 1e-8, max_iterations = 20))
@printf("full solve -> status = %s, iter = %d, solution = %s, dual_feas = %.3e\n",
        st.status, st.iter, st.solution, st.dual_feas)

# ---------------------------------------------------------------- C2
sep("C2. NaN ratio poisons the Hei multiplier (rules.jl:757)")
println("max(NaN, 1e-300) = ", max(NaN, 1e-300))
r = RAdaptiveGrad(μ = 1.0)
@printf("start          mu = %.6g\n", r.μ)
update_radius!(r, 1.0, NaN, true, 0.1, 0.9, 0.5, 1.0, 1.0)
@printf("after rho=NaN  mu = %.6g   branch = %s\n", r.μ, last_branch(r))
for (i, ρ) in enumerate((0.95, 0.95, 0.5, -1.0, 0.95))
    Δ = update_radius!(r, 1.0, ρ, true, 0.1, 0.9, 0.5, 1.0, 1.0)
    @printf("  recovery call %d (rho=%5.2f): mu = %-10.6g Delta = %.6g\n", i, ρ, r.μ, Δ)
end
println("mu recovers from NaN: ", !isnan(r.μ))

# ---------------------------------------------------------------- C3
sep("C3. rho = -Inf is one sink for distinct failures (common.jl:516)")
println("st.rho = (isfinite(f_cand) && st.predicted > 0) ? actual/predicted : -Inf")
println()

# (a) non-finite trial objective. f is finite at the iterate and -Inf a short way
#     down the descent direction, so f_cand overflows.
badf = ADNLPModel(x -> -exp(x[1]^4) + x[2]^2, [5.0, 1.0])
sa = tr_solve(badf; rule = RDelta(), subsolver = SteihaugCG(),
              params = TRParams(Δ0 = 2.0, tol = 1e-8, max_iterations = 8), trace = true)

# (b) non-positive predicted reduction, forced with a subsolver that returns an
#     ascent step. A correct subsolver minimises the model, so predicted > 0 for
#     any nonzero step; this branch is otherwise unreachable from tr_solve.
struct AscentSolver <: TrustRegionRadius.SubproblemSolver end
function TrustRegionRadius.solve_subproblem!(::AscentSolver, model, nlp, x, g, Δ,
                                             s, Hs, ws; curv = nothing)
    @. s = g * (Δ / max(norm(g), eps()))        # uphill, on the boundary
    TrustRegionRadius._apply_op!(Hs, hessian_op(model, nlp, x), s)
    return true
end
sb = tr_solve(ADNLPModel(x -> sum(abs2, x), [1.0, 1.0]); subsolver = AscentSolver(),
              params = TRParams(tol = 1e-8, max_iterations = 8), trace = true)

for (lbl, ss) in (("(a) non-finite f_cand", sa), ("(b) non-positive pred", sb))
    ρ = ss.solver_specific[:ratio_trajectory]
    @printf("%-24s status=%-12s count(rho == -Inf) = %d of %d\n",
            lbl, ss.status, count(==(-Inf), ρ), length(ρ))
    @printf("%-24s first four rho: %s\n", "", ρ[1:min(4, length(ρ))])
end
println()
println("Both reach rho = -Inf and :ratio_trajectory records the same value for")
println("both, so the trace cannot say which failure occurred.")
