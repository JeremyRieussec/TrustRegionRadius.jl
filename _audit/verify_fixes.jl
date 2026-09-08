# Each fix, checked by something that failed before it and passes after.
using TrustRegionRadius, ADNLPModels, NLPModels, LinearAlgebra, Printf, SolverCore, Random
import TrustRegionRadius: hessian_op, dense_hessian, reports_negative_curvature
ok(b) = b ? "PASS" : "**FAIL**"
sep(n, t) = (println(); println("-"^72); println("FIX $n. $t"); println("-"^72))

# ---- 1 -------------------------------------------------------------------
sep(1, "RGradCapped accepts mu_max < mu0 (D4b, A5)")
r = RGradCapped(μ = 1.0, μ_max = 1e-3)
@printf("  RGradCapped(μ=1.0, μ_max=1e-3) constructs: %s  (μ clamped to %g, μ_max %g)\n",
        ok(true), r.μ, r.μ_max)
@printf("  μ0 <= μ_max still holds: %s\n", ok(r.μ <= r.μ_max))
grid = [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2]
built = all(μ -> (RGradCapped(μ = 1.0, μ_max = μ); true), grid)
@printf("  whole D4_MU grid constructs: %s\n", ok(built))

# ---- 2 -------------------------------------------------------------------
sep(2, "sinc_1d builds (D6)")
sinc_1d(; x0 = 10.954122) = ADNLPModel(p -> sin(p[1]) / p[1], [float(x0)], name = "SINC1D")
st = tr_solve(sinc_1d(); rule = RDelta(), params = TRParams(tol = 1e-10, max_iterations = 500))
@printf("  builds and solves: %s  x = %.8f (root of tan x = x is 10.90412166)\n",
        ok(isapprox(st.solution[1], 10.90412166; atol = 1e-6)), st.solution[1])

# ---- 4, 6 ----------------------------------------------------------------
sep("4/6", "RAdaptiveGrad: hold branch present, NaN cannot poison mu")
η1, η2 = 0.1, 0.9
rag = RAdaptiveGrad(μ = 1.0)
before = rag.μ
update_radius!(rag, 1.0, 0.95, true, η1, η2, 0.1, 1.0, 1.0)     # ‖s‖ ≤ Δ/2
@printf("  very successful, short step -> branch %s, mu unchanged: %s\n",
        last_branch(rag), ok(rag.μ == before))
update_radius!(rag, 1.0, 0.95, true, η1, η2, 0.9, 1.0, 1.0)     # ‖s‖ > Δ/2
@printf("  very successful, long step  -> branch %s, mu grew: %s\n",
        last_branch(rag), ok(rag.μ > before))
rn = RAdaptiveGrad(μ = 1.0)
update_radius!(rn, 1.0, NaN, true, η1, η2, 0.9, 1.0, 1.0)
for ρ in (0.95, 0.5, -1.0); update_radius!(rn, 1.0, ρ, true, η1, η2, 0.9, 1.0, 1.0); end
@printf("  after rho = NaN and three good calls, mu = %g finite: %s\n", rn.μ, ok(isfinite(rn.μ)))

# ---- 9 -------------------------------------------------------------------
sep(9, "RRTRGrad: gamma2 shrink branch and the mu_max cap")
rrg = RRTRGrad(μ = 1.0, μ_max = Inf)
seen = Symbol[]
for (ρ̃, sn) in ((0.95, 0.9), (0.5, 0.9), (0.01, 0.9), (0.95, 0.1))
    update_radius!(rrg, 1.0, ρ̃, true, η1, η2, sn, 1.0, 1.0); push!(seen, last_branch(rrg))
end
@printf("  branches reachable: %s\n", seen)
@printf("  has :shrink (the intermediate branch): %s\n", ok(:shrink in seen))
rcap = RRTRGrad(μ = 1.0, μ_max = 1.0)
update_radius!(rcap, 1.0, 0.95, true, η1, η2, 0.9, 1.0, 1.0)
@printf("  cap applied: branch %s, mu = %g: %s\n",
        last_branch(rcap), rcap.μ, ok(rcap.μ == 1.0 && last_branch(rcap) === :expand_capped))
@printf("  has gamma2 field: %s   has mu_max field: %s\n",
        ok(hasfield(RRTRGrad, :γ2)), ok(hasfield(RRTRGrad, :μ_max)))

# ---- 7 -------------------------------------------------------------------
sep(7, "NaN curvature no longer classified as interior")
struct NaNHessian <: TrustRegionRadius.ModelHessian end
struct NaNOp; n::Int; end
Base.:*(::NaNOp, v::AbstractVector) = fill(NaN, length(v))
LinearAlgebra.mul!(y::AbstractVector, ::NaNOp, v::AbstractVector) = (fill!(y, NaN); y)
Base.eltype(::NaNOp) = Float64
Base.size(o::NaNOp) = (o.n, o.n)
hessian_op(::NaNHessian, nlp, x) = NaNOp(length(x))
dense_hessian(::NaNHessian, nlp, x) = fill(NaN, length(x), length(x))
reports_negative_curvature(::NaNHessian) = true
nlp = ADNLPModel(x -> (1 - x[1])^2 + 100(x[2] - x[1]^2)^2, [-1.2, 1.0])
s = zeros(2); Hs = zeros(2); ws = SubWorkspace(zeros(2)); g = grad(nlp, [-1.2, 1.0])
act = solve_subproblem!(SteihaugCG(), NaNHessian(), nlp, [-1.2, 1.0], g, 1.0, s, Hs, ws)
@printf("  active = %s (was false, i.e. reported INTERIOR): %s\n", act, ok(act))
@printf("  cg_iters = %d (was 100, the whole budget): %s\n", ws.iters, ok(ws.iters < 100))

# ---- 8 -------------------------------------------------------------------
sep(8, "a non-DomainError from the subsolver becomes :exception")
struct InfHessian <: TrustRegionRadius.ModelHessian end
dense_hessian(::InfHessian, nlp, x) = fill(Inf, length(x), length(x))
hessian_op(::InfHessian, nlp, x) = Inf * I
reports_negative_curvature(::InfHessian) = true
q = ADNLPModel(x -> sum(abs2, x), [1.0, 1.0])
res = try
    stx = tr_solve(q; model = InfHessian(), subsolver = ExactMS(),
                   params = TRParams(tol = 1e-8, max_iterations = 5))
    "status = $(stx.status)"
catch e
    "ESCAPED: $(typeof(e))"
end
@printf("  %s : %s\n", res, ok(startswith(res, "status = exception")))

# ---- 10 ------------------------------------------------------------------
sep(10, "model_grad_evals counts SPDTarget's own gradient calls")
qq = ADNLPModel(p -> p[1]^4 - p[1]^3 + (0.25 - p[1]/2)*p[2]^2 + p[2]^4/4, [-0.5, 0.6])
solver = DeterministicTRSolver(qq; rule = RDelta(), model = SPDTarget(target = [0.75, 0.0]),
                               params = TRParams(tol = 1e-8, max_iterations = 200))
sp = SolverCore.solve!(solver, qq)
@printf("  neval_grad = %d, model_grad_evals = %d, algorithm = %d: %s\n",
        neval_grad(qq), model_grad_evals(solver),
        neval_grad(qq) - model_grad_evals(solver), ok(model_grad_evals(solver) > 0))

# ---- 12 ------------------------------------------------------------------
sep(12, "curv_nmax / curv_lanczos_k reachable through TRParams")
p1 = TRParams(curv_nmax = 50, curv_lanczos_k = 10)
@printf("  TRParams(curv_nmax=50, curv_lanczos_k=10): %s\n",
        ok(p1.curv_nmax == 50 && p1.curv_lanczos_k == 10))
n = 400
w = collect(range(-2.0, 8.0; length = n))
big = ADNLPModel(z -> 0.5 * sum(w .* z .^ 2), zeros(n))
m = ExactHessian(); reset_model!(m, n)
lo = lambda_min_estimate(m, big, ones(n)./3; nmax = 200,   lanczos_k = 40)
hi = lambda_min_estimate(m, big, ones(n)./3; nmax = 10_000)
@printf("  lambda_min: Lanczos %.6f vs dense %.6f; a run can now force the dense branch\n", lo, hi)
