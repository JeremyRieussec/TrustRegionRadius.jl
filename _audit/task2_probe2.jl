# Two things the BigFloat result raises.
#
#  A. In one dimension SteihaugCG and ExactMS should return the same step, since
#     CG's first direction is -g and the exact minimiser along it is -g/a. If
#     they agree, the BigFloat arm can use SteihaugCG without changing the
#     subproblem solution, which matters because ExactMS needs eigen! and LAPACK
#     has no BigFloat method.
#  B. The BigFloat run ended at |g| = 1.06, which is not convergence. Find out
#     where it went. Hazard 2 says never assume the limit point.
include(joinpath(@__DIR__, "..", "benchmark", "initialisation.jl"))
using Printf

sinc(; x0 = 10.954122) = ADNLPModel(p -> sin(p[1]) / p[1], [float(x0)], name = "SINC1D")
const Y3 = 10.904121659428899827148702790189
const L3 = 0.09132520282305767214484313

sep(t) = (println(); println("="^76); println(t); println("="^76))

sep("A. SteihaugCG against ExactMS in one dimension, Float64")
@printf("%-10s %-12s %8s %8s %14s %14s\n", "mu_bar", "subsolver", "iter", "status",
        "final |g|", "final x")
for μ̄ in (1.0, 5.0, 20.0, 50.0)
    res = Dict{String, Any}()
    for (snm, mk) in (("SteihaugCG", () -> SteihaugCG()), ("ExactMS", () -> ExactMS()))
        nlp = sinc()
        st = tr_solve(nlp; rule = RGradCapped(γ1 = G1, γ2 = G2, γ3 = G3,
                                              μ = min(1.0, μ̄), μ_max = μ̄),
                      model = ExactHessian(), subsolver = mk(),
                      params = TRParams(η = 0.1, η1 = 0.1, η2 = 0.9, Δ0 = 1.0,
                                        tol = 1e-12, max_iterations = 3000),
                      trace = true)
        res[snm] = st
        @printf("%-10g %-12s %8d %8s %14.4e %14.9f\n", μ̄, snm, st.iter,
                string(st.status), Float64(st.dual_feas), st.solution[1])
    end
    a, b = res["SteihaugCG"], res["ExactMS"]
    sa = a.solver_specific[:step_trajectory]; sb = b.solver_specific[:step_trajectory]
    n = min(length(sa), length(sb))
    d = n == 0 ? NaN : maximum(abs(sa[i] - sb[i]) / max(sb[i], 1e-300) for i in 1:n)
    @printf("           largest relative step difference over %d shared iterations: %.3e\n",
            n, d)
end

sep("B. Where does the run go? mu_bar = 1, which is below the threshold")
@printf("  1/lambda*_3 = %.6f, so mu_bar*lambda*_3 = %.6f (below 1 means active)\n",
        1 / L3, 1.0 * L3)
nlp = sinc(); xs = [copy(nlp.meta.x0)]
st = tr_solve(nlp; rule = RGradCapped(γ1 = G1, γ2 = G2, γ3 = G3, μ = 1.0, μ_max = 1.0),
              model = ExactHessian(), subsolver = SteihaugCG(),
              params = TRParams(η = 0.1, η1 = 0.1, η2 = 0.9, Δ0 = 1.0,
                                tol = 1e-12, max_iterations = 400),
              trace = true, callback = (_n, _s, s) -> push!(xs, copy(s.solution)))
g = st.solver_specific[:grad_trajectory]
@printf("  status %s after %d iterations, x_end = %.9f, |g| = %.4e\n",
        st.status, st.iter, st.solution[1], Float64(st.dual_feas))
@printf("  distance to y_3 = %.6e\n", abs(st.solution[1] - Y3))
@printf("  x_k at k = 0,1,2,3,5,10,50,100: ")
for k in (1, 2, 3, 4, 6, 11, 51, 101)
    k <= length(xs) && @printf("%.6f ", xs[k][1])
end
println()
@printf("  |g_k| at the same k:            ")
for k in (1, 2, 3, 4, 6, 11, 51, 101)
    k <= length(g) && @printf("%.2e ", g[k])
end
println()
println("\nPROBE2 OK")
