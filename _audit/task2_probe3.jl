# The BigFloat run ended at |g| = 1.06, which is not convergence. Trace it.
include(joinpath(@__DIR__, "..", "benchmark", "initialisation.jl"))
using Printf
setprecision(BigFloat, 128)

const Y3 = BigFloat("10.904121659428899827148702790189")

for T in (Float64, BigFloat)
    nlp = ADNLPModel(p -> sin(p[1]) / p[1], [T == BigFloat ? BigFloat("10.954122") :
                                             10.954122], name = "SINC1D")
    p = T == BigFloat ?
        TRParams{BigFloat}(η = 0.1, η1 = 0.1, η2 = 0.9, Δ0 = 1.0, Δmin = 0.0,
                           Δmax = Inf, tol = BigFloat("1e-30"), max_iterations = 60) :
        TRParams(η = 0.1, η1 = 0.1, η2 = 0.9, Δ0 = 1.0, Δmin = 0.0,
                 Δmax = Inf, tol = 1e-30, max_iterations = 60)
    xs = Any[copy(nlp.meta.x0)]
    st = tr_solve(nlp; rule = RGradCapped(γ1 = G1, γ2 = G2, γ3 = G3, μ = 1.0, μ_max = 1.0),
                  model = ExactHessian(), subsolver = SteihaugCG(), params = p,
                  trace = true, callback = (_n, _s, s) -> push!(xs, copy(s.solution)))
    ss = st.solver_specific
    g, Δ = ss[:grad_trajectory], ss[:delta_trajectory]
    println("="^70)
    @printf("%s: status %s after %d iterations\n", T, st.status, st.iter)
    @printf("  x_end = %s\n", first(string(Float64(st.solution[1])), 20))
    @printf("  |g_end| = %.6e   dist to y_3 = %.4e\n",
            Float64(st.dual_feas), Float64(abs(st.solution[1] - Y3)))
    @printf("  %4s %22s %14s %14s\n", "k", "x_k", "|g_k|", "Delta_k")
    for k in (1, 2, 3, 4, 5, 10, 20, 40, min(60, length(g)))
        k <= length(g) || continue
        @printf("  %4d %22s %14.5e %14.5e\n", k - 1,
                first(string(Float64(xs[k][1])), 20), Float64(g[k]), Float64(Δ[k]))
    end
end
println("\nPROBE3 OK")
