# Preconditions for Task 2. Each is a stop condition or changes the design.
#
#  1. Does the branched sinc_1d build as an ADNLPModel?
#  2. Do the API names of the prompt exist?
#  3. Hazard 1: does the subsolver return the closed-form one-dimensional step?
#  4. Section 6: is the solver generic over the element type, so BigFloat runs?
include(joinpath(@__DIR__, "..", "benchmark", "initialisation.jl"))
using Printf

sep(t) = (println(); println("="^76); println(t); println("="^76))

# --- 1. the model ------------------------------------------------------------
sep("1. sinc_1d, branched as the prompt gives it, and unbranched")
sinc_branched(; x0 = 10.954122) =
    ADNLPModel(p -> (abs(p[1]) < 1e-12 ? 1.0 : sin(p[1]) / p[1]), [float(x0)],
               name = "SINC1D")
sinc_plain(; x0 = 10.954122) =
    ADNLPModel(p -> sin(p[1]) / p[1], [float(x0)], name = "SINC1D")

for (nm, mk) in (("branched", sinc_branched), ("plain", sinc_plain))
    try
        nlp = mk()
        g = grad(nlp, nlp.meta.x0); H = hess(nlp, nlp.meta.x0)
        @printf("  %-10s builds. f=%.10f  g=%.6e  H=%.10f\n", nm,
                obj(nlp, nlp.meta.x0), g[1], H[1, 1])
    catch e
        @printf("  %-10s REJECT: %s\n", nm,
                first(replace(sprint(showerror, e), '\n' => ' '), 150))
    end
end

# --- 2. the API --------------------------------------------------------------
sep("2. API names of the prompt against the package")
for s in (:theta_trajectory, :inactivity_index, :branch_counts, :kappa_bar,
          :kappa_bar_empirical, :observed_order, :active_fraction,
          :RDFO, :RGradCapped, :RGrad, :ExactMS, :ExactHessian)
    @printf("  %-22s %s\n", s, isdefined(TrustRegionRadius, s) ? "exists" : "MISSING")
end

# --- 3. hazard 1: the closed-form one-dimensional step -----------------------
sep("3. Does the subsolver return the closed-form 1-D step?")
# In one dimension with the exact Hessian a, the exact solution is -g/a when
# that lies inside the region and a > 0, and -Delta*sign(g) otherwise.
nlp = sinc_plain()
worst = 0.0; nchk = 0; nneg = 0
st = tr_solve(nlp; rule = RGradCapped(γ1 = G1, γ2 = G2, γ3 = G3, μ = 1.0, μ_max = 1.0),
              model = ExactHessian(), subsolver = ExactMS(),
              params = TRParams(η = 0.1, η1 = 0.1, η2 = 0.9, Δ0 = 1.0,
                                tol = 1e-12, max_iterations = 200),
              trace = true)
ss = st.solver_specific
Δ, gt, sn = ss[:delta_trajectory], ss[:grad_trajectory], ss[:step_trajectory]
# The iterate path is needed to evaluate a_k = f''(x_k).
xs = [copy(nlp.meta.x0)]
nlp2 = sinc_plain()
tr_solve(nlp2; rule = RGradCapped(γ1 = G1, γ2 = G2, γ3 = G3, μ = 1.0, μ_max = 1.0),
         model = ExactHessian(), subsolver = ExactMS(),
         params = TRParams(η = 0.1, η1 = 0.1, η2 = 0.9, Δ0 = 1.0,
                           tol = 1e-12, max_iterations = 200),
         trace = true, callback = (_n, _s, s) -> push!(xs, copy(s.solution)))
for k in eachindex(sn)
    x = xs[k]
    a = hess(nlp2, x)[1, 1]; g = grad(nlp2, x)[1]
    pred = (a > 0 && abs(g) / a <= Δ[k]) ? abs(g) / a : Δ[k]
    a <= 0 && (global nneg += 1)
    d = abs(pred - sn[k]) / max(pred, 1e-300)
    global worst = max(worst, d); global nchk += 1
end
@printf("  checked %d iterations, %d with a_k <= 0\n", nchk, nneg)
@printf("  largest relative discrepancy |s_k| against the closed form: %.3e\n", worst)
println(worst < 1e-8 ? "  AGREES" : "  DOES NOT AGREE, stop and report")

# --- 4. section 6: BigFloat end to end ---------------------------------------
sep("4. Is the solver generic over the element type?")
try
    setprecision(BigFloat, 128)
    nb = ADNLPModel(p -> sin(p[1]) / p[1], [BigFloat("10.954122")], name = "SINC1D-BF")
    @printf("  model eltype %s\n", eltype(nb.meta.x0))
    sb = tr_solve(nb; rule = RGradCapped(γ1 = G1, γ2 = G2, γ3 = G3, μ = 1.0, μ_max = 1.0),
                  model = ExactHessian(), subsolver = ExactMS(),
                  params = TRParams(η = 0.1, η1 = 0.1, η2 = 0.9, Δ0 = 1.0,
                                    tol = 1e-30, max_iterations = 200),
                  trace = true)
    @printf("  RUNS. status %s, iter %d, |g| = %.6e (type %s)\n",
            sb.status, sb.iter, Float64(sb.dual_feas), typeof(sb.dual_feas))
catch e
    @printf("  DOES NOT RUN: %s\n",
            first(replace(sprint(showerror, e), '\n' => ' '), 220))
end

println("\nPROBE OK")
