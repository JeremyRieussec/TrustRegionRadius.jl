# Smoke-run one configuration of D1, D3, D4b, D6 and A6 on the analytic set.
include(joinpath(@__DIR__, "..", "benchmark", "initialisation.jl"))
using Printf
sep(t) = (println(); println("="^72); println(t); println("="^72))
cut(e) = first(replace(sprint(showerror, e), "\n" => " "), 130)

const G1, G2, G3 = 0.25, 0.50, 2.0
const P8 = TRParams(η = 0.1, η1 = 0.1, η2 = 0.9, Δ0 = 1.0, Δmin = 0.0, Δmax = Inf,
                    max_iterations = 20_000, tol = 1e-8, tol_H = -1.0)
R8 = [("RDelta",        () -> RDelta(γ1=G1, γ2=G2, γ3=G3, Δmin=0.0)),
      ("RStep",         () -> RStep(γ1=G1, γ2=G2, γ3=G3, Δmin=0.0)),
      ("RDFO",          () -> RDFO(γ1=G1, γ2=G2, γ3=G3, ζ=1.0, Δmin=0.0)),
      ("RGrad",         () -> RGrad(γ1=G1, γ2=G2, γ3=G3, μ=1.0, Δmin=0.0)),
      ("RGradCapped",   () -> RGradCapped(γ1=G1, γ2=G2, γ3=G3, μ=1.0, μ_max=128.0, Δmin=0.0)),
      ("RRTR",          () -> RRTR(γ1=G1, γ2=G2, γ3=G3, Δmin=0.0)),
      ("RAdaptiveStep", () -> RAdaptiveStep(γ1=G1, γ2=G2, γ3=G3, λ1=5.0, λ2=5.0, Δmin=0.0)),
      ("RAdaptiveGrad", () -> RAdaptiveGrad(γ1=G1, γ2=G2, γ3=G3, λ1=5.0, λ2=5.0, μ=1.0, Δmin=0.0))]

# ------------------------------------------------------------------ D1
sep("D1. Radius asymptotics on exp(-x). Needs Delta trajectory, sum D, sum D^2.")
expd = () -> ADNLPModel(p -> exp(-p[1]), [0.0], name = "EXPDECAY")
d1 = [("RDelta", () -> RDelta(γ1=G1, γ2=G2, γ3=G3, Δmin=0.0)),
      ("RGrad(mu0=1)", () -> RGrad(γ1=G1, γ2=G2, γ3=G3, μ=1.0, Δmin=0.0)),
      ("RGradCapped(mubar=1)", () -> RGradCapped(γ1=G1, γ2=G2, γ3=G3, μ=1.0, μ_max=1.0, Δmin=0.0)),
      ("RDFO(zeta=1)", () -> RDFO(γ1=G1, γ2=G2, γ3=G3, ζ=1.0, Δmin=0.0))]
@printf("%-24s %-12s %6s %10s %12s %12s %10s\n",
        "rule", "status", "iter", "D_end/D_0", "sum D", "sum D^2", "finite?")
for (nm, mk) in d1
    try
        st = tr_solve(expd(); rule = mk(), model = ExactHessian(), subsolver = SteihaugCG(),
                      params = TRParams(η=0.1, η1=0.1, η2=0.9, Δ0=1.0, Δmin=0.0,
                                        tol=1e-300, max_iterations=500), trace = true)
        Δ = st.solver_specific[:delta_trajectory]
        @printf("%-24s %-12s %6d %10.3g %12.5g %12.5g %10s\n", nm, st.status, st.iter,
                Δ[end]/Δ[1], sum(Δ), sum(abs2, Δ), all(isfinite, Δ))
    catch e
        @printf("%-24s FAILED: %s\n", nm, cut(e))
    end
end

# ------------------------------------------------------------------ D3
sep("D3. Silencing on ILLCOND. Needs frac(cg_iters==1 AND active), cos(s,-g).")
ill = () -> ADNLPModel(x -> x[1]^2 + 1000x[2]^2 + 0.01x[1]*x[2], [1.0, 1.0], name="ILLCOND")
@printf("%-22s %-12s %6s %14s %14s\n", "config", "status", "iter", "frac cg1&act", "cos tail")
for μ in (1e-4, 1e-2, 1.0)
    try
        st = tr_solve(ill(); rule = RGradCapped(γ1=G1, γ2=G2, γ3=G3, μ=μ, μ_max=μ, Δmin=0.0),
                      model = ExactHessian(), subsolver = SteihaugCG(),
                      params = P8, trace = true)
        ss = st.solver_specific
        cg, ac, cs = ss[:cg_iters_trajectory], ss[:active_trajectory], ss[:cos_cauchy_trajectory]
        f = isempty(cg) ? NaN : count(i -> cg[i] == 1 && ac[i], eachindex(cg)) / length(cg)
        @printf("%-22s %-12s %6d %14.3f %14.4f\n",
                "RGradCapped(mu=$μ)", st.status, st.iter, f,
                isempty(cs) ? NaN : cs[end])
    catch e
        @printf("%-22s FAILED: %s\n", "RGradCapped(mu=$μ)", cut(e))
    end
end

# ------------------------------------------------------------------ D4b
sep("D4b. Flat well escape, RGradCapped(mu0 = 1, mubar in 1e-3..1e2)")
flat(ε) = ADNLPModel(p -> 0.5p[1]^2 + ε*(p[2]^4/4 - p[2]^2/2), [0.0, 1e-3], name="FLAT")
for μ in (1e-3, 1e-1, 1.0, 1e2)
    try
        r = RGradCapped(γ1=G1, γ2=G2, γ3=G3, μ=1.0, μ_max=μ, Δmin=0.0)
        st = tr_solve(flat(1e-2); rule = r, model = ExactHessian(), subsolver = SteihaugCG(),
                      params = P8, trace = true)
        @printf("  mubar=%-8g %-12s iter=%d\n", μ, st.status, st.iter)
    catch e
        @printf("  mubar=%-8g BLOCKED: %s\n", μ, cut(e))
    end
end

# ------------------------------------------------------------------ D6
sep("D6. Threshold on sin(x)/x")
try
    m = ADNLPModel(p -> (abs(p[1]) < 1e-12 ? 1.0 : sin(p[1])/p[1]), [10.954122], name="SINC1D")
    println("  sinc_1d as written builds")
catch e
    @printf("  BLOCKED at problem construction: %s\n", cut(e))
end

# ------------------------------------------------------------------ A6
sep("A6. Roster R8 on the analytic set, cost matrix + profile")
probs = [t[2] for t in analytic_problems()]   # run_matrix wants BARE thunks
cfgs = [TRConfig(nm; rule = mk(), model = ExactHessian(), subsolver = SteihaugCG(),
                 params = P8) for (nm, mk) in R8]
try
    T, S = run_matrix(probs, cfgs; cost = :iter)
    @printf("  cost matrix %dx%d, finite entries %d / %d, solved %d\n",
            size(T,1), size(T,2), count(isfinite, T), length(T), count(!isnothing, S))
    τ, prof = performance_profile(T)
    @printf("  performance_profile -> tau %d, prof %s, prof[end,:] = %s\n",
            length(τ), size(prof), round.(prof[end, :], digits=3))
    println("  solve rate per configuration:")
    for (j, (nm, _)) in enumerate(R8)
        @printf("    %-16s %d/%d\n", nm, count(S[:, j]), size(S, 1))
    end
catch e
    @printf("  FAILED: %s\n", cut(e))
end
