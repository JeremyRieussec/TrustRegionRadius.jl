# Why is zeta* NaN, and why is kappa_bar_empirical NaN?
include(joinpath(@__DIR__, "..", "benchmark", "initialisation.jl"))
using Printf

roots = sinc_roots()

println("="^76)
println("A. kappa_bar_empirical, both conventions")
println("="^76)
println("(re-running with direct stats access)")
for j in (1, 3)
    λ = roots[j].λ
    for (nm, m, km) in (("below", 0.5 / λ, 400), ("above", 4.0 / λ, 200))
        nlp = sinc_model(roots[j].y + SINC_OFFSET)
        st = tr_solve(nlp; rule = RGradCapped(γ1 = G1, γ2 = G2, γ3 = G3,
                                              μ = min(1.0, m), μ_max = m),
                      model = ExactHessian(), subsolver = ExactMS(),
                      params = sd_params(kmax = km), trace = true)
        @printf("  j=%d %-6s mu=%8.4g  inactive_only=true: %10.4g   false: %10.4g  ",
                j, nm, m, kappa_bar_empirical(st),
                kappa_bar_empirical(st; inactive_only = false))
        @printf("inactivity=%s\n", string(inactivity_index(st)))
    end
end

println()
println("="^76)
println("B. The RDFO brackets. Why does the predicate not straddle?")
println("="^76)
for j in (1, 3), Δ0 in (0.01, 1.0)
    λ = roots[j].λ
    println("--- j=$j  Delta_0=$Δ0  (1/lambda = $(round(1/λ, digits=4))) ---")
    for z in [0.005, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0] ./ λ
        r = sd_run(RDFO(γ1 = G1, γ2 = G2, γ3 = G3, ζ = z), j, roots;
                   kmax = 400, Δ0 = Δ0)
        c, w, _ = sd_class(r)
        @printf("    zeta=%9.4g  zeta*l=%7.3f  class=%-10s iter=%5d status=%-12s inact=%s\n",
                z, z * λ, string(c), r.iter, string(r.status), string(r.inactivity))
    end
end
println("\nDIAG OK")
