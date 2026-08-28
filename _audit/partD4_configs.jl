using TrustRegionRadius, ADNLPModels, NLPModels, LinearAlgebra, Printf
sep(t) = (println(); println("="^72); println(t); println("="^72))
cut(e) = first(replace(sprint(showerror, e), "\n" => " "), 120)
function try_it(lbl, f)
    try
        v = f()
        @printf("  OK      %-42s %s\n", lbl, typeof(v))
        return true
    catch e
        @printf("  REJECT  %-42s %s\n", lbl, cut(e))
        return false
    end
end

sep("D4. Does PartIII-run-configs-v1.jl construct?")
G1, G2, G3 = 0.25, 0.50, 2.0

println("0.1 HARMONISED_PARAMS")
try_it("TRParams(...)", () -> TRParams(η = 0.1, η1 = 0.1, η2 = 0.9, Δ0 = 1.0,
    Δmin = 0.0, Δmax = Inf, max_iterations = 20_000, tol = 1e-8, tol_H = -1.0))

println("\n0.2 RULES_8, the eight configurations at one set of constants")
try_it("RDelta",        () -> RDelta(γ1 = G1, γ2 = G2, γ3 = G3, Δmin = 0.0))
try_it("RStep",         () -> RStep(γ1 = G1, γ2 = G2, γ3 = G3, Δmin = 0.0))
try_it("RDFO",          () -> RDFO(γ1 = G1, γ2 = G2, γ3 = G3, ζ = 1.0, Δmin = 0.0))
try_it("RGrad",         () -> RGrad(γ1 = G1, γ2 = G2, γ3 = G3, μ = 1.0, Δmin = 0.0))
try_it("RGradCapped",   () -> RGradCapped(γ1 = G1, γ2 = G2, γ3 = G3, μ = 1.0,
                                          μ_max = 128.0, Δmin = 0.0))
try_it("RRTR",          () -> RRTR(γ1 = G1, γ2 = G2, γ3 = G3, η̃₁ = 0.05, η̃₂ = 0.9,
                                   Δmin = 0.0))
try_it("RAdaptiveStep", () -> RAdaptiveStep(γ1 = G1, γ2 = G2, γ3 = G3,
                                            λ1 = 5.0, λ2 = 5.0, Δmin = 0.0))
try_it("RAdaptiveGrad", () -> RAdaptiveGrad(γ1 = G1, γ2 = G2, γ3 = G3,
                                            λ1 = 5.0, λ2 = 5.0, μ = 1.0, Δmin = 0.0))

println("\n0.5 DEFAULT_SUBSOLVER_V2")
try_it("SteihaugCG(χ, θ, max_iters)", () -> SteihaugCG(χ = 0.1, θ = 0.5, max_iters = 200))
println("  SteihaugCG keywords actually accepted:")
for m in methods(SteihaugCG)
    println("    ", sort(collect(Base.kwarg_decl(m))))
end

println("\n1. The two designed problems")
exp_decay(; x0 = 0.0) = ADNLPModel(p -> exp(-p[1]), [float(x0)], name = "EXPDECAY")
try_it("exp_decay()", () -> exp_decay())
sinc_1d(; x0 = 10.954122) =
    ADNLPModel(p -> (abs(p[1]) < 1e-10 ? 1.0 : sin(p[1]) / p[1]), [float(x0)], name = "SINC1D")
try_it("sinc_1d()", () -> sinc_1d())

println("\n2. D4b / A5: does RGradCapped accept μ_max < μ?")
println("   configs line 231: RGradCapped(..., μ = 1.0, μ_max = μ) over D4_MU")
for μ in (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2)
    try_it(@sprintf("RGradCapped(μ = 1.0, μ_max = %g)", μ),
           () -> RGradCapped(γ1 = G1, γ2 = G2, γ3 = G3, μ = 1.0, μ_max = μ, Δmin = 0.0))
end
println("\n   D3 line 187 form, μ = μ_max together:")
for μ in (1e-3, 1.0, 1e2)
    try_it(@sprintf("RGradCapped(μ = %g, μ_max = %g)", μ, μ),
           () -> RGradCapped(γ1 = G1, γ2 = G2, γ3 = G3, μ = μ, μ_max = μ, Δmin = 0.0))
end

println("\n3. Hei requirement γ3 > 1 + γ2: is it enforced?")
try_it("RAdaptiveStep(γ2 = 0.5, γ3 = 1.2)  [violates]",
       () -> RAdaptiveStep(γ1 = G1, γ2 = 0.5, γ3 = 1.2))
try_it("RAdaptiveGrad(γ2 = 0.5, γ3 = 1.2)  [violates]",
       () -> RAdaptiveGrad(γ1 = G1, γ2 = 0.5, γ3 = 1.2))
