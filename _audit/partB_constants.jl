using TrustRegionRadius, Printf, InteractiveUtils

println("=== B1. Constants each rule in benchmark/config.jl RULES actually runs with")
rules = [("RDelta",        RDelta(γ1 = 0.25, γ2 = 0.50, γ3 = 2.0)),
         ("RStep",         RStep( γ1 = 0.25, γ2 = 0.80, γ3 = 2.0)),
         ("RDFO",          RDFO(  γ1 = 0.25, γ2 = 0.50, γ3 = 2.0, ζ = 100.0)),
         ("RGrad",         RGrad( γ1 = 0.25, γ2 = 0.50, γ3 = 2.0, μ = 1.0)),
         ("RGradCapped",   RGradCapped(γ1 = 0.25, γ2 = 0.50, γ3 = 2.0, μ = 1.0, μ_max = 128.0)),
         ("RAdaptiveStep", RAdaptiveStep()),
         ("RAdaptiveGrad", RAdaptiveGrad()),
         ("RRTR",          RRTR()),            # commented out of RULES
         ("RRTRGrad",      RRTRGrad(μ = 1.0))] # commented out of RULES

@printf("%-15s %-8s %-8s %-8s %-10s %-10s %s\n",
        "rule", "g1", "g2", "g3", "Dmin", "Dmax", "rule-specific")
for (nm, r) in rules
    g(f) = hasfield(typeof(r), f) ? string(getfield(r, f)) : "--"
    spec = String[]
    for f in (:ζ, :μ, :μ_max, :λ1, :λ2, :η̃₁, :η̃₂, :half_test, :contract_on_step)
        hasfield(typeof(r), f) && push!(spec, "$f=$(getfield(r, f))")
    end
    @printf("%-15s %-8s %-8s %-8s %-10s %-10s %s\n",
            nm, g(:γ1), g(:γ2), g(:γ3), g(:Δmin), g(:Δmax), join(spec, " "))
end

println()
println("=== B2. Which thresholds does each update_radius! actually consume?")
println("(a discarded argument is declared as `::Float64` with no name)")
for T in (RDelta, RStep, RDFO, RGrad, RGradCapped, RAdaptiveStep,
          RAdaptiveGrad, RRTR, RRTRGrad)
    m = first(methods(update_radius!, (T, Float64, Float64, Bool, Float64, Float64,
                                       Float64, Float64, Float64)))
    # argument names: #self#, r, Δ, ρ, accepted, η1, η2, s_norm, crit_old, crit_new
    nms = Base.method_argnames(m)[2:end]
    lbl = ["Δ", "ρ", "accepted", "η1", "η2", "s_norm", "crit_old", "crit_new"]
    used = [lbl[i] for i in 1:8 if string(nms[i+1]) != "#unused#"]
    drop = [lbl[i] for i in 1:8 if string(nms[i+1]) == "#unused#"]
    @printf("%-15s uses: %-45s discards: %s\n", string(T), join(used, ", "), join(drop, ", "))
end

println()
println("=== B3. validate_thresholds: which rules have a method?")
for m in methods(validate_thresholds)
    println("  ", m.sig)
end

println()
println("=== B5. Branch vocabulary each rule can emit")
println("(from the source; verified by exercising each rule below)")
η1, η2 = 0.1, 0.9
probe = [("RDelta",        () -> RDelta()),
         ("RStep",         () -> RStep()),
         ("RDFO",          () -> RDFO(ζ = 1.0)),
         ("RGrad",         () -> RGrad(μ = 1.0)),
         ("RGradCapped",   () -> RGradCapped(μ = 1.0, μ_max = 1.0)),
         ("RAdaptiveStep", () -> RAdaptiveStep()),
         ("RAdaptiveGrad", () -> RAdaptiveGrad()),
         ("RRTR",          () -> RRTR()),
         ("RRTRGrad",      () -> RRTRGrad(μ = 1.0))]
for (nm, mk) in probe
    seen = Set{Symbol}()
    for ρ in (-1.0, 0.0, 0.05, 0.5, 0.95, 2.0), acc in (false, true),
        sn in (0.05, 0.9), Δ in (1.0,)
        r = mk()
        update_radius!(r, Δ, ρ, acc, η1, η2, sn, 1.0, 1.0)
        push!(seen, last_branch(r))
    end
    @printf("%-15s %s\n", nm, join(sort(collect(seen)), ", "))
end
