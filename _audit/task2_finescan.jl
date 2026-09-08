# Is the RDFO classification monotone in zeta, or is there an isolated window?
# The 24-point scan of exp16 steps 0.377 -> 0.581 in zeta*lambda and so cannot
# see a window at 0.5. This resolves the region.
include(joinpath(@__DIR__, "..", "benchmark", "initialisation.jl"))
using Printf

roots = sinc_roots()
for j in (1, 3, 5, 7)
    λ = roots[j].λ
    println("="^74)
    @printf("j = %d,  Delta_0 = 0.01,  zeta*lambda from 0.20 to 2.00, 37 points\n", j)
    println("="^74)
    cls = Symbol[]; vals = Float64[]
    for t in range(0.20, 2.00; length = 37)
        z = t / λ
        r = sd_run(RDFO(γ1 = G1, γ2 = G2, γ3 = G3, ζ = z), j, roots;
                   kmax = 400, Δ0 = 0.01)
        c = sd_class(r)[1]
        push!(cls, c); push!(vals, t)
    end
    print("  zeta*l: ")
    for t in vals; @printf("%5.2f", t); end
    println()
    print("  class : ")
    for c in cls
        @printf("%5s", c === :active ? "a" : c === :inactive ? "i" : "n")
    end
    println()
    # monotone means every active precedes every non-active
    seen = false; mono = true
    for c in cls
        c === :active ? (seen && (mono = false)) : (seen = true)
    end
    first_non = findfirst(c -> c !== :active, cls)
    @printf("  monotone over this window: %s", mono)
    first_non === nothing || @printf("   first non-active at zeta*l = %.3f", vals[first_non])
    println()
end
println("\nFINESCAN OK")
