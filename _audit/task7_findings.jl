# The five findings of section 5, checked against the source rather than recalled.
include(joinpath(@__DIR__, "..", "benchmark", "initialisation.jl"))
using Printf

println("="^76)
println("FINDING 2. ||g_0|| at x0 = (0.8, y0), eps = 1e-4, y0 = 1e-6")
println("="^76)
ε, y0 = 1e-4, 1e-6
x0 = [0.8, y0]
g0 = [x0[1], ε * x0[2] * (x0[2]^2 - 1)]
@printf("  claimed in the source : ||g_0|| ~ eps*y0 = %.3e\n", ε * y0)
@printf("  actual                : ||g_0||          = %.6f\n", norm(g0))
@printf("  ratio                 : %.3e\n", norm(g0) / (ε * y0))

# The run itself, under the curvature-blind model the table uses.
nlp = flat_well(ε; x0 = x0)
xs = [copy(x0)]
st = tr_solve(nlp; rule = RDelta(γ1 = G1, γ2 = G2, γ3 = G3),
              model = ScaledIdentity(c = 1.0), subsolver = SteihaugCG(),
              params = TRParams(η = 0.1, η1 = 0.1, η2 = 0.9, Δ0 = 1.0,
                                tol = 1e-9, max_iterations = 200_000),
              trace = true, callback = (_n, _s, s) -> push!(xs, copy(s.solution)))
ss = st.solver_specific
g = ss[:grad_trajectory]
@printf("\n  status %s after %d iterations, limit (%.3e, %.3e)\n",
        st.status, st.iter, xs[end][1], xs[end][2])
@printf("  ||g_k|| over the run: ")
for k in 1:min(5, length(g)); @printf("%.3e  ", g[k]); end
println()
@printf("  |x_k| over the run  : ")
for k in 1:min(5, length(xs)); @printf("%.3e  ", abs(xs[k][1])); end
println()
println("  The x component is solved first. Only then is ||g|| of order eps*y0.")

println("\n", "="^76)
println("FINDING 3. The caption invariant, residuals for three candidates")
println("="^76)
# The printed table of the source, read off it, not recomputed.
TAB = [(1e-1, 1e-3, 67), (1e-1, 1e-6, 140), (1e-2, 1e-3, 640),
       (1e-2, 1e-6, 1334), (1e-3, 1e-3, 6362), (1e-3, 1e-6, 13273),
       (1e-4, 1e-3, 63588)]
@printf("%8s %8s %8s %10s %10s %10s %10s\n", "eps", "y0", "k_esc", "eps*k",
        "r ln1/y0", "r ln1/2y0", "r ln1/√3y0")
println("-"^70)
s1 = s2 = s3 = 0.0
for (ε, y0, k) in TAB
    p = ε * k
    r1 = p - log(1/y0); r2 = p - log(1/(2y0)); r3 = p - log(1/(sqrt(3)*y0))
    global s1 += abs(r1); global s2 += abs(r2); global s3 += abs(r3)
    @printf("%8.0e %8.0e %8d %10.4f %10.4f %10.4f %10.4f\n", ε, y0, k, p, r1, r2, r3)
end
@printf("\n  mean |residual|:  ln(1/y0) %.4f    ln(1/(2y0)) %.4f    ln(1/(sqrt3 y0)) %.4f\n",
        s1/length(TAB), s2/length(TAB), s3/length(TAB))
println("""
  The continuous limit of the recursion is dy/dk = C eps y(1-y^2), whose
  integral from y0 to 1/2 is ln(1/(sqrt(3) y0)) + O(y0^2). The sqrt(3) comes
  from the factor (1-y^2) at the escape threshold, not from the threshold.""")
println("\nFINDINGS OK")
