# Checks that must hold before the proposition of Task 7 is worth stating.
#
#  1. {x = 0} is invariant: the gradient lies along e2 and the Hessian is
#     diagonal, so every CG iterate stays along e2.
#  2. RGradCapped realises Delta_k = mu_k ||g_k|| with mu_k <= mu_bar, which is
#     the only property the proposition consumes.
#  3. tau-anchoring puts a floor under the radius near the saddle.
include(joinpath(@__DIR__, "..", "benchmark", "initialisation.jl"))
using Printf

const P = TRParams(η = 0.1, η1 = 0.1, η2 = 0.9, Δ0 = 1.0, Δmin = 0.0, Δmax = Inf,
                   tol = 1e-12, max_iterations = 400)

function path_of(rule; ε = 1e-2, y0 = 1e-3, x0 = 0.0, params = P, kw...)
    nlp = flat_well(ε; x0 = [x0, y0])
    xs = [Float64[x0, y0]]
    st = tr_solve(nlp; rule = rule, model = ExactHessian(), subsolver = SteihaugCG(),
                  params = params, trace = true,
                  callback = (_n, _s, s) -> push!(xs, copy(s.solution)), kw...)
    return st, xs
end

println("="^78)
println("1. Invariance of {x = 0}")
println("="^78)
for ε in (1e-1, 1e-2, 1e-3), rl in ("RGradCapped", "RDFO", "RDelta")
    r = rl == "RGradCapped" ? RGradCapped(γ1 = 0.25, γ2 = 0.5, γ3 = 2.0, μ = 1.0, μ_max = 1.0) :
        rl == "RDFO"        ? RDFO(γ1 = 0.25, γ2 = 0.5, γ3 = 2.0, ζ = 1.0) :
                              RDelta(γ1 = 0.25, γ2 = 0.5, γ3 = 2.0)
    _, xs = path_of(r; ε = ε)
    mx = maximum(abs(p[1]) for p in xs)
    @printf("  eps=%-8g %-12s max|x_k| over the run = %.3e  %s\n",
            ε, rl, mx, mx == 0 ? "exactly invariant" : "NOT INVARIANT")
end

println("\n", "="^78)
println("2. RGradCapped: Delta_k = mu_k ||g_k||, mu_k <= mu_bar")
println("="^78)
for μ̄ in (0.1, 1.0, 10.0)
    nlp = flat_well(1e-2; x0 = [0.0, 1e-3])
    st = tr_solve(nlp; rule = RGradCapped(γ1 = 0.25, γ2 = 0.5, γ3 = 2.0,
                                          μ = min(1.0, μ̄), μ_max = μ̄),
                  model = ExactHessian(), subsolver = SteihaugCG(),
                  params = P, trace = true)
    ss = st.solver_specific
    Δ, g = ss[:delta_trajectory], ss[:grad_trajectory]
    θ = Δ ./ g
    @printf("  mu_bar=%-6g  max theta_k = Delta/||g|| = %.6f   (must be <= %g)  %s\n",
            μ̄, maximum(θ), μ̄, maximum(θ) <= μ̄ * (1 + 1e-10) ? "holds" : "VIOLATED")
end

println("\n", "="^78)
println("3. tau-anchoring puts a floor under the radius near the saddle")
println("="^78)
for (nm, r) in (("RGradCapped", RGradCapped(γ1 = 0.25, γ2 = 0.5, γ3 = 2.0, μ = 1.0, μ_max = 1.0)),
                ("RGradCappedTau", RGradCappedTau(γ1 = 0.25, γ2 = 0.5, γ3 = 2.0, μ = 1.0, μ_max = 1.0)))
    nlp = flat_well(1e-2; x0 = [0.0, 1e-3])
    st = tr_solve(nlp; rule = r, model = ExactHessian(), subsolver = SteihaugCG(),
                  params = P, trace = true)
    ss = st.solver_specific
    Δ = ss[:delta_trajectory]
    @printf("  %-16s needs_curvature=%-6s  Delta_0=%.3e  min Delta=%.3e  iter=%d status=%s\n",
            nm, string(needs_curvature(r)), Δ[1], minimum(Δ), st.iter, st.status)
end

println("\nPROBE OK")
