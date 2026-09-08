# Test the second-order prediction for the ExactMS angle.
#
# s* = -(H + sigma I)^{-1} g, with sigma set by ||s*|| = Delta. Expanding for
# large sigma in the eigenbasis of H, and using sigma ~ ||g||/Delta,
#
#     1 - cos(s*, -g)  ~  (1/2) Var_w(lambda) mu^2,     mu = Delta/||g||,
#
# where w_i = g_i^2/||g||^2 are the weights of g on the eigenvectors and
# Var_w is the weighted variance of the spectrum. The expansion parameter is
# mu, not Delta. This script checks the prediction against the measured angle.
using TrustRegionRadius, ADNLPModels, NLPModels, LinearAlgebra, Printf, Statistics
include(joinpath(@__DIR__, "..", "notebooks", "Saddle", "saddle_problem.jl"))

@printf("%-8s %6s %22s %18s %26s %24s\n", "mu_bar", "n",
        "median(actual/pred)", "IQR", "mu range", "Var_w range")
println("-"^110)
for μm in (0.05, 0.5, 4.0, 128.0)
    nlp = make_nlp(); xs = [collect(float.(X0_DEFAULT))]
    st = tr_solve(nlp; rule = RGradCapped(γ1 = G1, γ2 = G2, γ3 = G3,
                                          μ = min(MU0, μm), μ_max = μm),
                  model = ExactHessian(), subsolver = ExactMS(),
                  params = TRParams(η = ETA1, η1 = ETA1, η2 = ETA2, Δ0 = DELTA0,
                                    tol = TOL, max_iterations = KMAX),
                  trace = true,
                  callback = (_n, _s, s) -> push!(xs, copy(s.solution)))
    ss = st.solver_specific
    cs, Δ = ss[:cos_cauchy_trajectory], ss[:delta_trajectory]
    act = Float64[]; pred = Float64[]; mus = Float64[]; vws = Float64[]
    for i in eachindex(cs)
        x = xs[i]; g = grad(x); H = Symmetric(hess(x)); ng = norm(g)
        ng == 0 && continue
        F = eigen(H)
        w = (transpose(F.vectors) * g) .^ 2 ./ ng^2
        v = sum(w .* F.values .^ 2) - sum(w .* F.values)^2
        μ = Δ[i] / ng
        y = 1 - cs[i]
        (isfinite(y) && y > 0 && v > 0) || continue
        push!(act, y); push!(pred, 0.5 * v * μ^2); push!(mus, μ); push!(vws, v)
    end
    if isempty(act)
        @printf("%-8g %6d  no usable iterations\n", μm, 0); continue
    end
    r = act ./ pred
    @printf("%-8g %6d %22.4f %18s %26s %24s\n", μm, length(act), median(r),
            @sprintf("[%.3f, %.3f]", quantile(r, 0.25), quantile(r, 0.75)),
            @sprintf("[%.3g, %.3g]", minimum(mus), maximum(mus)),
            @sprintf("[%.3g, %.3g]", minimum(vws), maximum(vws)))
end
println("\nPROBE OK")
