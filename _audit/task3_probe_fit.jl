# Fit the exponent of the ExactMS angle, against the right variable.
#
# Within a run of RGradCapped the ratio mu = Delta/||g|| is pinned at mu_bar, so
# no exponent in mu can be fitted from one run. Pooling the runs across mu_bar
# supplies the variation. The predicted law is
#
#     (1 - cos) / Var_w(lambda)  =  (1/2) mu^2,
#
# so a fit of log of the left side on log mu should return slope 2 and intercept
# log10(1/2) = -0.301.
using TrustRegionRadius, ADNLPModels, NLPModels, LinearAlgebra, Printf, Statistics
include(joinpath(@__DIR__, "..", "notebooks", "Saddle", "saddle_problem.jl"))

const MU_SWEEP = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 128.0]

function loglog_fit(x, y)
    idx = [i for i in eachindex(y) if isfinite(x[i]) && isfinite(y[i]) &&
                                      x[i] > 0 && y[i] > 0]
    length(idx) < 5 && return (slope = NaN, inter = NaN, r2 = NaN, n = length(idx))
    X, Y = log10.(x[idx]), log10.(y[idx])
    X̄, Ȳ = mean(X), mean(Y)
    sxx = sum(abs2, X .- X̄)
    sxx == 0 && return (slope = NaN, inter = NaN, r2 = NaN, n = length(idx))
    b = sum((X .- X̄) .* (Y .- Ȳ)) / sxx
    a = Ȳ - b * X̄
    ss = sum(abs2, Y .- Ȳ)
    r2 = ss == 0 ? NaN : 1 - sum(abs2, Y .- (a .+ b .* X)) / ss
    return (slope = b, inter = a, r2 = r2, n = length(idx))
end

MU = Float64[]; YY = Float64[]; VW = Float64[]; DD = Float64[]; RAW = Float64[]
for μm in MU_SWEEP
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
    for i in eachindex(cs)
        x = xs[i]; g = grad(x); H = Symmetric(hess(x)); ng = norm(g)
        ng == 0 && continue
        F = eigen(H)
        w = (transpose(F.vectors) * g) .^ 2 ./ ng^2
        v = sum(w .* F.values .^ 2) - sum(w .* F.values)^2
        y = 1 - cs[i]
        (isfinite(y) && y > 0 && v > 1e-12) || continue
        push!(MU, Δ[i] / ng); push!(YY, y / v); push!(VW, v)
        push!(DD, Δ[i]); push!(RAW, y)
    end
end

@printf("pooled points: %d,  mu range [%.3g, %.3g]\n", length(MU),
        minimum(MU), maximum(MU))

f1 = loglog_fit(DD, RAW)
@printf("\n(1) 1-cos  against Delta      : slope %+.4f  R2 %.4f  n %d\n",
        f1.slope, f1.r2, f1.n)

f2 = loglog_fit(MU, YY)
@printf("(2) (1-cos)/Var_w against mu  : slope %+.4f  R2 %.4f  n %d  intercept %+.4f\n",
        f2.slope, f2.r2, f2.n, f2.inter)
@printf("    predicted slope 2, predicted intercept log10(1/2) = %+.4f\n", log10(0.5))

# restricted to the regime where the expansion is valid
small = [i for i in eachindex(MU) if MU[i] <= 0.5]
f3 = loglog_fit(MU[small], YY[small])
@printf("(3) same, restricted to mu <= 0.5 : slope %+.4f  R2 %.4f  n %d  intercept %+.4f\n",
        f3.slope, f3.r2, f3.n, f3.inter)
println("\nPROBE OK")
