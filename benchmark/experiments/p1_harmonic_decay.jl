using TrustRegionRadius, ADNLPModels, Printf

nlp = ADNLPModel(x -> exp(-x[1]), [0.0])
stats = tr_solve(nlp;
    rule      = RGradCapped(μ = 1.0, μ_max = 1.0),
    model     = ExactHessian(),
    subsolver = SteihaugCG(),
    params    = TRParams(tol = 1e-12, max_iterations = 200_000),
    trace     = true)

Δ = stats.solver_specific[:delta_trajectory]
n = length(Δ)
@printf("status = %s   iterations = %d\n", stats.status, n)
for k in (1, 10, 100, 1_000, 10_000, 100_000, n-1)
    k < 1 && continue
    k > n && continue
    @printf("k = %-8d  Delta_k = %.6e   k*Delta_k = %.6f\n", k, Δ[k], k*Δ[k])
end
S1 = sum(Δ); S2 = sum(abs2, Δ)
@printf("sum Delta_k    = %.4f  over %d iterations\n", S1, n)
@printf("sum Delta_k^2  = %.6f\n", S2)
# partial sums of Delta, to show the divergence is logarithmic
for m in (100, 1_000, 10_000, 100_000)
    m > n && continue
    @printf("  sum_{k<=%-7d} Delta_k = %8.4f    ln(m) = %8.4f\n", m, sum(Δ[1:m]), log(m))
end
