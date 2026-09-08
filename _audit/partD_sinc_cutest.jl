using TrustRegionRadius, ADNLPModels, NLPModels, LinearAlgebra, Printf
sep(t) = (println(); println("="^72); println(t); println("="^72))
cut(e) = first(replace(sprint(showerror, e), "\n" => " "), 150)

sep("sinc_1d exactly as PartIII-run-configs-v1.jl:120 defines it")
sinc_1d(; x0 = 10.954122) =
    ADNLPModel(p -> (abs(p[1]) < 1e-12 ? 1.0 : sin(p[1]) / p[1]), [float(x0)],
               name = "SINC1D")
ok = try
    sinc_1d(); println("  builds"); true
catch e
    println("  REJECT: ", cut(e)); false
end

println("\nThe branch is the problem. Without it, on the same objective:")
try
    m = ADNLPModel(p -> sin(p[1]) / p[1], [10.954122], name = "SINC1D-nobranch")
    st = tr_solve(m; rule = RDelta(), params = TRParams(tol = 1e-10, max_iterations = 200))
    @printf("  builds and solves: status = %s, x = %.8f, f = %.8f\n",
            st.status, st.solution[1], st.objective)
catch e
    println("  REJECT: ", cut(e))
end

sep("The paper's lambda* = 1/sqrt(1+x^2) at the minimiser, checked")
function sinc_root(k::Int; iters::Int = 60)
    A = (k + 0.5) * π
    x = A - 1 / A
    for _ in 1:iters
        x -= (tan(x) - x) / (sec(x)^2 - 1)
    end
    return x
end
for k in (1, 3, 5)
    x = sinc_root(k)
    # f(x) = sin x / x ; f'' by central differences on the analytic first derivative
    fp(t) = (t * cos(t) - sin(t)) / t^2
    h = 1e-5
    fpp = (fp(x + h) - fp(x - h)) / (2h)
    claim = 1 / sqrt(1 + x^2)
    @printf("k=%d  x_k = %.8f   f''(x_k) = %+.10f   1/sqrt(1+x^2) = %+.10f   match=%s\n",
            k, x, fpp, claim, isapprox(abs(fpp), claim; rtol = 1e-4))
end
println("(k odd gives a MINIMISER only when f''>0; the sign is reported above.)")

sep("D5. CUTEst")
have = try
    @eval using CUTEst
    true
catch e
    println("  CUTEst not loadable: ", cut(e)); false
end
if have
    ps = cutest_problems(min_var = 2, max_var = 200, max_con = 0)
    @printf("  cutest_problems(min_var=2, max_var=200, max_con=0) -> %d problems\n", length(ps))
    @printf("  paper/plan expects 185; difference = %d\n", length(ps) - 185)
else
    println("  not executed: CUTEst unavailable")
end
