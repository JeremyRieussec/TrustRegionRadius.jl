# Section 6 of Task 2: is the solver generic over the element type?
include(joinpath(@__DIR__, "..", "benchmark", "initialisation.jl"))
using Printf

setprecision(BigFloat, 128)
nb = ADNLPModel(p -> sin(p[1]) / p[1], [BigFloat("10.954122")], name = "SINC1D-BF")

println("model eltype: ", eltype(nb.meta.x0))
println("obj  type:    ", typeof(obj(nb, nb.meta.x0)))
println("grad type:    ", eltype(grad(nb, nb.meta.x0)))
println("hess type:    ", eltype(hess(nb, nb.meta.x0)))

p = TRParams{BigFloat}(η = 0.1, η1 = 0.1, η2 = 0.9, Δ0 = 1.0, Δmin = 0.0,
                       Δmax = Inf, tol = BigFloat("1e-30"), max_iterations = 400)
println("TRParams eltype: ", typeof(p).parameters[1])

for (snm, sub) in (("ExactMS", ExactMS()), ("SteihaugCG", SteihaugCG()))
    try
        st = tr_solve(nb; rule = RGradCapped(γ1 = G1, γ2 = G2, γ3 = G3,
                                             μ = 1.0, μ_max = 1.0),
                      model = ExactHessian(), subsolver = sub, params = p, trace = true)
        @printf("%-11s RUNS. status=%s iter=%d eltype(solution)=%s\n",
                snm, st.status, st.iter, eltype(st.solution))
        @printf("            |g| = %s\n", first(string(st.dual_feas), 32))
    catch e
        msg = sprint(showerror, e)
        @printf("%-11s FAILS\n", snm)
        for ln in first(split(msg, '\n'), 4)
            println("            ", first(ln, 150))
        end
    end
end
println("\nBIGFLOAT PROBE OK")
