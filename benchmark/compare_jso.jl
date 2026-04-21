
# ============================================================
# benchmark/compare_jso.jl
#
# Benchmarks TRRSolver (R1, R2, R3, R4, Hei-family) against
# JSOSolvers.trunk / lbfgs / R2 / fomo on a set of CUTEst
# problems, and produces a Dolan-Moré performance profile on
# the number of gradient evaluations.
#
# Requirements
# ------------
#   ] add CUTEst JSOSolvers SolverBenchmark Plots
#
# Usage
# -----
#   julia --project=. benchmark/compare_jso.jl
# ============================================================

using CUTEst
using JSOSolvers
using NLPModels
using SolverBenchmark
using Plots
using Printf
using TrustRegionRadius

# ------------------------------------------------------------
# Problem set -- small unconstrained CUTEst problems
# ------------------------------------------------------------
problem_names = [
    "ROSENBR", "BEALE", "DIXMAANA", "DIXMAANB", "DIXMAANC",
    "DIXMAAND", "DIXMAANE", "DIXMAANF", "DIXMAANG", "DIXMAANH",
    "BROYDN7D", "FREUROTH", "PENALTY1", "GENROSE", "NONDQUAR",
    "POWELLSG", "SROSENBR", "TQUARTIC", "VAREIGVL", "WOODS",
    "ENGVAL1", "EXTROSNB", "QUARTC", "LIARWHD", "DENSCHNA",
]

problems = AbstractNLPModel[]
for name in problem_names
    try
        push!(problems, CUTEstModel(name))
    catch e
        @warn "Skipping $name: $e"
    end
end
@info "Loaded $(length(problems)) problems"

# ------------------------------------------------------------
# Solver dictionary
#
# Each entry is a function  nlp -> GenericExecutionStats
# SolverBenchmark.solve_problems! applies each entry to every
# problem and collects neval_* counters automatically.
# ------------------------------------------------------------

hei_default = (0.25, 0.1, 0.25, 2.0, 4.0, 2.0, 2.0)
common_params(T) = TRSolverParams{T}(tol = 1e-6, max_iterations = 10_000)

solvers = Dict{Symbol, Function}(
    # ------- TRRSolver with each canonical rule -------
    :TRR_R1    => nlp -> trust_region_radius(nlp;
                    rule   = R1ClassicalUpdate(),
                    params = common_params(eltype(nlp.meta.x0))),
    :TRR_R2    => nlp -> trust_region_radius(nlp;
                    rule   = R2StepSizeUpdate(),
                    params = common_params(eltype(nlp.meta.x0))),
    :TRR_R3    => nlp -> trust_region_radius(nlp;
                    rule   = R3DFOLikeUpdate(),
                    params = common_params(eltype(nlp.meta.x0))),
    :TRR_R4    => nlp -> trust_region_radius(nlp;
                    rule   = R4RelativeGradUpdate(),
                    params = common_params(eltype(nlp.meta.x0))),
    :TRR_Hei   => nlp -> trust_region_radius(nlp;
                    rule   = HeiUpdate(hei_default...),
                    params = common_params(eltype(nlp.meta.x0))),
    :TRR_HeiFY => nlp -> trust_region_radius(nlp;
                    rule   = HeiFanYuanUpdate(1.0, hei_default...),
                    params = common_params(eltype(nlp.meta.x0))),

    # ------- JSOSolvers.jl baselines -------
    :trunk  => nlp -> trunk(nlp;  atol = 1e-6, rtol = 1e-6, max_time = 60.0),
    :lbfgs  => nlp -> lbfgs(nlp;  atol = 1e-6, rtol = 1e-6, max_time = 60.0),
    :R2     => nlp -> JSOSolvers.R2(nlp;    atol = 1e-6, rtol = 1e-6, max_time = 60.0),
    :fomo   => nlp -> fomo(nlp;   atol = 1e-6, rtol = 1e-6, max_time = 60.0),
)

# ------------------------------------------------------------
# Run every solver on every problem
# ------------------------------------------------------------
stats = bmark_solvers(solvers, problems)

# ------------------------------------------------------------
# Print per-solver markdown summary
# ------------------------------------------------------------
for (name, df) in stats
    println("\n## ", name)
    pretty_stats(stdout, df[:, [:name, :nvar, :status, :objective,
                                :elapsed_time, :iter, :neval_grad]];
                 tf = markdown_table())
end

# ------------------------------------------------------------
# Dolan-Moré performance profile on neval_grad
# ------------------------------------------------------------
success(df) = df.status .== :first_order
cost(df)    = ifelse.(success(df), df.neval_grad, Inf)

plt = performance_profile(stats, cost;
    legend = :bottomright,
    title  = "Performance profile -- #gradient evaluations",
    xlabel = "within factor of best",
    ylabel = "fraction of problems")

savefig(plt, "benchmark_gradient_profile.pdf")
println("\nSaved profile to benchmark_gradient_profile.pdf")

# ------------------------------------------------------------
# Clean up CUTEst handles
# ------------------------------------------------------------
for p in problems
    finalize(p)
end
