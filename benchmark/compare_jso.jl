
# ============================================================
# benchmark/compare_jso.jl
#
# Performance comparison: TrustRegionRadius.jl update rules
# (R1–R4, Hei family) vs JSOSolvers (trunk, lbfgs, R2, fomo)
# on a selection of unconstrained CUTEst problems.
#
# Uses SolverBenchmark.jl to produce Dolan–Moré performance
# profiles and summary tables.
#
# Prerequisites (add to the benchmark/ Project.toml if separate):
#   CUTEst, NLPModels, JSOSolvers, SolverBenchmark,
#   TrustRegionRadius, Plots, Printf
# ============================================================

using CUTEst
using NLPModels
using JSOSolvers
using SolverBenchmark
using Plots
using Printf
using TrustRegionRadius

# ------------------------------------------------------------
# Problem list — small-to-medium unconstrained CUTEst problems
# ------------------------------------------------------------
const PROBLEM_NAMES = [
    "ROSENBR", "BROWN4",    "PENALTY1",  "ARWHEAD",   "BDQRTIC",
    "CRAGGLVY", "DIXMAANA", "DIXMAANB",  "DQRTIC",    "ENGVAL1",
    "FLETCHCR", "FREUROTH", "GENROSE",   "HIMMELBB",  "JENSMP",
    "LIARWHD",  "MANCINO",  "NONDQUAR",  "PENALTY2",  "TRIDIA",
]

# ------------------------------------------------------------
# Solver wrappers
# Each function signature: (nlp; kwargs...) -> GenericExecutionStats
# ------------------------------------------------------------

# Default TRSolverParams for all TRR variants
const TRR_PARAMS = TRSolverParams(tol=1e-5, max_iterations=10_000)

const SOLVERS = Dict{Symbol, Function}(

    # --- TrustRegionRadius.jl update rules ---
    :TRR_R1 => (nlp; kw...) -> trust_region_radius(nlp;
                    rule   = R1ClassicalUpdate(),
                    params = TRR_PARAMS, kw...),

    :TRR_R2 => (nlp; kw...) -> trust_region_radius(nlp;
                    rule   = R2StepSizeUpdate(),
                    params = TRR_PARAMS, kw...),

    :TRR_R3 => (nlp; kw...) -> trust_region_radius(nlp;
                    rule   = R3DFOLikeUpdate(),
                    params = TRR_PARAMS, kw...),

    :TRR_R4 => (nlp; kw...) -> trust_region_radius(nlp;
                    rule   = R4RelativeGradUpdate(),
                    params = TRR_PARAMS, kw...),

    :TRR_Hei => (nlp; kw...) -> trust_region_radius(nlp;
                    rule   = HeiUpdate(0.1, 0.25, 0.25, 0.25, 4.0, 2.0, 2.0),
                    params = TRR_PARAMS, kw...),

    :TRR_HeiGrad => (nlp; kw...) -> trust_region_radius(nlp;
                    rule   = HeiGradUpdate(0.1, 0.25, 0.25, 0.25, 4.0, 2.0, 2.0),
                    params = TRR_PARAMS, kw...),

    # --- JSOSolvers reference solvers ---
    :trunk  => (nlp; kw...) -> trunk(nlp; kw...),
    :lbfgs  => (nlp; kw...) -> lbfgs(nlp; kw...),
    :R2JSO  => (nlp; kw...) -> R2(nlp; kw...),
    :fomo   => (nlp; kw...) -> fomo(nlp; kw...),
)

# ------------------------------------------------------------
# Run benchmark
# ------------------------------------------------------------

println("Running JSO benchmark on $(length(PROBLEM_NAMES)) CUTEst problems …")
println("Solvers: ", join(string.(keys(SOLVERS)), ", "))

stats = bmark_solvers(
    SOLVERS,
    PROBLEM_NAMES;
    skipif  = prob -> (prob.meta.ncon > 0 || prob.meta.nvar > 1_000),
    max_eval = 50_000,
    atol     = 1e-5,
    rtol     = 1e-5,
)

# ------------------------------------------------------------
# Performance profiles
# ------------------------------------------------------------

results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)

# Profile on number of gradient evaluations
p_grad = performance_profile(
    stats,
    df -> df.neval_grad,
    title = "Performance Profile — gradient evaluations",
)
savefig(p_grad, joinpath(results_dir, "perf_profile_neval_grad.pdf"))
println("Saved: ", joinpath(results_dir, "perf_profile_neval_grad.pdf"))

# Profile on number of objective evaluations
p_obj = performance_profile(
    stats,
    df -> df.neval_obj,
    title = "Performance Profile — objective evaluations",
)
savefig(p_obj, joinpath(results_dir, "perf_profile_neval_obj.pdf"))
println("Saved: ", joinpath(results_dir, "perf_profile_neval_obj.pdf"))

# Profile on elapsed time
p_time = performance_profile(
    stats,
    df -> df.elapsed_time,
    title = "Performance Profile — wall-clock time (s)",
)
savefig(p_time, joinpath(results_dir, "perf_profile_time.pdf"))
println("Saved: ", joinpath(results_dir, "perf_profile_time.pdf"))

# ------------------------------------------------------------
# Summary table
# ------------------------------------------------------------
println("\n", "="^70)
println("Benchmark summary")
println("="^70)
pretty_stats(stats)

# ------------------------------------------------------------
# Per-solver win counts  (simple analysis)
# ------------------------------------------------------------
println("\n", "="^70)
println("First-order convergence counts per solver")
println("="^70)
for (name, df) in stats
    n_solved = sum(df.status .== :first_order)
    @printf("  %-16s : %3d / %3d problems solved\n",
            string(name), n_solved, nrow(df))
end
