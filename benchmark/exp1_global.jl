# =============================================================================
# benchmark/exp1_global.jl
#
# Experiment 1 — Global comparison of R1–R4
#
# Produces:
#   figures/exp1_perf_profile_iter.pdf   — performance profile on iteration count
#   figures/exp1_perf_profile_fevals.pdf — performance profile on f-evaluations
#   figures/exp1_data_profile.pdf        — data profile (τ = 1e-1 accuracy)
#   tables/exp1_success_rate.txt         — success-rate table (LaTeX-ready)
#
# Usage (from repo root):
#   julia --project=benchmark benchmark/exp1_global.jl
# =============================================================================

using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "load_results.jl"))

using BenchmarkProfiles
using Plots
using PGFPlotsX
using Printf
using LinearAlgebra
using Statistics

pgfplotsx()   # comment out and use gr() / pyplot() if PGFPlotsX is not installed
# gr()

# ---------------------------------------------------------------------------
# I/O directories
# ---------------------------------------------------------------------------
# RESULTS_DIR can be overridden by TR_RESULTS_DIR env var so that
# generate_all_figures.jl can point exp scripts at temp_results/.
RESULTS_DIR = get(ENV, "TR_RESULTS_DIR", joinpath(@__DIR__, "results"))
FIGURES_DIR = joinpath(@__DIR__, "figures")
TABLES_DIR  = joinpath(@__DIR__, "tables")
mkpath(FIGURES_DIR)
mkpath(TABLES_DIR)

# ---------------------------------------------------------------------------
# Colour / style cycle — supports any number of rules
# ---------------------------------------------------------------------------
# Assign colours and line styles positionally so that rules added to config.jl
# (R4-Alt, Hei, HeiG, HFY, …) automatically get a distinct appearance.
const _COLOR_CYCLE = [
    colorant"#3266AD", colorant"#1D9E75", colorant"#D85A30", colorant"#9933AA",
    colorant"#E6AB02", colorant"#66A61E", colorant"#E7298A", colorant"#A6761D",
]
const _STYLE_CYCLE = [:solid, :dash, :dot, :dashdot, :solid, :dash, :dot, :dashdot]
rule_color(i::Int) = _COLOR_CYCLE[mod1(i, length(_COLOR_CYCLE))]
rule_style(i::Int) = _STYLE_CYCLE[mod1(i, length(_STYLE_CYCLE))]

# ---------------------------------------------------------------------------
# Load benchmark data
# ---------------------------------------------------------------------------
@info "Loading results…"
results, prob_names, rule_names = load_all_results(RESULTS_DIR)

# Work only on problems where every mechanism has a result (any status)
common_probs = filter(prob_names) do p
    all(r -> haskey(results[p], r), rule_names)
end
@info "Problems with results for all mechanisms: $(length(common_probs))"

# ---------------------------------------------------------------------------
# Build metric matrices  (rows = problems, cols = solvers)
# Inf → failure / max_iter
# ---------------------------------------------------------------------------
function metric_matrix(metric_fn)
    m = length(common_probs)
    s = length(rule_names)
    M = fill(Inf, m, s)
    for (i, p) in enumerate(common_probs), (j, r) in enumerate(rule_names)
        entry = results[p][r]
        if entry.status == :solved
            M[i, j] = metric_fn(entry)
        end
    end
    return M
end

T_iter   = metric_matrix(e -> Float64(e.iterations))
T_fevals = metric_matrix(e -> Float64(e.f_evals))

# ---------------------------------------------------------------------------
# Performance profiles
# ---------------------------------------------------------------------------
function make_perf_profile(T, metric_label, fname)
    # Pass colours through `palette` so they go through the normal Plots
    # pipeline — avoid post-hoc series_list mutation which confuses PGFPlotsX.
    rule_palette = [rule_color(j) for j in 1:length(rule_names)]
    p = performance_profile(
        PlotsBackend(),
        T, rule_names;
        title    = "Performance profile ($metric_label)",
        xlabel   = "Performance ratio τ",
        ylabel   = "Fraction of problems solved",
        logscale = true,
        legend   = :bottomright,
        palette  = rule_palette,
        linewidth = 2,
    )
    savefig(p, joinpath(FIGURES_DIR, fname))
    @info "Saved $fname"
    return p
end

p_iter   = make_perf_profile(T_iter,   "iterations",         "exp1_perf_profile_iter.pdf")
p_fevals = make_perf_profile(T_fevals, "function evaluations", "exp1_perf_profile_fevals.pdf")

# ---------------------------------------------------------------------------
# Data profile
# ---------------------------------------------------------------------------
# H[k, p, s] = objective value after the k-th function evaluation for
# problem p, solver s.  We use obj_trajectory (one entry per iteration,
# starting from k=0) as a proxy for f-eval count.
# N[p] = n(p) + 1  — simplex-gradient scaling (Moré–Wild convention).
# Unsolved / missing entries remain Inf, which data_profile treats as failure.

np_c = length(common_probs)
ns_c = length(rule_names)

# Maximum trajectory length across all (problem, solver) pairs
max_K = maximum(
    length(results[p][r].obj_trajectory)
    for p in common_probs for r in rule_names
    if haskey(results[p], r) && !isempty(results[p][r].obj_trajectory);
    init = 2,
)

H = fill(Inf, max_K, np_c, ns_c)
for (i, p) in enumerate(common_probs), (j, r) in enumerate(rule_names)
    haskey(results[p], r) || continue
    entry = results[p][r]
    traj  = entry.obj_trajectory
    isempty(traj) && continue
    nt = length(traj)
    for k in 1:min(nt, max_K)
        H[k, i, j] = traj[k]
    end
    # Pad with final value for solved problems (already at optimum)
    if entry.status == :solved && nt < max_K
        H[nt+1:max_K, i, j] .= traj[end]
    end
end

# N[p] = n + 1 (simplex-gradient scaling); fall back to 2 if n unknown
N_scale = Float64[max(results[p][rule_names[1]].n + 1, 2) for p in common_probs]

rule_palette = [rule_color(j) for j in 1:length(rule_names)]
p_data = data_profile(
    PlotsBackend(),
    H,
    N_scale,
    AbstractString[r for r in rule_names];
    τ          = 1.0e-3,
    operations = "function evaluations",
    title      = "Data profile (τ = 1e-3)",
    palette    = rule_palette,
    linewidth  = 2,
)
savefig(p_data, joinpath(FIGURES_DIR, "exp1_data_profile.pdf"))
@info "Saved exp1_data_profile.pdf"

# ---------------------------------------------------------------------------
# Success-rate table
# ---------------------------------------------------------------------------
println("\n" * "="^70)
println("EXP 1 — Success rates")
println("="^70)

header  = @sprintf("%-6s  %8s  %8s  %8s  %8s  %7s",
                   "Rule", "Solved", "MaxIter", "Failed", "Total", "Rate%")
divider = "-"^55
println(header)
println(divider)

latex_rows = String[]
for rname in rule_names
    ns  = count(p -> haskey(results[p], rname) && results[p][rname].status == :solved,   common_probs)
    nm  = count(p -> haskey(results[p], rname) && results[p][rname].status == :max_iter, common_probs)
    nf  = count(p -> haskey(results[p], rname) && results[p][rname].status ∉ (:solved, :max_iter), common_probs)
    tot = ns + nm + nf
    rate = tot > 0 ? 100 * ns / tot : 0.0

    println(@sprintf("%-6s  %8d  %8d  %8d  %8d  %7.1f", rname, ns, nm, nf, tot, rate))

    # LaTeX row
    push!(latex_rows,
          @sprintf("%s & %d & %d & %d & %d & %.1f\\%%  \\\\",
                   rname, ns, nm, nf, tot, rate))
end
println("="^70)

# Write LaTeX table
table_path = joinpath(TABLES_DIR, "exp1_success_rate.txt")
open(table_path, "w") do io
    println(io, "% Exp 1 — Success rate table (auto-generated by exp1_global.jl)")
    println(io, "% Columns: Rule | Solved | MaxIter | Failed | Total | Rate%")
    println(io, "\\begin{tabular}{lrrrrr}")
    println(io, "\\toprule")
    println(io, "Rule & Solved & Max-iter & Failed & Total & Rate (\\%) \\\\")
    println(io, "\\midrule")
    for row in latex_rows
        println(io, row)
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
end
@info "Saved tables/exp1_success_rate.txt"

# ---------------------------------------------------------------------------
# Median-iteration / median-fevals summary
# ---------------------------------------------------------------------------
println("\nMedian metrics (solved problems only):")
println(@sprintf("%-6s  %10s  %10s", "Rule", "Med-iter", "Med-fevals"))
println("-"^32)
for (j, rname) in enumerate(rule_names)
    solved_iter   = T_iter[isfinite.(T_iter[:, j]),   j]
    solved_fevals = T_fevals[isfinite.(T_fevals[:, j]), j]
    med_i = isempty(solved_iter)   ? NaN : median(solved_iter)
    med_f = isempty(solved_fevals) ? NaN : median(solved_fevals)
    println(@sprintf("%-6s  %10.1f  %10.1f", rname, med_i, med_f))
end
