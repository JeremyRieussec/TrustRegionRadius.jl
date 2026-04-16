# =============================================================================
# benchmark/exp2_radius_regime.jl
#
# Experiment 2 — Radius regime and trajectory analysis
#
# For a representative set of problems, plots:
#   (a) Trust-region radius Δ_k vs iteration (log scale)
#   (b) Gradient norm ‖g_k‖ vs iteration (log scale)
#   (c) Ratio Δ_k / ‖g_k‖ vs iteration — shows whether the radius stays
#       proportional to gradient (R4 invariant), bounded away (R1/R2), or
#       is capped by ζ‖g‖ (R3)
#   (d) Summability proxy: cumulative sum Σ_{i≤k} Δ_i vs k — a convergent
#       partial sum suggests Σ Δ_k < ∞ (theoretical property of R3/R4)
#
# Produces:
#   figures/exp2_delta_traj_<prob>.pdf
#   figures/exp2_grad_traj_<prob>.pdf
#   figures/exp2_ratio_traj_<prob>.pdf
#   figures/exp2_cumsum_delta_<prob>.pdf
#   figures/exp2_cumsum_summary.pdf   — all problems in one panel
#
# Usage (from repo root):
#   julia --project=benchmark benchmark/exp2_radius_regime.jl
# =============================================================================

using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "load_results.jl"))

using Plots
using PGFPlotsX

using Printf
using LinearAlgebra
using Statistics

pgfplotsx()   # uncomment if PGFPlotsX is installed (better PDF quality)
# gr()

RESULTS_DIR = get(ENV, "TR_RESULTS_DIR", joinpath(@__DIR__, "results"))
FIGURES_DIR = joinpath(@__DIR__, "figures")
mkpath(FIGURES_DIR)

# Positional colour / style cycle — handles any number of rules
const _COLOR_CYCLE_E2 = [
    colorant"#3266AD", colorant"#1D9E75", colorant"#D85A30", colorant"#9933AA",
    colorant"#E6AB02", colorant"#66A61E", colorant"#E7298A", colorant"#A6761D",
]
const _STYLE_CYCLE_E2 = [:solid, :dash, :dot, :dashdot, :solid, :dash, :dot, :dashdot]
rule_color_e2(i::Int) = _COLOR_CYCLE_E2[mod1(i, length(_COLOR_CYCLE_E2))]
rule_style_e2(i::Int) = _STYLE_CYCLE_E2[mod1(i, length(_STYLE_CYCLE_E2))]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@info "Loading results…"
results, prob_names, rule_names = load_all_results(RESULTS_DIR)

# Select representative problems: all-solved, 50 ≤ n ≤ 200, moderate iter
all_solved = problems_all_solved(results, prob_names, rule_names)

function representative_problems(all_solved, results, rule_names; n_select = 6)
    # Score by median iteration count across rules — prefer moderately hard ones
    scored = map(all_solved) do p
        iters = [results[p][r].iterations for r in rule_names if haskey(results[p], r)]
        (p, median(iters))
    end
    filter!(x -> 20 ≤ x[2] ≤ 2000, scored)
    isempty(scored) && return first.(sort(scored, by = x -> x[2]))[1:min(n_select, length(scored))]
    # Pick problems spread across the iteration range
    sort!(scored, by = x -> x[2])
    idx = round.(Int, range(1, length(scored), length = min(n_select, length(scored))))
    return [scored[i][1] for i in idx]
end

rep_probs = representative_problems(all_solved, results, rule_names; n_select = 6)
@info "Representative problems: $rep_probs"

if isempty(rep_probs)
    @warn "No suitable all-solved problems found. Falling back to all_solved[1:6]."
    rep_probs = first(all_solved, 6)
end

# ---------------------------------------------------------------------------
# Per-problem trajectory plots
# ---------------------------------------------------------------------------
for prob in rep_probs

    # (a) Delta trajectory
    pd = plot(; title  = "Radius Δ_k — $prob",
                xlabel = "Iteration k",
                ylabel = "Δ_k  (log scale)",
                yscale = :log10,
                legend = :topright,
                size   = (600, 380))
    for rname in rule_names
        haskey(results[prob], rname) || continue
        entry = results[prob][rname]
        entry.status == :solved || continue
        traj = entry.delta_trajectory
        plot!(pd, 0:length(traj)-1, traj;
              label     = rname,
              color     = rule_color_e2(findfirst(==(rname), rule_names)),
              linestyle = rule_style_e2(findfirst(==(rname), rule_names)),
              linewidth = 1.5)
    end
    savefig(pd, joinpath(FIGURES_DIR, "exp2_delta_traj_$(prob).pdf"))

    # (b) Gradient norm trajectory
    pg = plot(; title  = "Gradient norm ‖g_k‖ — $prob",
                xlabel = "Iteration k",
                ylabel = "‖g_k‖  (log scale)",
                yscale = :log10,
                legend = :topright,
                size   = (600, 380))
    for rname in rule_names
        haskey(results[prob], rname) || continue
        entry = results[prob][rname]
        entry.status == :solved || continue
        traj = entry.grad_norm_trajectory
        plot!(pg, 0:length(traj)-1, max.(traj, 1e-16);
              label     = rname,
              color     = rule_color_e2(findfirst(==(rname), rule_names)),
              linestyle = rule_style_e2(findfirst(==(rname), rule_names)),
              linewidth = 1.5)
    end
    savefig(pg, joinpath(FIGURES_DIR, "exp2_grad_traj_$(prob).pdf"))

    # (c) Ratio Δ_k / ‖g_k‖
    pr = plot(; title  = "Ratio Δ_k / ‖g_k‖ — $prob",
                xlabel = "Iteration k",
                ylabel = "Δ_k / ‖g_k‖  (log scale)",
                yscale = :log10,
                legend = :topright,
                size   = (600, 380))
    for rname in rule_names
        haskey(results[prob], rname) || continue
        entry = results[prob][rname]
        entry.status == :solved || continue
        dt = entry.delta_trajectory
        gt = entry.grad_norm_trajectory
        n  = min(length(dt), length(gt))
        ratio = dt[1:n] ./ max.(gt[1:n], 1e-16)
        plot!(pr, 0:n-1, ratio;
              label     = rname,
              color     = rule_color_e2(findfirst(==(rname), rule_names)),
              linestyle = rule_style_e2(findfirst(==(rname), rule_names)),
              linewidth = 1.5)
    end
    savefig(pr, joinpath(FIGURES_DIR, "exp2_ratio_traj_$(prob).pdf"))

    @info "Saved trajectory plots for $prob"
end

# ---------------------------------------------------------------------------
# (d) Cumulative-sum plots — summability proxy
# ---------------------------------------------------------------------------
# Panel: one subplot per representative problem; 4 rules overlaid
n_probs = length(rep_probs)
nc = min(3, n_probs)
nr = ceil(Int, n_probs / nc)

p_cum = plot(layout = (nr, nc),
             size   = (360 * nc, 280 * nr),
             legend = :topleft,
             title  = "Cumulative Σ Δ_k")

for (idx, prob) in enumerate(rep_probs)
    for rname in rule_names
        haskey(results[prob], rname) || continue
        entry = results[prob][rname]
        entry.status == :solved || continue
        cs = cumsum(entry.delta_trajectory)
        plot!(p_cum, 0:length(cs)-1, cs;
              subplot   = idx,
              label     = rname,
              color     = rule_color_e2(findfirst(==(rname), rule_names)),
              linestyle = rule_style_e2(findfirst(==(rname), rule_names)),
              linewidth = 1.5,
              title     = prob,
              xlabel    = "k",
              ylabel    = "Σ Δ_i")
    end
end
savefig(p_cum, joinpath(FIGURES_DIR, "exp2_cumsum_summary.pdf"))
@info "Saved exp2_cumsum_summary.pdf"

# ---------------------------------------------------------------------------
# Summability statistics
# ---------------------------------------------------------------------------
println("\n" * "="^70)
println("EXP 2 — Summability proxy (final cumulative sum of Δ trajectory)")
println("="^70)
println("(Smaller final Σ Δ_k relative to iterations suggests faster decay)")
println()
println(@sprintf("%-6s  %12s  %12s  %12s", "Rule", "Median Σ Δ_k", "Min Σ Δ_k", "Max Σ Δ_k"))
println("-"^50)

for rname in rule_names
    sums = Float64[]
    for p in all_solved
        haskey(results[p], rname) || continue
        entry = results[p][rname]
        isempty(entry.delta_trajectory) && continue
        push!(sums, sum(entry.delta_trajectory))
    end
    if isempty(sums)
        println(@sprintf("%-6s  %12s  %12s  %12s", rname, "—", "—", "—"))
    else
        println(@sprintf("%-6s  %12.4g  %12.4g  %12.4g",
                         rname, median(sums), minimum(sums), maximum(sums)))
    end
end
println("="^70)
