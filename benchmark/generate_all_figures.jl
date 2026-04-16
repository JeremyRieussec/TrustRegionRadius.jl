# =============================================================================
# benchmark/generate_all_figures.jl
#
# Generates all figures and tables for a given benchmark run and writes them
# into the run's archive directory alongside the JLD2 results.
#
# Usage (from repo root):
#   julia --project=benchmark benchmark/generate_all_figures.jl <archive_name>
#
# where <archive_name> is the name of a directory under benchmark/results/,
# for example:
#   julia --project=benchmark benchmark/generate_all_figures.jl exp_2026-04-16_00-40-55
#
# The script expects the archive to contain a jld2/ subdirectory with the
# raw result files produced by run_benchmark.jl.
#
# Outputs are written to:
#   benchmark/results/<archive_name>/figures/
#   benchmark/results/<archive_name>/tables/
#   benchmark/results/<archive_name>/experiment_summary.md
#
# To skip specific experiments while iterating:
#   SKIP_EXP3=1 SKIP_EXP4=1 julia --project=benchmark benchmark/generate_all_figures.jl <archive_name>
# =============================================================================

using Pkg
Pkg.activate(@__DIR__)

using Dates
using TOML

include(joinpath(@__DIR__, "load_results.jl"))

using BenchmarkProfiles
using Plots
using PGFPlotsX
using Printf
using LinearAlgebra
using Statistics


t_total = time()
println("=" ^ 70)
println("  TrustRegionRadius.jl — generate all figures and tables")
println("=" ^ 70)
println()

# ---------------------------------------------------------------------------
# Parse argument
# ---------------------------------------------------------------------------
if isempty(ARGS)
    error("""
    Usage: julia --project=benchmark benchmark/generate_all_figures.jl <archive_name>

    <archive_name> is a directory under benchmark/results/, e.g.:
      exp_2026-04-16_00-40-55
    """)
end

archive_name = ARGS[1]

const BENCH_DIR   = @__DIR__
archive_dir  = joinpath(BENCH_DIR, "results", archive_name)
archive_jld2 = joinpath(archive_dir, "jld2")

# ---------------------------------------------------------------------------
# Point exp scripts at the archive's jld2/ directory
# ---------------------------------------------------------------------------
ENV["TR_RESULTS_DIR"] = archive_dir


if !isdir(archive_dir)
    error("Archive directory not found: $archive_dir\n" *
          "Run benchmark/run_benchmark.jl first.")
end

if !isdir(archive_jld2)
    error("No jld2/ subdirectory in $archive_dir\n" *
          "Run benchmark/run_benchmark.jl first.")
end

jld2_files = filter(f -> endswith(f, ".jld2"), readdir(archive_jld2))
if isempty(jld2_files)
    error("No JLD2 files found in $archive_jld2\n" *
          "Run benchmark/run_benchmark.jl first.")
end

@info "Archive: $archive_dir"
@info "Found $(length(jld2_files)) JLD2 result files"

archive_figs = joinpath(archive_dir, "figures")
mkpath(archive_figs)
archive_tabs = joinpath(archive_dir, "tables")
mkpath(archive_tabs)


# ---------------------------------------------------------------------------
# Exp scripts write figures/tables to benchmark/figures/ and benchmark/tables/
# by default. Clear those staging directories before running.
# ---------------------------------------------------------------------------
staging_figs = joinpath(BENCH_DIR, "figures")
staging_tabs = joinpath(BENCH_DIR, "tables")
mkpath(staging_figs)
mkpath(staging_tabs)
for f in readdir(staging_figs); rm(joinpath(staging_figs, f)); end
for f in readdir(staging_tabs); rm(joinpath(staging_tabs, f)); end

# ---------------------------------------------------------------------------
# Helper: run one experiment script
# ---------------------------------------------------------------------------
function run_experiment(name::String, script::String; skip_env::String = "")
    if !isempty(skip_env) && get(ENV, skip_env, "") == "1"
        println("  [SKIP] $name ($(skip_env)=1)")
        return
    end

    println("─" ^ 70)
    println("  Running $name …")
    println("─" ^ 70)

    t0 = time()
    try
        include(script)
        elapsed = round(time() - t0, digits = 1)
        println("\n  ✓ $name completed in $(elapsed)s")
    catch e
        @warn "$name FAILED: $e"
        showerror(stdout, e, catch_backtrace())
        println("\n  ✗ $name FAILED — continuing with next experiment")
    end
    println()
end

# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------
run_experiment("Exp 1 — Global comparison",
               joinpath(BENCH_DIR, "exp1_global.jl");       skip_env = "SKIP_EXP1")

run_experiment("Exp 2 — Radius regime & trajectories",
               joinpath(BENCH_DIR, "exp2_radius_regime.jl"); skip_env = "SKIP_EXP2")

run_experiment("Exp 3 — R3 sensitivity to ζ",
               joinpath(BENCH_DIR, "exp3_zeta.jl");          skip_env = "SKIP_EXP3")

run_experiment("Exp 4 — R4 sensitivity to μ₀",
               joinpath(BENCH_DIR, "exp4_mu.jl");            skip_env = "SKIP_EXP4")

run_experiment("Exp 5 — Ill-conditioned problems",
               joinpath(BENCH_DIR, "exp5_illcond.jl");       skip_env = "SKIP_EXP5")

# run_experiment("Exp 6 — Sensitivity to Δ₀",
#                joinpath(BENCH_DIR, "exp6_delta0.jl");        skip_env = "SKIP_EXP6")

# run_experiment("Exp 7 — Superlinear convergence",
#                joinpath(BENCH_DIR, "exp7_superlinear.jl");   skip_env = "SKIP_EXP7")

# ---------------------------------------------------------------------------
# Copy staging figures/tables into the archive
# ---------------------------------------------------------------------------
@info "Copying outputs into archive…"
for f in readdir(staging_figs)
    cp(joinpath(staging_figs, f), joinpath(archive_figs, f); force = true)
end
for f in readdir(staging_tabs)
    cp(joinpath(staging_tabs, f), joinpath(archive_tabs, f); force = true)
end

# ---------------------------------------------------------------------------
# Write experiment_summary.md
# ---------------------------------------------------------------------------
config_path = joinpath(archive_dir, "experiment_config.toml")
cfg_dict    = isfile(config_path) ? TOML.parsefile(config_path) : Dict{String,Any}()

summary_path = joinpath(archive_dir, "experiment_summary.md")
open(summary_path, "w") do io
    println(io, "# Experiment Summary")
    println(io)
    println(io, "**Archive:** `", archive_name, "`")
    println(io, "**Generated:** ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println(io)

    # --- Solver parameters ---
    if haskey(cfg_dict, "solver_params")
        sp = cfg_dict["solver_params"]
        println(io, "## Solver Parameters")
        println(io)
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        for (k, v) in sort(collect(sp), by = first)
            println(io, "| `", k, "` | ", v, " |")
        end
        println(io)
    end

    # --- Problem selection ---
    if haskey(cfg_dict, "problem_selection")
        ps = cfg_dict["problem_selection"]
        println(io, "## Problem Selection (CUTEst)")
        println(io)
        println(io, "| Criterion | Value |")
        println(io, "|-----------|-------|")
        for (k, v) in sort(collect(ps), by = first)
            println(io, "| `", k, "` | ", v, " |")
        end
        println(io)
    end

    # --- Rules ---
    if haskey(cfg_dict, "rules")
        println(io, "## Radius Update Rules")
        println(io)
        for rule in cfg_dict["rules"]
            name = get(rule, "name", "?")
            typ  = get(rule, "type", "?")
            params_str = join(
                [string(k, " = ", v)
                 for (k, v) in sort(collect(rule), by = first)
                 if k ∉ ("name", "type")],
                ", ")
            println(io, "- **", name, "** (`", typ, "`): ", params_str)
        end
        println(io)
    end

    # --- Output counts ---
    println(io, "## Outputs")
    println(io)
    println(io, "| Item | Count |")
    println(io, "|------|-------|")
    println(io, "| JLD2 result files | ", length(jld2_files),             " |")
    println(io, "| Figures           | ", length(readdir(archive_figs)), " |")
    println(io, "| Tables            | ", length(readdir(archive_tabs)), " |")
    println(io)

    println(io, "### Figures")
    println(io)
    for f in sort(readdir(archive_figs))
        println(io, "- `figures/", f, "`")
    end
    println(io)

    println(io, "### Tables")
    println(io)
    for f in sort(readdir(archive_tabs))
        println(io, "- `tables/", f, "`")
    end
end
@info "Wrote experiment_summary.md"

# ---------------------------------------------------------------------------
# Final inventory
# ---------------------------------------------------------------------------
elapsed_total = round(time() - t_total, digits = 1)

println()
println("=" ^ 70)
println("  OUTPUT INVENTORY  ($archive_name)")
println("=" ^ 70)

for (label, dir) in [("Figures", archive_figs), ("Tables", archive_tabs)]
    files = sort(readdir(dir))
    println()
    println("  $label ($(length(files)) files):")
    for f in files
        sz = round(stat(joinpath(dir, f)).size / 1024, digits = 1)
        println("    $(lpad(sz, 7)) KB  $f")
    end
end

println()
println("=" ^ 70)
println("  Archive: $archive_dir")
println("  All experiments complete.  Total wall time: $(elapsed_total)s")
println("=" ^ 70)
