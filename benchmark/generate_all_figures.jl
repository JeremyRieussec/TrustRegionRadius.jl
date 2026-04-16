# =============================================================================
# benchmark/generate_all_figures.jl
#
# Master script — reads results from temp_results/, runs all seven experiment
# scripts to produce figures and tables, then archives everything into a
# timestamped folder under results/.
#
# Prerequisites:
#   • Run `julia --project=benchmark benchmark/run_benchmark.jl` first to
#     produce the JLD2 results and experiment_config.toml in temp_results/.
#   • For CUTEst experiments (exp3, exp4, exp6) the MASTSIF environment
#     variable must be set; otherwise ADNLPModels problems are used as fallback.
#
# Usage (from repo root):
#   julia --project=benchmark benchmark/generate_all_figures.jl
#
# To skip specific experiments while iterating:
#   SKIP_EXP3=1 SKIP_EXP4=1 julia --project=benchmark benchmark/generate_all_figures.jl
#
# Archive layout:
#   benchmark/results/exp_YYYY-MM-DD_HH-MM-SS/
#     experiment_config.toml   — parameters used (solver + rules)
#     experiment_summary.md    — human-readable summary
#     jld2/                    — all raw JLD2 result files
#     figures/                 — all generated figures
#     tables/                  — all generated tables
# =============================================================================

using Pkg
Pkg.activate(@__DIR__)

using Dates
using TOML

t_total = time()
println("=" ^ 70)
println("  TrustRegionRadius.jl — generate all figures and tables")
println("=" ^ 70)
println()

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
const BENCH_DIR        = @__DIR__
const TEMP_RESULTS_DIR = joinpath(BENCH_DIR, "temp_results")
const FIGURES_DIR      = joinpath(BENCH_DIR, "figures")
const TABLES_DIR       = joinpath(BENCH_DIR, "tables")
const ARCHIVE_BASE_DIR = joinpath(BENCH_DIR, "results")

# ---------------------------------------------------------------------------
# Validate temp_results/
# ---------------------------------------------------------------------------
jld2_files = filter(f -> endswith(f, ".jld2"), readdir(TEMP_RESULTS_DIR))

if isempty(jld2_files)
    error("No JLD2 files found in temp_results/.\n" *
          "Run benchmark/run_benchmark.jl first, then re-run this script.")
end

config_src = joinpath(TEMP_RESULTS_DIR, "experiment_config.toml")
if !isfile(config_src)
    @warn "experiment_config.toml not found in temp_results/ — archive will be created without it."
end

@info "Found $(length(jld2_files)) result files in temp_results/"

# ---------------------------------------------------------------------------
# Point exp scripts at temp_results/ so they load the current run's data
# ---------------------------------------------------------------------------
ENV["TR_RESULTS_DIR"] = TEMP_RESULTS_DIR

# ---------------------------------------------------------------------------
# Clear figures/ and tables/ so the archive only contains this run's outputs
# ---------------------------------------------------------------------------
mkpath(FIGURES_DIR)
mkpath(TABLES_DIR)
for f in readdir(FIGURES_DIR); rm(joinpath(FIGURES_DIR, f)); end
for f in readdir(TABLES_DIR);  rm(joinpath(TABLES_DIR,  f)); end

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
               joinpath(BENCH_DIR, "exp1_global.jl");    skip_env = "SKIP_EXP1")

# run_experiment("Exp 2 — Radius regime & trajectories",
#                joinpath(BENCH_DIR, "exp2_radius_regime.jl"); skip_env = "SKIP_EXP2")

# run_experiment("Exp 3 — R3 sensitivity to ζ",
#                joinpath(BENCH_DIR, "exp3_zeta.jl");      skip_env = "SKIP_EXP3")

# run_experiment("Exp 4 — R4 sensitivity to μ₀",
#                joinpath(BENCH_DIR, "exp4_mu.jl");        skip_env = "SKIP_EXP4")

# run_experiment("Exp 5 — Ill-conditioned problems",
#                joinpath(BENCH_DIR, "exp5_illcond.jl");   skip_env = "SKIP_EXP5")

# run_experiment("Exp 6 — Sensitivity to Δ₀",
#                joinpath(BENCH_DIR, "exp6_delta0.jl");    skip_env = "SKIP_EXP6")

# run_experiment("Exp 7 — Superlinear convergence",
#                joinpath(BENCH_DIR, "exp7_superlinear.jl"); skip_env = "SKIP_EXP7")

# ---------------------------------------------------------------------------
# Create timestamped archive directory
# ---------------------------------------------------------------------------
timestamp    = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
archive_dir  = joinpath(ARCHIVE_BASE_DIR, "exp_$timestamp")
archive_jld2 = joinpath(archive_dir, "jld2")
archive_figs = joinpath(archive_dir, "figures")
archive_tabs = joinpath(archive_dir, "tables")

mkpath(archive_dir)
mkpath(archive_jld2)
mkpath(archive_figs)
mkpath(archive_tabs)

@info "Archiving to $archive_dir …"

# Copy JLD2 results
for f in jld2_files
    cp(joinpath(TEMP_RESULTS_DIR, f), joinpath(archive_jld2, f))
end

# Copy config
if isfile(config_src)
    cp(config_src, joinpath(archive_dir, "experiment_config.toml"))
end

# Copy figures and tables
for f in readdir(FIGURES_DIR); cp(joinpath(FIGURES_DIR, f), joinpath(archive_figs, f)); end
for f in readdir(TABLES_DIR);  cp(joinpath(TABLES_DIR,  f), joinpath(archive_tabs,  f)); end

# ---------------------------------------------------------------------------
# Write experiment_summary.md
# ---------------------------------------------------------------------------
cfg_dict = isfile(config_src) ? TOML.parsefile(config_src) : Dict{String,Any}()

summary_path = joinpath(archive_dir, "experiment_summary.md")
open(summary_path, "w") do io
    println(io, "# Experiment Summary")
    println(io)
    println(io, "**Archived:** ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
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
        for (k, v) in sort(collect(ps))
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
                 for (k, v) in sort(collect(rule), by = first) if k ∉ ("name", "type")],
                ", ")
            println(io, "- **", name, "** (`", typ, "`): ", params_str)
        end
        println(io)
    end

    # --- Output counts ---
    n_jld2    = length(jld2_files)
    n_figures = length(readdir(archive_figs))
    n_tables  = length(readdir(archive_tabs))

    println(io, "## Outputs")
    println(io)
    println(io, "| Item | Count |")
    println(io, "|------|-------|")
    println(io, "| JLD2 result files | ", n_jld2,    " |")
    println(io, "| Figures           | ", n_figures, " |")
    println(io, "| Tables            | ", n_tables,  " |")
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
# Clear temp_results/ after successful archive
# ---------------------------------------------------------------------------
@info "Clearing temp_results/…"
for f in jld2_files
    rm(joinpath(TEMP_RESULTS_DIR, f))
end
isfile(config_src) && rm(config_src)

# ---------------------------------------------------------------------------
# Final inventory
# ---------------------------------------------------------------------------
elapsed_total = round(time() - t_total, digits = 1)

println()
println("=" ^ 70)
println("  OUTPUT INVENTORY")
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
