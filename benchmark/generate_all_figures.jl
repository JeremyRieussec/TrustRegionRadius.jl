# =============================================================================
# benchmark/generate_all_figures.jl
#
# Master script — runs all seven experiment scripts in sequence and
# collects the list of generated figures and tables.
#
# Prerequisites:
#   • Run `julia --project=benchmark benchmark/run_benchmark.jl` first
#     (or open benchmark_notebook.jl in Pluto) to produce the JLD2 results.
#   • For CUTEst experiments (3, 4, 6) the MASTSIF environment variable
#     must be set; otherwise ADNLPModels problems are used as a fallback.
#
# Usage (from repo root):
#   julia --project=benchmark benchmark/generate_all_figures.jl
#
# To skip specific experiments (e.g. while iterating), set env vars:
#   SKIP_EXP3=1 SKIP_EXP4=1 julia --project=benchmark benchmark/generate_all_figures.jl
# =============================================================================

using Pkg
Pkg.activate(@__DIR__)

t_total = time()
println("=" ^ 70)
println("  TrustRegionRadius.jl — generate all figures and tables")
println("=" ^ 70)
println()

# ---------------------------------------------------------------------------
# Helper
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
# Experiment order
# ---------------------------------------------------------------------------
EXP_DIR = @__DIR__

run_experiment("Exp 1 — Global comparison",
               joinpath(EXP_DIR, "exp1_global.jl");
               skip_env = "SKIP_EXP1")

run_experiment("Exp 2 — Radius regime & trajectories",
               joinpath(EXP_DIR, "exp2_radius_regime.jl");
               skip_env = "SKIP_EXP2")

run_experiment("Exp 3 — R3 sensitivity to ζ",
               joinpath(EXP_DIR, "exp3_zeta.jl");
               skip_env = "SKIP_EXP3")

run_experiment("Exp 4 — R4 sensitivity to μ₀",
               joinpath(EXP_DIR, "exp4_mu.jl");
               skip_env = "SKIP_EXP4")

run_experiment("Exp 5 — Ill-conditioned problems",
               joinpath(EXP_DIR, "exp5_illcond.jl");
               skip_env = "SKIP_EXP5")

run_experiment("Exp 6 — Sensitivity to Δ₀",
               joinpath(EXP_DIR, "exp6_delta0.jl");
               skip_env = "SKIP_EXP6")

run_experiment("Exp 7 — Superlinear convergence",
               joinpath(EXP_DIR, "exp7_superlinear.jl");
               skip_env = "SKIP_EXP7")

# ---------------------------------------------------------------------------
# Inventory of outputs
# ---------------------------------------------------------------------------
FIGURES_DIR = joinpath(EXP_DIR, "figures")
TABLES_DIR  = joinpath(EXP_DIR, "tables")

println("=" ^ 70)
println("  OUTPUT INVENTORY")
println("=" ^ 70)

for (label, dir) in [("Figures", FIGURES_DIR), ("Tables", TABLES_DIR)]
    if isdir(dir)
        files = sort(readdir(dir))
        println("\n  $label ($(length(files)) files in $dir):")
        for f in files
            path = joinpath(dir, f)
            sz   = round(stat(path).size / 1024, digits = 1)
            println("    $(lpad(sz, 7)) KB  $f")
        end
    else
        println("\n  $label directory not found: $dir")
    end
end

elapsed_total = round(time() - t_total, digits = 1)
println()
println("=" ^ 70)
println("  All experiments complete.  Total wall time: $(elapsed_total)s")
println("=" ^ 70)
