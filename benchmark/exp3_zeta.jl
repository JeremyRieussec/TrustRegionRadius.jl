# =============================================================================
# benchmark/exp3_zeta.jl
#
# Experiment 3 — R3 sensitivity to the parameter ζ
#
# Runs R3DFOLikeUpdate with multiple ζ values on a test suite and compares:
#   • performance profile on iteration count
#   • solve rate vs ζ
#   • median iterations vs ζ
#
# Test suite (in order of preference):
#   1. CUTEst unconstrained problems (if available)
#   2. Expanded ADNLPModels suite (fallback, always works)
#
# Produces:
#   figures/exp3_perf_profile_zeta.pdf
#   figures/exp3_solverate_vs_zeta.pdf
#   tables/exp3_zeta_summary.txt
#
# Usage (from repo root):
#   julia --project=benchmark benchmark/exp3_zeta.jl
# =============================================================================

using Pkg
Pkg.activate(@__DIR__)

using TrustRegionRadius
using LinearAlgebra
using Plots
using PGFPlotsX

using Printf
using Statistics
using BenchmarkProfiles
using ADNLPModels

pgfplotsx()   # uncomment if PGFPlotsX is installed
# gr()

# CUTEst is optional — attempt to load at top level
CUTEST_AVAILABLE = false
try
    using CUTEst
    global CUTEST_AVAILABLE = true
catch
    @info "CUTEst not available; will use ADNLPModels problems"
end

const FIGURES_DIR = joinpath(@__DIR__, "figures")
const TABLES_DIR  = joinpath(@__DIR__, "tables")
mkpath(FIGURES_DIR)
mkpath(TABLES_DIR)

# ---------------------------------------------------------------------------
# Solver parameters (fixed across all ζ)
# ---------------------------------------------------------------------------
const PARAMS = TRSolverParams(
    η₁ = 0.1,
    η₂ = 0.9,
    Δ₀ = 1.0,
    max_iterations = 5_000,
    tol = 1e-5,
)

# ζ values to sweep
const ZETA_VALUES = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

# Colour gradient (blue → red) for different ζ values
zeta_color(i, n) = RGB(i / n, 0.2, 1.0 - i / n)

# ---------------------------------------------------------------------------
# Problem suite
# ---------------------------------------------------------------------------
function build_adnlp_suite()
    return [
        ("Rosenbrock_2",    ADNLPModel(x -> (1-x[1])^2 + 100*(x[2]-x[1]^2)^2,
                                       [-1.2, 1.0])),
        ("Rosenbrock_4",    ADNLPModel(x -> sum(100*(x[i+1]-x[i]^2)^2+(1-x[i])^2 for i in 1:3),
                                       fill(-1.2, 4))),
        ("Rosenbrock_10",   ADNLPModel(x -> sum(100*(x[i+1]-x[i]^2)^2+(1-x[i])^2 for i in 1:9),
                                       fill(-1.2, 10))),
        ("Rosenbrock_20",   ADNLPModel(x -> sum(100*(x[i+1]-x[i]^2)^2+(1-x[i])^2 for i in 1:19),
                                       fill(-1.2, 20))),
        ("Quadratic_5",     ADNLPModel(x -> sum((i*x[i])^2 for i in 1:5), ones(5))),
        ("Quadratic_20",    ADNLPModel(x -> sum((i*x[i])^2 for i in 1:20), ones(20))),
        ("Wood",            ADNLPModel(x -> 100*(x[2]-x[1]^2)^2+(1-x[1])^2
                                           +90*(x[4]-x[3]^2)^2+(1-x[3])^2
                                           +10*(x[2]+x[4]-2)^2+(x[2]-x[4])^2/10,
                                       [-3.0,-1.0,-3.0,-1.0])),
        ("Trigonometric_5", ADNLPModel(x -> begin n=5
                                           sum((n-sum(cos.(x))+i*(1-cos(x[i]))-sin(x[i]))^2
                                               for i in 1:n) end, fill(1/5, 5))),
        ("Powell_8",        ADNLPModel(x -> sum((x[4i-3]+10x[4i-2])^2+5*(x[4i-1]-x[4i])^2
                                               +(x[4i-2]-2x[4i-1])^4+10*(x[4i-3]-x[4i])^4
                                               for i in 1:2),
                                       [3.0,-1.0,0.0,1.0,3.0,-1.0,0.0,1.0])),
        ("BrownBadlyScaled",ADNLPModel(x -> (x[1]-1e6)^2+(x[2]-2e-6)^2+(x[1]*x[2]-2)^2,
                                       [1.0,1.0])),
        ("Beale",           ADNLPModel(x -> (1.5-x[1]*(1-x[2]))^2+(2.25-x[1]*(1-x[2]^2))^2
                                           +(2.625-x[1]*(1-x[2]^3))^2, [1.0,1.0])),
        ("Himmelblau",      ADNLPModel(x -> (x[1]^2+x[2]-11)^2+(x[1]+x[2]^2-7)^2,
                                       [0.0,0.0])),
    ]
end

# Decide which suite to run
if CUTEST_AVAILABLE
    cutest_list = try
        CUTEst.select(min_var = 2, max_var = 200, max_con = 0)
    catch e
        @warn "CUTEst.select failed ($e); falling back to ADNLPModels"
        String[]
    end
else
    cutest_list = String[]
end

if !isempty(cutest_list)
    suite_type = :cutest
    problems   = cutest_list
    all_labels = cutest_list
    @info "Using CUTEst suite: $(length(problems)) problems"
else
    suite_type = :adnlp
    problems   = build_adnlp_suite()
    all_labels = first.(problems)
    @info "Using ADNLPModels suite: $(length(problems)) problems"
end

np = length(all_labels)

# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------
results_zeta = Dict{String, Dict{String, NamedTuple}}()

for ζ in ZETA_VALUES
    zlabel = @sprintf("ζ=%.2g", ζ)
    results_zeta[zlabel] = Dict{String, NamedTuple}()

    if suite_type == :cutest
        for prob_name in problems
            nlp = nothing
            try
                nlp = CUTEstModel(prob_name)
                rule = R3DFOLikeUpdate(0.25, 0.50, 2.0, ζ)
                out  = trust_region_solver(nlp, rule, PARAMS)
                results_zeta[zlabel][prob_name] = (
                    status     = out.status,
                    iterations = out.iterations,
                    f_evals    = out.f_evals,
                )
            catch e
                @warn "ζ=$ζ, $prob_name: $e"
                results_zeta[zlabel][prob_name] = (
                    status=:failure, iterations=typemax(Int), f_evals=typemax(Int)
                )
            finally
                nlp !== nothing && finalize(nlp)
            end
        end
    else
        for (label, nlp) in problems
            try
                rule = R3DFOLikeUpdate(0.25, 0.50, 2.0, ζ)
                out  = trust_region_solver(nlp, rule, PARAMS)
                results_zeta[zlabel][label] = (
                    status     = out.status,
                    iterations = out.iterations,
                    f_evals    = out.f_evals,
                )
            catch e
                @warn "ζ=$ζ, $label: $e"
                results_zeta[zlabel][label] = (
                    status=:failure, iterations=typemax(Int), f_evals=typemax(Int)
                )
            end
        end
    end

    n_solved = count(e -> e.status == :solved, values(results_zeta[zlabel]))
    @info "ζ=$(ζ): $n_solved / $np solved"
end

# ---------------------------------------------------------------------------
# Assemble metric matrix for performance profile
# ---------------------------------------------------------------------------
zeta_labels = [@sprintf("ζ=%.2g", ζ) for ζ in ZETA_VALUES]
nz = length(ZETA_VALUES)

T_iter = fill(Inf, np, nz)
for (j, zlabel) in enumerate(zeta_labels)
    for (i, plabel) in enumerate(all_labels)
        haskey(results_zeta[zlabel], plabel) || continue
        e = results_zeta[zlabel][plabel]
        e.status == :solved && (T_iter[i, j] = Float64(e.iterations))
    end
end

# ---------------------------------------------------------------------------
# Performance profile
# ---------------------------------------------------------------------------
pp = performance_profile(
    PlotsBackend(),
    T_iter, zeta_labels;
    title    = "R3 performance profile — varying ζ",
    xlabel   = "Performance ratio τ",
    ylabel   = "Fraction of problems solved",
    logscale = true,
    legend   = :bottomright,
)
savefig(pp, joinpath(FIGURES_DIR, "exp3_perf_profile_zeta.pdf"))
@info "Saved exp3_perf_profile_zeta.pdf"

# ---------------------------------------------------------------------------
# Solve rate and median iterations vs ζ
# ---------------------------------------------------------------------------
solve_rates  = Float64[]
median_iters = Float64[]

for zlabel in zeta_labels
    its = [Float64(e.iterations) for e in values(results_zeta[zlabel]) if e.status == :solved]
    push!(solve_rates,  isempty(its) ? 0.0 : 100 * length(its) / np)
    push!(median_iters, isempty(its) ? NaN  : median(its))
end

p_rate = plot(log10.(ZETA_VALUES), solve_rates;
              xlabel    = "log₁₀(ζ)",
              ylabel    = "Solve rate (%)",
              title     = "R3: solve rate vs ζ",
              marker    = :circle,
              linewidth = 2,
              legend    = false,
              color     = colorant"#D85A30")
savefig(p_rate, joinpath(FIGURES_DIR, "exp3_solverate_vs_zeta.pdf"))
@info "Saved exp3_solverate_vs_zeta.pdf"

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
println("\n" * "="^60)
println("EXP 3 — R3 sensitivity to ζ")
println("="^60)
println(@sprintf("%-12s  %8s  %12s", "ζ", "Rate%", "Med-iter"))
println("-"^36)
for (j, ζ) in enumerate(ZETA_VALUES)
    println(@sprintf("%-12.4g  %8.1f  %12.1f", ζ, solve_rates[j], median_iters[j]))
end
println("="^60)

table_path = joinpath(TABLES_DIR, "exp3_zeta_summary.txt")
open(table_path, "w") do io
    println(io, "% Exp 3 — R3 sensitivity to zeta (auto-generated)")
    println(io, "\\begin{tabular}{lrr}")
    println(io, "\\toprule")
    println(io, "\\zeta & Rate (\\%) & Median iterations \\\\")
    println(io, "\\midrule")
    for (j, ζ) in enumerate(ZETA_VALUES)
        println(io, @sprintf("%.4g & %.1f & %.1f \\\\", ζ, solve_rates[j], median_iters[j]))
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
end
@info "Saved tables/exp3_zeta_summary.txt"
