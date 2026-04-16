# =============================================================================
# benchmark/exp5_illcond.jl
#
# Experiment 5 — Comparison on ill-conditioned problems
#
# Selects a subset of the full benchmark results where the problem is
# "ill-conditioned" — proxied by requiring that at least one mechanism
# took more than ITER_HARD iterations.  This subset isolates cases where
# the radius update strategy matters most.
#
# If CUTEst results are not available, falls back to a hand-crafted set of
# ill-conditioned ADNLPModels problems.
#
# Produces:
#   figures/exp5_perf_profile_illcond.pdf
#   figures/exp5_iter_comparison_illcond.pdf  — bar chart median iter per rule
#   tables/exp5_illcond_summary.txt
#
# Usage (from repo root):
#   julia --project=benchmark benchmark/exp5_illcond.jl
# =============================================================================

using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "load_results.jl"))

using TrustRegionRadius
using LinearAlgebra
using BenchmarkProfiles
using Plots
using PGFPlotsX

using Printf
using Statistics
using ADNLPModels

pgfplotsx()   # uncomment if PGFPlotsX is installed
# gr()

RESULTS_DIR = get(ENV, "TR_RESULTS_DIR", joinpath(@__DIR__, "results"))
FIGURES_DIR = joinpath(@__DIR__, "figures")
TABLES_DIR  = joinpath(@__DIR__, "tables")
mkpath(FIGURES_DIR)
mkpath(TABLES_DIR)

# Positional colour cycle — handles any number of rules
const _COLOR_CYCLE_E5 = [
    colorant"#3266AD", colorant"#1D9E75", colorant"#D85A30", colorant"#9933AA",
    colorant"#E6AB02", colorant"#66A61E", colorant"#E7298A", colorant"#A6761D",
]
rule_color_e5(i::Int) = _COLOR_CYCLE_E5[mod1(i, length(_COLOR_CYCLE_E5))]

# Threshold: problems where the *hardest* mechanism needed ≥ ITER_HARD iters
const ITER_HARD = 100

# ---------------------------------------------------------------------------
# Path A — use existing JLD2 results if available
# ---------------------------------------------------------------------------
results_loaded = false
results        = nothing
rule_names     = nothing
illcond_probs  = String[]

if isdir(RESULTS_DIR) && !isempty(filter(f -> endswith(f, ".jld2"), readdir(RESULTS_DIR)))
    @info "Loading existing benchmark results…"
    results, prob_names, rule_names = load_all_results(RESULTS_DIR)

    all_solved = problems_all_solved(results, prob_names, rule_names)
    illcond_probs = filter(all_solved) do p
        max_iter = maximum(results[p][r].iterations for r in rule_names)
        max_iter >= ITER_HARD
    end
    @info "Ill-conditioned subset (≥ $ITER_HARD iters): $(length(illcond_probs)) problems"
    results_loaded = !isempty(illcond_probs)
end

# ---------------------------------------------------------------------------
# Path B — synthetic ill-conditioned ADNLPModels problems
# ---------------------------------------------------------------------------
if !results_loaded
    @info "No suitable CUTEst results; running ADNLPModels ill-conditioned suite…"

    const PARAMS_E5 = TRSolverParams(η₁=0.1, η₂=0.9, Δ₀=1.0, max_iterations=5_000, tol=1e-5)
    const RULES_E5 = [
        ("R1", R1ClassicalUpdate(0.25, 0.50, 2.0)),
        ("R2", R2StepSizeUpdate(0.25, 0.80, 2.0)),
        ("R3", R3DFOLikeUpdate(0.25, 0.50, 2.0, 1.0)),
        ("R4", R4RelativeGradUpdate(0.25, 2.0,  1.0)),
    ]
    rule_names = first.(RULES_E5)

    illcond_nlps = [
        ("IllQuad_κ1e3",     ADNLPModel(x -> sum(10^((i-1)*3/(10-1)) * x[i]^2 / 2 for i in 1:10), ones(10))),
        ("IllQuad_κ1e5",     ADNLPModel(x -> sum(10^((i-1)*5/(10-1)) * x[i]^2 / 2 for i in 1:10), ones(10))),
        ("IllQuad_κ1e7",     ADNLPModel(x -> sum(10^((i-1)*7/(10-1)) * x[i]^2 / 2 for i in 1:10), ones(10))),
        ("Rosenbrock_2",     ADNLPModel(x -> (1-x[1])^2+100*(x[2]-x[1]^2)^2, [-1.2,1.0])),
        ("Rosenbrock_10",    ADNLPModel(x -> sum(100*(x[i+1]-x[i]^2)^2+(1-x[i])^2 for i in 1:9), fill(-1.2,10))),
        ("Rosenbrock_20",    ADNLPModel(x -> sum(100*(x[i+1]-x[i]^2)^2+(1-x[i])^2 for i in 1:19), fill(-1.2,20))),
        ("BrownBadlyScaled", ADNLPModel(x -> (x[1]-1e6)^2+(x[2]-2e-6)^2+(x[1]*x[2]-2)^2, [1.0,1.0])),
        ("NarrowValley",     ADNLPModel(x -> (x[1]-1)^2+1e4*(x[2]-x[1]^2)^2, [-1.0,1.0])),
        ("Trigonometric_10", ADNLPModel(x -> begin n=10
                                            sum((n-sum(cos.(x))+i*(1-cos(x[i]))-sin(x[i]))^2 for i in 1:n) end,
                                        fill(1/10,10))),
        ("Powell_8",         ADNLPModel(x -> sum((x[4i-3]+10x[4i-2])^2+5*(x[4i-1]-x[4i])^2
                                                +(x[4i-2]-2x[4i-1])^4+10*(x[4i-3]-x[4i])^4 for i in 1:2),
                                        [3.0,-1.0,0.0,1.0,3.0,-1.0,0.0,1.0])),
    ]

    illcond_probs = first.(illcond_nlps)
    nlp_map = Dict(illcond_nlps)

    results = Dict{String, Dict{String, NamedTuple}}()
    for label in illcond_probs
        results[label] = Dict{String, NamedTuple}()
        nlp = nlp_map[label]
        for (rname, rule_proto) in RULES_E5
            try
                rule = deepcopy(rule_proto)
                out  = trust_region_solver(nlp, rule, PARAMS_E5)
                results[label][rname] = (
                    status     = out.status,
                    iterations = out.iterations,
                    f_evals    = out.f_evals,
                    n          = nlp.meta.nvar,
                )
            catch e
                @warn "$rname, $label: $e"
                results[label][rname] = (status=:failure, iterations=typemax(Int), f_evals=typemax(Int), n=0)
            end
        end
    end
end

np = length(illcond_probs)
nr = length(rule_names)
@info "Ill-conditioned subset: $np problems × $nr rules"

# ---------------------------------------------------------------------------
# Metric matrix
# ---------------------------------------------------------------------------
T_iter = fill(Inf, np, nr)
for (i, p) in enumerate(illcond_probs), (j, r) in enumerate(rule_names)
    haskey(results[p], r) || continue
    e = results[p][r]
    e.status == :solved && (T_iter[i, j] = Float64(e.iterations))
end

# ---------------------------------------------------------------------------
# Performance profile
# ---------------------------------------------------------------------------
pp = performance_profile(
    PlotsBackend(),
    T_iter, rule_names;
    title    = "Performance profile — ill-conditioned problems",
    xlabel   = "Performance ratio τ",
    ylabel   = "Fraction of problems solved",
    logscale = true,
    legend   = :bottomright,
)
savefig(pp, joinpath(FIGURES_DIR, "exp5_perf_profile_illcond.pdf"))
@info "Saved exp5_perf_profile_illcond.pdf"

# ---------------------------------------------------------------------------
# Bar chart: median iterations per rule
# ---------------------------------------------------------------------------
med_iters = map(rule_names) do r
    j   = findfirst(==(r), rule_names)
    col = T_iter[:, j]
    finite_col = col[isfinite.(col)]
    isempty(finite_col) ? 0.0 : median(finite_col)
end

pb = bar(rule_names, med_iters;
         xlabel    = "Rule",
         ylabel    = "Median iterations",
         title     = "Ill-conditioned: median iterations",
         legend    = false,
         color     = [rule_color_e5(j) for j in 1:length(rule_names)],
         bar_width = 0.6)
savefig(pb, joinpath(FIGURES_DIR, "exp5_iter_comparison_illcond.pdf"))
@info "Saved exp5_iter_comparison_illcond.pdf"

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
println("\n" * "="^65)
println("EXP 5 — Ill-conditioned problems (n=$np)")
println("="^65)
println(@sprintf("%-6s  %8s  %8s  %8s  %8s  %7s",
                 "Rule", "Solved", "MaxIter", "Failed", "Total", "Rate%"))
println("-"^55)

latex_rows = String[]
for rname in rule_names
    ns  = count(p -> haskey(results[p], rname) && results[p][rname].status == :solved,   illcond_probs)
    nm  = count(p -> haskey(results[p], rname) && results[p][rname].status == :max_iter, illcond_probs)
    nf  = count(p -> haskey(results[p], rname) && results[p][rname].status ∉ (:solved, :max_iter), illcond_probs)
    tot = ns + nm + nf
    rate = tot > 0 ? 100 * ns / tot : 0.0
    println(@sprintf("%-6s  %8d  %8d  %8d  %8d  %7.1f", rname, ns, nm, nf, tot, rate))
    push!(latex_rows, @sprintf("%s & %d & %d & %d & %d & %.1f\\%% \\\\", rname, ns, nm, nf, tot, rate))
end
println("="^65)

table_path = joinpath(TABLES_DIR, "exp5_illcond_summary.txt")
open(table_path, "w") do io
    println(io, "% Exp 5 — Ill-conditioned subset (auto-generated)")
    println(io, "\\begin{tabular}{lrrrrr}")
    println(io, "\\toprule")
    println(io, "Rule & Solved & Max-iter & Failed & Total & Rate (\\%) \\\\")
    println(io, "\\midrule")
    foreach(row -> println(io, row), latex_rows)
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
end
@info "Saved tables/exp5_illcond_summary.txt"
