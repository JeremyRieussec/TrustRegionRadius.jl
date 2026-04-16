# =============================================================================
# benchmark/exp7_superlinear.jl
#
# Experiment 7 — Superlinear / quadratic convergence verification
#
# Estimates the asymptotic convergence order q by fitting
#
#     log ‖g_{k+1}‖ ≈ q · log ‖g_k‖ + c
#
# over the last CONV_WINDOW iterations of each solved run.
# q > 1 indicates superlinear convergence.
#
# Produces:
#   figures/exp7_conv_order_boxplot.pdf
#   figures/exp7_conv_order_scatter.pdf
#   tables/exp7_conv_order_summary.txt
#
# Usage (from repo root):
#   julia --project=benchmark benchmark/exp7_superlinear.jl
# =============================================================================

using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "load_results.jl"))

using TrustRegionRadius
using LinearAlgebra
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
const _COLOR_CYCLE_E7 = [
    colorant"#3266AD", colorant"#1D9E75", colorant"#D85A30", colorant"#9933AA",
    colorant"#E6AB02", colorant"#66A61E", colorant"#E7298A", colorant"#A6761D",
]
rule_color_e7(i::Int) = _COLOR_CYCLE_E7[mod1(i, length(_COLOR_CYCLE_E7))]

const CONV_WINDOW = 15
const GNORM_FLOOR = 1e-14

# ---------------------------------------------------------------------------
# Load or synthesise results
# ---------------------------------------------------------------------------
results_available = false
results    = nothing
prob_names = nothing
rule_names = nothing

if isdir(RESULTS_DIR) && !isempty(filter(f -> endswith(f, ".jld2"), readdir(RESULTS_DIR)))
    @info "Loading existing benchmark results…"
    results, prob_names, rule_names = load_all_results(RESULTS_DIR)
    all_solved = problems_all_solved(results, prob_names, rule_names)
    results_available = length(all_solved) >= 5
end

if !results_available
    @info "No CUTEst results; running ADNLPModels suite for Exp 7…"

    const PARAMS_E7 = TRSolverParams(η₁=0.1, η₂=0.9, Δ₀=1.0,
                                     max_iterations=5_000, tol=1e-8)
    const RULES_E7 = [
        ("R1", R1ClassicalUpdate(0.25, 0.50, 2.0)),
        ("R2", R2StepSizeUpdate(0.25, 0.80, 2.0)),
        ("R3", R3DFOLikeUpdate(0.25, 0.50, 2.0, 1.0)),
        ("R4", R4RelativeGradUpdate(0.25, 2.0,  1.0)),
    ]
    rule_names = first.(RULES_E7)

    nlps = [
        ("Rosenbrock_2",     ADNLPModel(x -> (1-x[1])^2+100*(x[2]-x[1]^2)^2, [-1.2,1.0])),
        ("Rosenbrock_4",     ADNLPModel(x -> sum(100*(x[i+1]-x[i]^2)^2+(1-x[i])^2 for i in 1:3), fill(-1.2,4))),
        ("Rosenbrock_10",    ADNLPModel(x -> sum(100*(x[i+1]-x[i]^2)^2+(1-x[i])^2 for i in 1:9), fill(-1.2,10))),
        ("Rosenbrock_20",    ADNLPModel(x -> sum(100*(x[i+1]-x[i]^2)^2+(1-x[i])^2 for i in 1:19), fill(-1.2,20))),
        ("Quadratic_5",      ADNLPModel(x -> sum((i*x[i])^2 for i in 1:5), ones(5))),
        ("Quadratic_20",     ADNLPModel(x -> sum((i*x[i])^2 for i in 1:20), ones(20))),
        ("Wood",             ADNLPModel(x -> 100*(x[2]-x[1]^2)^2+(1-x[1])^2+90*(x[4]-x[3]^2)^2
                                           +(1-x[3])^2+10*(x[2]+x[4]-2)^2+(x[2]-x[4])^2/10,
                                        [-3.0,-1.0,-3.0,-1.0])),
        ("Powell_8",         ADNLPModel(x -> sum((x[4i-3]+10x[4i-2])^2+5*(x[4i-1]-x[4i])^2
                                                +(x[4i-2]-2x[4i-1])^4+10*(x[4i-3]-x[4i])^4 for i in 1:2),
                                        [3.0,-1.0,0.0,1.0,3.0,-1.0,0.0,1.0])),
        ("Beale",            ADNLPModel(x -> (1.5-x[1]*(1-x[2]))^2+(2.25-x[1]*(1-x[2]^2))^2
                                           +(2.625-x[1]*(1-x[2]^3))^2, [1.0,1.0])),
        ("Himmelblau",       ADNLPModel(x -> (x[1]^2+x[2]-11)^2+(x[1]+x[2]^2-7)^2, [0.0,0.0])),
        ("BrownBadlyScaled", ADNLPModel(x -> (x[1]-1e6)^2+(x[2]-2e-6)^2+(x[1]*x[2]-2)^2,[1.0,1.0])),
        ("Trigonometric_5",  ADNLPModel(x -> begin n=5
                                            sum((n-sum(cos.(x))+i*(1-cos(x[i]))-sin(x[i]))^2 for i in 1:n) end,
                                        fill(1/5,5))),
    ]
    prob_names = first.(nlps)
    nlp_map    = Dict(nlps)

    results = Dict{String, Dict{String, NamedTuple}}()
    for label in prob_names
        results[label] = Dict{String, NamedTuple}()
        nlp = nlp_map[label]
        for (rname, rule_proto) in RULES_E7
            try
                rule = deepcopy(rule_proto)
                out  = trust_region_solver(nlp, rule, PARAMS_E7)
                results[label][rname] = (
                    status               = out.status,
                    iterations           = out.iterations,
                    grad_norm_trajectory = out.grad_norm_trajectory,
                    n                    = nlp.meta.nvar,
                )
            catch e
                @warn "$rname, $label: $e"
                results[label][rname] = (
                    status=:failure, iterations=0,
                    grad_norm_trajectory=Float64[], n=nlp.meta.nvar,
                )
            end
        end
    end
    all_solved = filter(p -> all(r -> haskey(results[p], r) && results[p][r].status == :solved,
                                 rule_names), prob_names)
end

# ---------------------------------------------------------------------------
# Convergence order estimation
# ---------------------------------------------------------------------------
"""
    estimate_conv_order(traj, window) -> Float64

Fit log(‖g_{k+1}‖) = q·log(‖g_k‖) + c over the last `window` steps.
Returns q, or NaN if the fit is unreliable.
"""
function estimate_conv_order(traj::Vector{Float64}, window::Int)
    n = length(traj)
    n < window + 2 && return NaN
    lo   = max(1, n - window)
    hi   = n - 1
    xs   = log.(max.(traj[lo:hi],     GNORM_FLOOR))
    ys   = log.(max.(traj[lo+1:hi+1], GNORM_FLOOR))
    mask = (traj[lo:hi] .> 1e2 * GNORM_FLOOR) .& (traj[lo+1:hi+1] .< traj[lo:hi])
    sum(mask) < 3 && return NaN
    xs, ys = xs[mask], ys[mask]
    x̄ = mean(xs); ȳ = mean(ys)
    q = sum((xs .- x̄) .* (ys .- ȳ)) / (sum((xs .- x̄).^2) + 1e-30)
    return q
end

q_data = Dict(r => Float64[] for r in rule_names)
n_data = Dict(r => Int[]     for r in rule_names)

for p in all_solved, r in rule_names
    haskey(results[p], r) || continue
    entry = results[p][r]
    entry.status == :solved || continue
    isempty(entry.grad_norm_trajectory) && continue
    q = estimate_conv_order(entry.grad_norm_trajectory, CONV_WINDOW)
    isnan(q) && continue
    push!(q_data[r], q)
    push!(n_data[r], entry.n)
end

# ---------------------------------------------------------------------------
# Box plot (scatter + median line)
# ---------------------------------------------------------------------------
p_box = plot(; xlabel="Rule", ylabel="Convergence order q",
               title="Convergence order distribution", legend=:topright)

for (j, rname) in enumerate(rule_names)
    qs = q_data[rname]
    isempty(qs) && continue
    jitter = 0.15 * (rand(length(qs)) .- 0.5)
    scatter!(p_box, fill(j, length(qs)) .+ jitter, qs;
             color=rule_color_e7(j), alpha=0.5, ms=4, label=rname)
    plot!(p_box, [j-0.3, j+0.3], fill(median(qs), 2);
          color=rule_color_e7(j), linewidth=2.5, label="")
end
hline!(p_box, [1.0]; linestyle=:dash, color=:black, label="q=1 (linear)")
xticks!(p_box, 1:length(rule_names), rule_names)
savefig(p_box, joinpath(FIGURES_DIR, "exp7_conv_order_boxplot.pdf"))
@info "Saved exp7_conv_order_boxplot.pdf"

# ---------------------------------------------------------------------------
# Scatter: q vs problem size n
# ---------------------------------------------------------------------------
p_sc = plot(; xlabel="n", ylabel="q", title="Convergence order vs problem size",
              legend=:outerright)
for rname in rule_names
    qs = q_data[rname]; ns = n_data[rname]
    isempty(qs) && continue
    scatter!(p_sc, ns, qs; label=rname,
             color=rule_color_e7(findfirst(==(rname), rule_names)), alpha=0.6, ms=4)
end
hline!(p_sc, [1.0]; linestyle=:dash, color=:black, label="q=1")
savefig(p_sc, joinpath(FIGURES_DIR, "exp7_conv_order_scatter.pdf"))
@info "Saved exp7_conv_order_scatter.pdf"

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
println("\n" * "="^70)
println("EXP 7 — Convergence order q  (window = $CONV_WINDOW)")
println("="^70)
println(@sprintf("%-6s  %8s  %8s  %8s  %8s  %7s",
                 "Rule","Count","Median-q","Mean-q","Std-q","q>1 (%)"))
println("-"^55)

latex_rows = String[]
for rname in rule_names
    qs = q_data[rname]
    if isempty(qs)
        println(@sprintf("%-6s  %8s  %8s  %8s  %8s  %7s", rname,"—","—","—","—","—"))
        continue
    end
    med_q = median(qs); mea_q = mean(qs); std_q = std(qs)
    frac  = 100 * count(>(1.0), qs) / length(qs)
    println(@sprintf("%-6s  %8d  %8.3f  %8.3f  %8.3f  %7.1f",
                     rname, length(qs), med_q, mea_q, std_q, frac))
    push!(latex_rows,
          @sprintf("%s & %d & %.3f & %.3f & %.3f & %.1f\\%% \\\\",
                   rname, length(qs), med_q, mea_q, std_q, frac))
end
println("="^70)

table_path = joinpath(TABLES_DIR, "exp7_conv_order_summary.txt")
open(table_path, "w") do io
    println(io, "% Exp 7 — Convergence order (auto-generated, window=$CONV_WINDOW)")
    println(io, "\\begin{tabular}{lrrrrr}")
    println(io, "\\toprule")
    println(io, "Rule & n & Median q & Mean q & Std q & q>1 (\\%) \\\\")
    println(io, "\\midrule")
    foreach(row -> println(io, row), latex_rows)
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
end
@info "Saved tables/exp7_conv_order_summary.txt"
