# =============================================================================
# benchmark/exp4_mu.jl
#
# Experiment 4 — R4 sensitivity to the initial multiplier μ₀
#
# R4 sets Δ₀ = μ₀ · ‖g₀‖.  This experiment sweeps μ₀ and measures:
#   • solve rate and median iteration count vs μ₀
#   • final μ distribution (drift from μ₀)
#
# Produces:
#   figures/exp4_perf_profile_mu.pdf
#   figures/exp4_solverate_vs_mu.pdf
#   figures/exp4_mu_drift.pdf
#   tables/exp4_mu_summary.txt
#
# Usage (from repo root):
#   julia --project=benchmark benchmark/exp4_mu.jl
# =============================================================================

using Pkg
Pkg.activate(@__DIR__)

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

const PARAMS = TRSolverParams(
    η₁ = 0.1,
    η₂ = 0.9,
    Δ₀ = 1.0,         # ignored by R4 (uses μ · ‖g₀‖ instead)
    max_iterations = 5_000,
    tol = 1e-5,
)

const MU_VALUES = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

# ---------------------------------------------------------------------------
# Bounded R4 variant — caps μ at μ_max after each update
# ---------------------------------------------------------------------------
mutable struct R4BoundedUpdate <: AbstractRadiusUpdate
    base  ::R4RelativeGradUpdate
    μ_max ::Float64
end

function TrustRegionRadius.update_radius!(rule::R4BoundedUpdate,
        Δ::Float64, ρ::Float64, η₁::Float64, η₂::Float64,
        s_norm::Float64, g_norm_old::Float64, g_norm_new::Float64)
    Δ_new = update_radius!(rule.base, Δ, ρ, η₁, η₂, s_norm, g_norm_old, g_norm_new)
    rule.base.μ = min(rule.base.μ, rule.μ_max)
    return min(Δ_new, rule.μ_max * g_norm_new)
end

TrustRegionRadius.initial_radius(rule::R4BoundedUpdate, Δ₀::Float64, g_norm::Float64) =
    rule.base.μ * g_norm

# ---------------------------------------------------------------------------
# Problem suite
# ---------------------------------------------------------------------------
function build_adnlp_suite()
    return [
        ("Rosenbrock_2",    ADNLPModel(x -> (1-x[1])^2+100*(x[2]-x[1]^2)^2, [-1.2,1.0])),
        ("Rosenbrock_4",    ADNLPModel(x -> sum(100*(x[i+1]-x[i]^2)^2+(1-x[i])^2 for i in 1:3), fill(-1.2,4))),
        ("Rosenbrock_10",   ADNLPModel(x -> sum(100*(x[i+1]-x[i]^2)^2+(1-x[i])^2 for i in 1:9), fill(-1.2,10))),
        ("Rosenbrock_20",   ADNLPModel(x -> sum(100*(x[i+1]-x[i]^2)^2+(1-x[i])^2 for i in 1:19), fill(-1.2,20))),
        ("Quadratic_5",     ADNLPModel(x -> sum((i*x[i])^2 for i in 1:5), ones(5))),
        ("Quadratic_20",    ADNLPModel(x -> sum((i*x[i])^2 for i in 1:20), ones(20))),
        ("Wood",            ADNLPModel(x -> 100*(x[2]-x[1]^2)^2+(1-x[1])^2+90*(x[4]-x[3]^2)^2
                                          +(1-x[3])^2+10*(x[2]+x[4]-2)^2+(x[2]-x[4])^2/10,
                                       [-3.0,-1.0,-3.0,-1.0])),
        ("Trigonometric_5", ADNLPModel(x -> begin n=5
                                           sum((n-sum(cos.(x))+i*(1-cos(x[i]))-sin(x[i]))^2
                                               for i in 1:n) end, fill(1/5,5))),
        ("Powell_8",        ADNLPModel(x -> sum((x[4i-3]+10x[4i-2])^2+5*(x[4i-1]-x[4i])^2
                                               +(x[4i-2]-2x[4i-1])^4+10*(x[4i-3]-x[4i])^4
                                               for i in 1:2),
                                       [3.0,-1.0,0.0,1.0,3.0,-1.0,0.0,1.0])),
        ("BrownBadlyScaled",ADNLPModel(x -> (x[1]-1e6)^2+(x[2]-2e-6)^2+(x[1]*x[2]-2)^2,[1.0,1.0])),
        ("Beale",           ADNLPModel(x -> (1.5-x[1]*(1-x[2]))^2+(2.25-x[1]*(1-x[2]^2))^2
                                          +(2.625-x[1]*(1-x[2]^3))^2, [1.0,1.0])),
        ("Himmelblau",      ADNLPModel(x -> (x[1]^2+x[2]-11)^2+(x[1]+x[2]^2-7)^2,[0.0,0.0])),
    ]
end

if CUTEST_AVAILABLE
    cutest_list = try
        CUTEst.select(min_var = 2, max_var = 200, max_con = 0)
    catch e
        @warn "CUTEst.select failed ($e); falling back to ADNLPModels"; String[]
    end
else
    cutest_list = String[]
end

if !isempty(cutest_list)
    suite_type = :cutest;  problems = cutest_list;  all_labels = cutest_list
    @info "Using CUTEst suite: $(length(problems)) problems"
else
    suite_type = :adnlp;   problems = build_adnlp_suite();  all_labels = first.(problems)
    @info "Using ADNLPModels suite: $(length(problems)) problems"
end
np = length(all_labels)

# ---------------------------------------------------------------------------
# Sweep μ₀
# ---------------------------------------------------------------------------
mu_labels = [@sprintf("μ₀=%.3g", μ) for μ in MU_VALUES]
nmu       = length(MU_VALUES)

records = Dict{String, Dict{String, NamedTuple}}()

for μ₀ in MU_VALUES
    mlabel = @sprintf("μ₀=%.3g", μ₀)
    records[mlabel] = Dict{String, NamedTuple}()

    _solve! = (label, nlp) -> begin
        rule = R4RelativeGradUpdate(0.25, 2.0, μ₀)
        out  = trust_region_solver(nlp, rule, PARAMS)
        records[mlabel][label] = (
            status     = out.status,
            iterations = out.iterations,
            f_evals    = out.f_evals,
            final_mu   = rule.μ,
        )
    end

    if suite_type == :cutest
        for pname in problems
            nlp = nothing
            try
                nlp = CUTEstModel(pname); _solve!(pname, nlp)
            catch e
                @warn "μ₀=$μ₀, $pname: $e"
                records[mlabel][pname] = (status=:failure, iterations=typemax(Int), f_evals=typemax(Int), final_mu=μ₀)
            finally
                nlp !== nothing && finalize(nlp)
            end
        end
    else
        for (label, nlp) in problems
            try
                _solve!(label, nlp)
            catch e
                @warn "μ₀=$μ₀, $label: $e"
                records[mlabel][label] = (status=:failure, iterations=typemax(Int), f_evals=typemax(Int), final_mu=μ₀)
            end
        end
    end

    ns = count(e -> e.status == :solved, values(records[mlabel]))
    @info "μ₀=$μ₀: $ns / $np solved"
end

# ---------------------------------------------------------------------------
# Performance profile
# ---------------------------------------------------------------------------
T_iter = fill(Inf, np, nmu)
for (j, mlabel) in enumerate(mu_labels), (i, plabel) in enumerate(all_labels)
    haskey(records[mlabel], plabel) || continue
    e = records[mlabel][plabel]
    e.status == :solved && (T_iter[i, j] = Float64(e.iterations))
end

pp = performance_profile(
    PlotsBackend(),
    T_iter, mu_labels;
    title    = "R4 performance profile — varying μ₀",
    xlabel   = "Performance ratio τ",
    ylabel   = "Fraction of problems solved",
    logscale = true,
    legend   = :outerright,
)
savefig(pp, joinpath(FIGURES_DIR, "exp4_perf_profile_mu.pdf"))
@info "Saved exp4_perf_profile_mu.pdf"

# ---------------------------------------------------------------------------
# Solve rate vs μ₀
# ---------------------------------------------------------------------------
solve_rates  = Float64[]
median_iters = Float64[]
for mlabel in mu_labels
    its = [Float64(e.iterations) for e in values(records[mlabel]) if e.status == :solved]
    push!(solve_rates,  isempty(its) ? 0.0 : 100 * length(its) / np)
    push!(median_iters, isempty(its) ? NaN  : median(its))
end

p_rate = plot(log10.(MU_VALUES), solve_rates;
              xlabel=  "log₁₀(μ₀)", ylabel = "Solve rate (%)",
              title  = "R4: solve rate vs μ₀",
              marker = :circle, linewidth = 2, legend = false, color = colorant"#9933AA")
savefig(p_rate, joinpath(FIGURES_DIR, "exp4_solverate_vs_mu.pdf"))
@info "Saved exp4_solverate_vs_mu.pdf"

# ---------------------------------------------------------------------------
# μ drift scatter — final μ vs initial μ₀ (solved problems only)
# ---------------------------------------------------------------------------
p_drift = plot(; xlabel = "log₁₀(μ₀)", ylabel = "log₁₀(final μ)",
                 title  = "R4: μ drift", legend = false)

for (j, (μ₀, mlabel)) in enumerate(zip(MU_VALUES, mu_labels))
    fmus = [log10(e.final_mu)
            for e in values(records[mlabel])
            if e.status == :solved && e.final_mu > 0]
    isempty(fmus) && continue
    scatter!(p_drift, fill(log10(μ₀), length(fmus)), fmus;
             alpha = 0.4, marker = :circle, ms = 4, color = colorant"#9933AA")
end
μ_range = log10.([MU_VALUES[1], MU_VALUES[end]])
plot!(p_drift, μ_range, μ_range; linestyle = :dash, color = :black)
savefig(p_drift, joinpath(FIGURES_DIR, "exp4_mu_drift.pdf"))
@info "Saved exp4_mu_drift.pdf"

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
println("\n" * "="^55)
println("EXP 4 — R4 sensitivity to μ₀")
println("="^55)
println(@sprintf("%-12s  %8s  %12s", "μ₀", "Rate%", "Med-iter"))
println("-"^36)
for (j, μ₀) in enumerate(MU_VALUES)
    println(@sprintf("%-12.4g  %8.1f  %12.1f", μ₀, solve_rates[j], median_iters[j]))
end
println("="^55)

table_path = joinpath(TABLES_DIR, "exp4_mu_summary.txt")
open(table_path, "w") do io
    println(io, "% Exp 4 — R4 sensitivity to mu0 (auto-generated)")
    println(io, "\\begin{tabular}{lrr}")
    println(io, "\\toprule")
    println(io, "\\mu_0 & Rate (\\%) & Median iterations \\\\")
    println(io, "\\midrule")
    for (j, μ₀) in enumerate(MU_VALUES)
        println(io, @sprintf("%.4g & %.1f & %.1f \\\\", μ₀, solve_rates[j], median_iters[j]))
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
end
@info "Saved tables/exp4_mu_summary.txt"
