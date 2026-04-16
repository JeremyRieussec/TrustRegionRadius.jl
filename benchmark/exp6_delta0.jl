# =============================================================================
# benchmark/exp6_delta0.jl
#
# Experiment 6 — Sensitivity to the initial radius Δ₀
#
# Sweeps Δ₀ for R1, R2, and R3.
# (R4 excluded since it ignores Δ₀ and uses μ₀ · ‖g₀‖ instead.)
#
# Produces:
#   figures/exp6_perf_profile_delta0_R<n>.pdf  — one profile per rule
#   figures/exp6_solverate_vs_delta0.pdf
#   figures/exp6_median_iter_vs_delta0.pdf
#   tables/exp6_delta0_summary.txt
#
# Usage (from repo root):
#   julia --project=benchmark benchmark/exp6_delta0.jl
# =============================================================================

pgfplotsx()   # uncomment if PGFPlotsX is installed
# gr()

CUTEST_AVAILABLE = false
try
    using CUTEst
    global CUTEST_AVAILABLE = true
catch
    @info "CUTEst not available; will use ADNLPModels problems"
end

# ---------------------------------------------------------------------------
# I/O directories
# ---------------------------------------------------------------------------
# RESULTS_DIR can be overridden by TR_RESULTS_DIR env var so that
# generate_all_figures.jl can point exp scripts at temp_results/.
RESULTS_DIR = get(ENV, "TR_RESULTS_DIR", joinpath(@__DIR__, "results"))
FIGURES_DIR = joinpath(RESULTS_DIR, "figures")
TABLES_DIR  = joinpath(RESULTS_DIR, "tables")

const DELTA0_VALUES = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
const RULE_COLORS   = Dict("R1"=>colorant"#3266AD","R2"=>colorant"#1D9E75","R3"=>colorant"#D85A30")
const RULE_STYLES   = Dict("R1"=>:solid,"R2"=>:solid,"R3"=>:dash)

const RULES_D0 = [
    ("R1", (Δ₀) -> R1ClassicalUpdate(0.25, 0.50, 2.0)),
    ("R2", (Δ₀) -> R2StepSizeUpdate(0.25, 0.80, 2.0)),
    ("R3", (Δ₀) -> R3DFOLikeUpdate(0.25, 0.50, 2.0, 1.0)),
]

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
# Run sweep
# ---------------------------------------------------------------------------
records = Dict{String, Dict{String, Dict{String, NamedTuple}}}()

for (rname, make_rule) in RULES_D0
    records[rname] = Dict{String, Dict{String, NamedTuple}}()

    for Δ₀ in DELTA0_VALUES
        dlabel = @sprintf("Δ₀=%.4g", Δ₀)
        records[rname][dlabel] = Dict{String, NamedTuple}()
        params = TRSolverParams(η₁=0.1, η₂=0.9, Δ₀=Δ₀, max_iterations=5_000, tol=1e-5)

        _solve! = (label, nlp) -> begin
            rule = make_rule(Δ₀)
            out  = trust_region_solver(nlp, rule, params)
            records[rname][dlabel][label] = (status=out.status, iterations=out.iterations, f_evals=out.f_evals)
        end

        if suite_type == :cutest
            for pname in problems
                nlp = nothing
                try
                    nlp = CUTEstModel(pname); _solve!(pname, nlp)
                catch e
                    @warn "$rname, Δ₀=$Δ₀, $pname: $e"
                    records[rname][dlabel][pname] = (status=:failure, iterations=typemax(Int), f_evals=typemax(Int))
                finally
                    nlp !== nothing && finalize(nlp)
                end
            end
        else
            for (label, nlp) in problems
                try
                    _solve!(label, nlp)
                catch e
                    @warn "$rname, Δ₀=$Δ₀, $label: $e"
                    records[rname][dlabel][label] = (status=:failure, iterations=typemax(Int), f_evals=typemax(Int))
                end
            end
        end

        ns = count(e -> e.status == :solved, values(records[rname][dlabel]))
        @info "$rname, Δ₀=$Δ₀: $ns / $np solved"
    end
end

# ---------------------------------------------------------------------------
# Per-rule performance profiles
# ---------------------------------------------------------------------------
delta0_labels = [@sprintf("Δ₀=%.4g", Δ₀) for Δ₀ in DELTA0_VALUES]
nd = length(DELTA0_VALUES)

for (rname, _) in RULES_D0
    T = fill(Inf, np, nd)
    for (j, dlabel) in enumerate(delta0_labels), (i, plabel) in enumerate(all_labels)
        haskey(records[rname][dlabel], plabel) || continue
        e = records[rname][dlabel][plabel]
        e.status == :solved && (T[i, j] = Float64(e.iterations))
    end

    pp = performance_profile(
        PlotsBackend(),
        T, delta0_labels;
        title    = "$rname: performance profile — varying Δ₀",
        xlabel   = "Performance ratio τ",
        ylabel   = "Fraction of problems solved",
        logscale = true,
        legend   = :outerright,
    )
    savefig(pp, joinpath(FIGURES_DIR, "exp6_perf_profile_delta0_$(rname).pdf"))
    @info "Saved exp6_perf_profile_delta0_$(rname).pdf"
end

# ---------------------------------------------------------------------------
# Solve rate and median iterations vs Δ₀ — all rules in one plot
# ---------------------------------------------------------------------------
p_rate = plot(; xlabel="log₁₀(Δ₀)", ylabel="Solve rate (%)",
                title="Sensitivity to Δ₀", legend=:outerright)
p_med  = plot(; xlabel="log₁₀(Δ₀)", ylabel="Median iterations",
                title="Median iterations vs Δ₀", legend=:outerright)

for (rname, _) in RULES_D0
    rates = Float64[]; meds = Float64[]
    for dlabel in delta0_labels
        its = [Float64(e.iterations) for e in values(records[rname][dlabel]) if e.status == :solved]
        push!(rates, isempty(its) ? 0.0 : 100 * length(its) / np)
        push!(meds,  isempty(its) ? NaN  : median(its))
    end
    plot!(p_rate, log10.(DELTA0_VALUES), rates;
          label=rname, color=RULE_COLORS[rname], linestyle=RULE_STYLES[rname],
          marker=:circle, linewidth=2)
    plot!(p_med,  log10.(DELTA0_VALUES), meds;
          label=rname, color=RULE_COLORS[rname], linestyle=RULE_STYLES[rname],
          marker=:square, linewidth=2)
end
savefig(p_rate, joinpath(FIGURES_DIR, "exp6_solverate_vs_delta0.pdf"))
savefig(p_med,  joinpath(FIGURES_DIR, "exp6_median_iter_vs_delta0.pdf"))
@info "Saved exp6_solverate_vs_delta0.pdf and exp6_median_iter_vs_delta0.pdf"

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
println("\n" * "="^70)
println("EXP 6 — Sensitivity to Δ₀  (solve rate %)")
println("="^70)
rule_header = join([r for (r, _) in RULES_D0], "  ")
println(@sprintf("%-8s  %s", "Δ₀", rule_header))
println("-"^50)

latex_rows = String[]
for Δ₀ in DELTA0_VALUES
    dlabel = @sprintf("Δ₀=%.4g", Δ₀)
    rates  = map(RULES_D0) do (rname, _)
        its = [e.iterations for e in values(records[rname][dlabel]) if e.status == :solved]
        isempty(its) ? 0.0 : 100 * length(its) / np
    end
    println(@sprintf("%-8.4g  " * join(["%-8.1f" for _ in rates], "  "), Δ₀, rates...))
    push!(latex_rows,
          @sprintf("%.4g & " * join(["%.1f\\%%" for _ in rates], " & ") * " \\\\",
                   Δ₀, rates...))
end
println("="^70)

table_path = joinpath(TABLES_DIR, "exp6_delta0_summary.txt")
open(table_path, "w") do io
    println(io, "% Exp 6 — Sensitivity to Delta0 (auto-generated)")
    rule_hdr = join([r for (r, _) in RULES_D0], " & ")
    println(io, "\\begin{tabular}{l" * repeat("r", length(RULES_D0)) * "}")
    println(io, "\\toprule")
    println(io, "\\Delta_0 & $rule_hdr \\\\")
    println(io, "\\midrule")
    foreach(row -> println(io, row), latex_rows)
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
end
@info "Saved tables/exp6_delta0_summary.txt"
