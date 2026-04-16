# =============================================================================
# benchmark/run_benchmark.jl
#
# Runs all radius update rules on every unconstrained CUTEst problem with
# 2 ≤ n ≤ 500 and saves one JLD2 file per (problem, rule) pair to
# benchmark/temp_results/.
#
# After the run, an `experiment_config.toml` is written to temp_results/ so
# that generate_all_figures.jl can archive everything self-consistently.
#
# Usage (from repo root):
#   julia --project=benchmark benchmark/run_benchmark.jl
#
# To discard all existing temp results and restart from scratch:
#   julia --project=benchmark benchmark/run_benchmark.jl --force
#
# Results are saved as:
#   benchmark/temp_results/<PROBLEM>_<RULE>.jld2
#
# Each JLD2 file contains scalar fields only (no struct dependency at
# load time), making the files self-contained.
# =============================================================================

using Pkg
Pkg.activate(@__DIR__)

# Auto-develop the parent package if it is not yet in the environment
if !haskey(Pkg.project().dependencies, "TrustRegionRadius")
    @info "Developing TrustRegionRadius from parent directory…"
    Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))
end

using TrustRegionRadius
using CUTEst
using JLD2
using LinearAlgebra
using Printf

# =============================================================================
# Configuration  (edit benchmark/config.jl to change any of these values)
# =============================================================================

const TEMP_RESULTS_DIR = joinpath(@__DIR__, "temp_results")
mkpath(TEMP_RESULTS_DIR)

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "config_utils.jl"))

# =============================================================================
# --force flag: optionally wipe existing temp results before starting
# =============================================================================

const FORCE = "--force" in ARGS

existing_jld2 = filter(f -> endswith(f, ".jld2"), readdir(TEMP_RESULTS_DIR))

if !isempty(existing_jld2) && FORCE
    @info "Clearing $(length(existing_jld2)) existing JLD2 file(s) from temp_results/…"
    for f in existing_jld2
        rm(joinpath(TEMP_RESULTS_DIR, f))
    end
    cfg_old = joinpath(TEMP_RESULTS_DIR, "experiment_config.toml")
    isfile(cfg_old) && rm(cfg_old)
elseif !isempty(existing_jld2)
    @info "temp_results/ has $(length(existing_jld2)) existing file(s) — already-computed " *
          "(problem, rule) pairs will be skipped.  Pass --force to restart from scratch."
end

# =============================================================================
# Problem selection
# =============================================================================

@info "Problem selection: min_var=$MIN_VAR, max_var=$MAX_VAR, max_con=$MAX_CON"
@info "Rules: $(join(first.(RULES), ", "))"

@info "Querying CUTEst problem list…"
prob_list = try
    CUTEst.select_sif_problems(min_var = MIN_VAR, max_var = MAX_VAR, max_con = MAX_CON)
catch e
    @warn "CUTEst.select failed ($e); falling back to an empty list."
    String[]
end

if isempty(prob_list)
    error("No CUTEst problems found. Check that CUTEst is properly installed " *
          "(MASTSIF environment variable must be set) and try again.")
end

@info "Found $(length(prob_list)) candidate problems."

# =============================================================================
# Benchmark loop
# =============================================================================

n_total   = 0
n_solved  = Dict(name => 0 for (name, _) in RULES)
n_failed  = Dict(name => 0 for (name, _) in RULES)
n_maxiter = Dict(name => 0 for (name, _) in RULES)

for (prob_idx, prob_name) in enumerate(sort(prob_list))

    global n_total, n_solved, n_failed, n_maxiter

    println("\n[$prob_idx/$(length(prob_list))] $prob_name")
    n_total += 1

    for (rule_name, make_rule) in RULES

        out_path = joinpath(TEMP_RESULTS_DIR, "$(prob_name)_$(rule_name).jld2")

        if isfile(out_path)
            println("  $rule_name: result exists — skipping")
            # Update counters from existing file
            try
                d = load(out_path)
                st = d["status"]
                if st == "solved";       n_solved[rule_name]  += 1
                elseif st == "max_iter"; n_maxiter[rule_name] += 1
                else;                    n_failed[rule_name]  += 1
                end
            catch; end
            continue
        end

        rule = make_rule()
        nlp  = nothing

        try
            nlp = CUTEstModel(prob_name)

            # Safety check: skip if problem has constraints or wrong size
            nvar = nlp.meta.nvar
            ncon = nlp.meta.ncon
            if nvar < MIN_VAR || nvar > MAX_VAR || ncon > MAX_CON
                @warn "  $rule_name: skipping $prob_name (nvar=$nvar, ncon=$ncon)"
                finalize(nlp)
                nlp = nothing
                continue
            end

            t0  = time()
            out = trust_region_solver(nlp, rule, SOLVER_PARAMS)
            elapsed = time() - t0

            status_str = string(out.status)
            println("  $rule_name: $(status_str) in $(out.iterations) iter, " *
                    "f_evals=$(out.f_evals), ‖g‖=$(round(out.final_grad_norm, sigdigits=3)), " *
                    "time=$(round(elapsed, digits=2))s")

            # Count outcomes
            if out.status == :solved
                n_solved[rule_name]  += 1
            elseif out.status == :max_iter
                n_maxiter[rule_name] += 1
            else
                n_failed[rule_name]  += 1
            end

            # Save results as plain fields — no struct type needed at load time
            jldsave(out_path;
                problem_name         = prob_name,
                rule_name            = rule_name,
                n                    = nlp.meta.nvar,
                status               = status_str,
                iterations           = out.iterations,
                f_evals              = out.f_evals,
                final_grad_norm      = out.final_grad_norm,
                final_delta          = out.final_delta,
                delta_trajectory     = out.delta_trajectory,
                grad_norm_trajectory = out.grad_norm_trajectory,
                obj_trajectory       = out.obj_trajectory,
                solve_time           = out.solve_time,
            )

        catch e
            @warn "  $rule_name: EXCEPTION — $e"
            n_failed[rule_name] += 1

            # Save a failure record so this (problem, rule) is not retried
            try
                jldsave(out_path;
                    problem_name         = prob_name,
                    rule_name            = rule_name,
                    n                    = (nlp !== nothing ? nlp.meta.nvar : 0),
                    status               = "failure",
                    iterations           = 0,
                    f_evals              = 0,
                    final_grad_norm      = NaN,
                    final_delta          = NaN,
                    delta_trajectory     = Float64[],
                    grad_norm_trajectory = Float64[],
                    obj_trajectory       = Float64[],
                    solve_time           = 0.0,
                )
            catch save_err
                @warn "  Could not save failure record: $save_err"
            end

        finally
            if nlp !== nothing
                try
                    finalize(nlp)
                catch; end
            end
        end

    end  # rules loop
end  # problems loop

# =============================================================================
# Summary
# =============================================================================

println("\n" * "="^60)
println("BENCHMARK COMPLETE")
println("="^60)
println("Total problems attempted: $n_total")
println()
println(@sprintf("%-8s  %6s  %6s  %6s  %6s  %6s",
        "Rule", "Solved", "MaxIter", "Failed", "Total", "Rate%"))
println("-"^52)
for (rule_name, _) in RULES
    ns = n_solved[rule_name]
    nm = n_maxiter[rule_name]
    nf = n_failed[rule_name]
    tot = ns + nm + nf
    rate = tot > 0 ? round(100 * ns / tot, digits=1) : 0.0
    println(@sprintf("%-8s  %6d  %6d  %6d  %6d  %6.1f",
            rule_name, ns, nm, nf, tot, rate))
end
println("="^60)
println("Results saved to: $TEMP_RESULTS_DIR")

# =============================================================================
# Write experiment config — records exactly what was run and with what params
# =============================================================================

config_path = joinpath(TEMP_RESULTS_DIR, "experiment_config.toml")
save_experiment_config(config_path, RULES, SOLVER_PARAMS, MIN_VAR, MAX_VAR, MAX_CON)

println()
println("Next step:")
println("  julia --project=benchmark benchmark/generate_all_figures.jl")
