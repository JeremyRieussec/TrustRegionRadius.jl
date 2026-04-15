# =============================================================================
# benchmark/run_benchmark.jl
#
# Runs all four canonical radius update mechanisms (R1–R4) on every
# unconstrained CUTEst problem with 2 ≤ n ≤ 500 and saves one JLD2
# file per (problem, mechanism) pair to benchmark/results/.
#
# Usage (from repo root):
#   julia --project=benchmark benchmark/run_benchmark.jl
#
# Results are saved as:
#   benchmark/results/<PROBLEM>_<RULE>.jld2
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
# Configuration
# =============================================================================

const RESULTS_DIR = joinpath(@__DIR__, "results")
mkpath(RESULTS_DIR)

const SOLVER_PARAMS = TRSolverParams(
    η₁ = 0.1,
    η₂ = 0.9,
    Δ₀ = 1.0,
    max_iterations = 10_000,
    tol = 1e-5,
)

# Factory functions so each run gets a fresh (and for R4 mutable) rule
const RULES = [
    ("R1", () -> R1ClassicalUpdate(0.25, 0.50, 2.0)),
    ("R2", () -> R2StepSizeUpdate(0.25, 0.80, 2.0)),
    ("R3", () -> R3DFOLikeUpdate(0.25, 0.50, 2.0, 1.0)),
    ("R4", () -> R4RelativeGradUpdate(0.25, 2.0,  1.0)),
]

# =============================================================================
# Problem selection
# =============================================================================
println("Enter problem selection parameters:")
print(" - Minimum number of variables: ")
input = readline()
min_vars =  try
    parse(Int, input)
    # println("Input was: ", min_vars)
catch
    println("Error: Please enter a valid integer.")
end

print(" - Maximum number of variables: ")
input = readline()
max_vars = try
    parse(Int, input)
    # println("Input was: ", max_vars)
catch
    println("Error: Please enter a valid integer.")
end
print(" - Maximum number of constraints: ")
input = readline()
max_cons = try
    parse(Int, input)
    # println("Input was: ", max_cons)
catch
    println("Error: Please enter a valid integer.") 
end
println("Using problem selection: min_var=$min_vars, max_var=$max_vars, max_con=$max_cons")

@info "Querying CUTEst problem list…"
prob_list = try
    CUTEst.select_sif_problems(min_var = min_vars, max_var = max_vars, max_con = max_cons)
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

    global n_total, n_solved, n_failed, n_maxiter, min_var, max_var, max_con 
    # Make these available in the global scope for the finally block

    println("\n[$prob_idx/$(length(prob_list))] $prob_name")
    n_total += 1

    for (rule_name, make_rule) in RULES

        out_path = joinpath(RESULTS_DIR, "$(prob_name)_$(rule_name).jld2")

        if isfile(out_path)
            println("  $rule_name: result exists — skipping")
            # Update counters from existing file
            try
                d = load(out_path)
                st = d["status"]
                if st == "solved";   n_solved[rule_name]  += 1
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
            if nvar < min_var || nvar > max_var || ncon > max_con
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
println(@sprintf("%-6s  %6s  %6s  %6s  %6s  %6s",
        "Rule", "Solved", "MaxIter", "Failed", "Total", "Rate%"))
println("-"^48)
for (rule_name, _) in RULES
    ns = n_solved[rule_name]
    nm = n_maxiter[rule_name]
    nf = n_failed[rule_name]
    tot = ns + nm + nf
    rate = tot > 0 ? round(100 * ns / tot, digits=1) : 0.0
    println(@sprintf("%-6s  %6d  %6d  %6d  %6d  %6.1f",
            rule_name, ns, nm, nf, tot, rate))
end
println("="^60)
println("Results saved to: $RESULTS_DIR")
