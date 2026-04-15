# =============================================================================
# benchmark/load_results.jl
#
# Shared utility used by all experiment scripts.
# Loads every JLD2 file in results_dir/ and organises the data into a
# nested dictionary:
#
#   results[problem_name][rule_name]  →  NamedTuple (all TROutput fields
#                                         plus problem metadata)
#
# Usage:
#   include(joinpath(@__DIR__, "load_results.jl"))
#   results, prob_names, rule_names = load_all_results(joinpath(@__DIR__, "results"))
# =============================================================================

using JLD2
using LinearAlgebra

"""
    load_all_results(results_dir) -> (results, prob_names, rule_names)

Load all `.jld2` files in `results_dir` and return:
- `results`    : `Dict{String, Dict{String, NamedTuple}}` indexed by
                 problem name → rule name → data
- `prob_names` : sorted `Vector{String}` of problem names that have at
                 least one result file
- `rule_names` : sorted `Vector{String}` of mechanism names found

Files that cannot be loaded are skipped with a warning.
"""
function load_all_results(results_dir::String)

    if !isdir(results_dir)
        error("Results directory not found: $results_dir\n" *
              "Run benchmark/run_benchmark.jl first.")
    end

    jld2_files = filter(f -> endswith(f, ".jld2"), readdir(results_dir))

    if isempty(jld2_files)
        error("No .jld2 files found in $results_dir.\n" *
              "Run benchmark/run_benchmark.jl first.")
    end

    results    = Dict{String, Dict{String, NamedTuple}}()
    rule_set   = Set{String}()

    for fname in jld2_files
        fpath = joinpath(results_dir, fname)
        try
            d = load(fpath)

            prob  = get(d, "problem_name", splitext(fname)[1])
            rule  = get(d, "rule_name",    "unknown")

            entry = (
                problem_name         = String(prob),
                rule_name            = String(rule),
                n                    = Int(get(d, "n", 0)),
                status               = Symbol(get(d, "status", "failure")),
                iterations           = Int(get(d, "iterations", 0)),
                f_evals              = Int(get(d, "f_evals", 0)),
                final_grad_norm      = Float64(get(d, "final_grad_norm", NaN)),
                final_delta          = Float64(get(d, "final_delta", NaN)),
                delta_trajectory     = Vector{Float64}(get(d, "delta_trajectory",     Float64[])),
                grad_norm_trajectory = Vector{Float64}(get(d, "grad_norm_trajectory", Float64[])),
                obj_trajectory       = Vector{Float64}(get(d, "obj_trajectory",       Float64[])),
                solve_time           = Float64(get(d, "solve_time", 0.0)),
            )

            if !haskey(results, entry.problem_name)
                results[entry.problem_name] = Dict{String, NamedTuple}()
            end
            results[entry.problem_name][entry.rule_name] = entry
            push!(rule_set, entry.rule_name)

        catch e
            @warn "Skipping $fname: $e"
        end
    end

    prob_names = sort(collect(keys(results)))
    rule_names = sort(collect(rule_set))

    @info "Loaded results: $(length(prob_names)) problems × $(length(rule_names)) mechanisms"

    return results, prob_names, rule_names
end


"""
    problems_all_solved(results, prob_names, rule_names) -> Vector{String}

Return the subset of `prob_names` where every mechanism in `rule_names`
has `status == :solved`.
"""
function problems_all_solved(results, prob_names, rule_names)
    filter(prob_names) do p
        all(rule_names) do r
            haskey(results[p], r) && results[p][r].status == :solved
        end
    end
end


"""
    problems_all_failed(results, prob_names, rule_names) -> Vector{String}

Return the subset of `prob_names` where every mechanism failed.
These problems should be excluded from all experiments.
"""
function problems_all_failed(results, prob_names, rule_names)
    filter(prob_names) do p
        all(rule_names) do r
            !haskey(results[p], r) || results[p][r].status == :failure
        end
    end
end


"""
    build_metric_matrix(results, prob_names, rule_names, metric_fn;
                        failure_val = Inf) -> Matrix{Float64}

Assemble an `(n_problems × n_solvers)` matrix whose `(i, j)` entry is
`metric_fn(results[prob_names[i]][rule_names[j]])`, or `failure_val`
if the entry is missing or the status is not `:solved`.

`metric_fn` receives the NamedTuple for a single (problem, rule) result.
"""
function build_metric_matrix(
        results::Dict, prob_names::Vector{String}, rule_names::Vector{String},
        metric_fn::Function; failure_val::Float64 = Inf)

    m = length(prob_names)
    s = length(rule_names)
    M = fill(failure_val, m, s)

    for (i, p) in enumerate(prob_names), (j, r) in enumerate(rule_names)
        if haskey(results, p) && haskey(results[p], r)
            entry = results[p][r]
            if entry.status == :solved
                M[i, j] = metric_fn(entry)
            end
        end
    end
    return M
end
