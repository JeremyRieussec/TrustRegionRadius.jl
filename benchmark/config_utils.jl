# =============================================================================
# benchmark/config_utils.jl
#
# Utilities for serialising the benchmark configuration to a TOML file so
# that every archived experiment is self-documenting and exactly reproducible.
#
# Usage (in run_benchmark.jl, after the benchmark loop):
#
#   include("config_utils.jl")
#   include("config.jl")          # defines RULES, SOLVER_PARAMS, MIN_VAR, ...
#   save_experiment_config(
#       joinpath(@__DIR__, "temp_results", "experiment_config.toml"),
#       RULES, SOLVER_PARAMS, MIN_VAR, MAX_VAR, MAX_CON,
#   )
# =============================================================================

using TOML
using Dates

# -----------------------------------------------------------------------------
# Unicode field-name → ASCII TOML-key mapping
# -----------------------------------------------------------------------------

"""
    UNICODE_TO_ASCII :: Dict{Symbol, String}

Maps the Unicode symbols used as struct field names throughout the package to
plain-ASCII TOML keys.  Only symbols that appear as field names in
`TRSolverParams` or any `AbstractRadiusUpdate` subtype need entries here; all
others fall back to `string(sym)`.
"""
const UNICODE_TO_ASCII = Dict{Symbol, String}(
    :η₁             => "eta1",
    :η₂             => "eta2",
    :γ₁             => "gamma1",
    :γ₂             => "gamma2",
    :γ₃             => "gamma3",
    :Δ₀             => "Delta0",
    :ζ              => "zeta",
    :μ              => "mu",
    :β              => "beta",
    :λ₁             => "lambda1",
    :λ₂             => "lambda2",
    :M              => "M",
    :η              => "eta",
    :max_iterations  => "max_iterations",
    :tol             => "tol",
)

"""
    field_to_ascii(sym) -> String

Convert a (possibly Unicode) field-name symbol to an ASCII TOML key.
Falls back to `string(sym)` for unmapped symbols.
"""
field_to_ascii(sym::Symbol) = get(UNICODE_TO_ASCII, sym, string(sym))

# -----------------------------------------------------------------------------
# Per-rule serialisation
# -----------------------------------------------------------------------------

"""
    rule_to_dict(name, factory) -> Dict{String, Any}

Call `factory()` to create a fresh rule instance, then introspect its fields
and return a plain `Dict` with:
- `"name"` → the rule's display name (e.g. `"R1"`)
- `"type"` → the concrete struct name (e.g. `"R1ClassicalUpdate"`)
- one entry per field, keyed by its ASCII alias

Only the *initial* field values are captured (μ from mutable rules reflects
the starting state, not the state after a run).
"""
function rule_to_dict(name::String, factory)::Dict{String, Any}
    rule = factory()
    d = Dict{String, Any}()
    d["name"] = name
    d["type"] = string(nameof(typeof(rule)))
    for fname in fieldnames(typeof(rule))
        d[field_to_ascii(fname)] = Float64(getfield(rule, fname))
    end
    return d
end

# -----------------------------------------------------------------------------
# Solver-parameters serialisation
# -----------------------------------------------------------------------------

"""
    solver_params_to_dict(params) -> Dict{String, Any}

Serialise a `TRSolverParams` instance to a plain ASCII-keyed `Dict`.
"""
function solver_params_to_dict(params)::Dict{String, Any}
    d = Dict{String, Any}()
    for fname in fieldnames(typeof(params))
        val = getfield(params, fname)
        # Store integers as integers, floats as floats
        d[field_to_ascii(fname)] = val
    end
    return d
end

# -----------------------------------------------------------------------------
# Top-level writer
# -----------------------------------------------------------------------------

"""
    save_experiment_config(path, rules, params, min_var, max_var, max_con)

Write the full experiment configuration to a TOML file at `path`.

The resulting file records:
- timestamp of the run
- solver parameters
- CUTEst problem-selection bounds
- every radius-update rule with all its parameters

This file is archived alongside the JLD2 results by `generate_all_figures.jl`
so that every experiment folder is self-documenting.
"""
function save_experiment_config(
        path::String,
        rules,
        params,
        min_var::Int,
        max_var::Int,
        max_con::Int,
)
    cfg = Dict{String, Any}()
    cfg["generated_at"]      = string(now())
    cfg["solver_params"]     = solver_params_to_dict(params)
    cfg["problem_selection"] = Dict{String, Any}(
        "min_var" => min_var,
        "max_var" => max_var,
        "max_con" => max_con,
    )
    cfg["rules"] = [rule_to_dict(name, factory) for (name, factory) in rules]

    open(path, "w") do io
        TOML.print(io, cfg)
    end
    @info "Wrote experiment config → $path"
    return path
end
