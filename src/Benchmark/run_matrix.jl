# =============================================================================
# src/Benchmark/run_matrix.jl
#
# Cross-product experiment harness.
#
# `run_matrix` runs every configuration on every problem and returns a cost
# matrix ready for `performance_profile` / `data_profile`. It is built on the
# JSO `solve!` path, so counters come from NLPModels rather than being tracked
# by hand.
#
# Problems are supplied as *thunks* (zero-argument functions returning a fresh
# model) rather than as models. This matters for CUTEst, where each
# `CUTEstModel` holds an open handle to a compiled SIF problem and must be
# finalised before the next one is opened.
# =============================================================================

"""
    TRConfig(label; rule, model, subsolver, params)

One cell of the experiment matrix: a named triple of radius rule, model
Hessian, and subproblem solver, together with the solver parameters.

`rule` and `model` carry mutable state (μ for `RGrad`, the operator for
L-BFGS/SR1), so `run_matrix` deep-copies them per run; a single `TRConfig` can
therefore be reused across problems without contamination.
"""
struct TRConfig{R <: AbstractRadiusUpdate, M <: ModelHessian,
                S <: AbstractTRSubproblemSolver, P}
    label::String
    rule::R
    model::M
    subsolver::S
    params::P
end

function TRConfig(label::AbstractString;
                  rule::AbstractRadiusUpdate = R1ClassicalUpdate(),
                  model::ModelHessian = ExactHessian(),
                  subsolver::AbstractTRSubproblemSolver = SteihaugTointCG(),
                  params = TRSolverParams())
    TRConfig(String(label), rule, model, subsolver, params)
end

Base.show(io::IO, c::TRConfig) = print(io, "TRConfig(", c.label, ")")

"""
    run_matrix(problems, configs; cost = :iter, verbose = true)
        -> (T, stats_matrix)

Run every configuration on every problem.

# Arguments
- `problems`: vector of **thunks**, each returning a fresh `AbstractNLPModel`.
  Thunks rather than models so that CUTEst handles are opened and finalised
  around a single run.
- `configs`:  vector of [`TRConfig`](@ref).

# Keyword arguments
- `cost`:    which quantity fills the matrix — `:iter`, `:obj`, `:grad`,
             `:hprod` or `:time`.
- `verbose`: print one line per problem as it completes.

# Returns
- `T[p, c]`: the cost, or `Inf` where the run did not reach `:first_order`.
- `stats_matrix[p, c]`: the `GenericExecutionStats`, or `nothing` on error.

Only runs terminating with status `:first_order` are counted as solved;
`:max_iter`, `:user` and any thrown error give `Inf`. That convention keeps the
profile honest: a solver that stops early with a large gradient has not solved
the problem, whatever its iteration count.

# Example
```julia
problems = [() -> ADNLPModel(x -> sum(x.^2), ones(5), name = "sphere")]
configs  = [TRConfig("R-delta"; rule = RDelta()),
            TRConfig("R-grad";  rule = RGrad())]
T, S = run_matrix(problems, configs)
summarise(T, [c.label for c in configs])
```
"""
function run_matrix(problems, configs; cost::Symbol = :iter, verbose::Bool = true)
    cost in (:iter, :obj, :grad, :hprod, :time) || throw(ArgumentError(
        "run_matrix: cost must be one of :iter, :obj, :grad, :hprod, :time"))

    npb, ns = length(problems), length(configs)
    T = fill(Inf, npb, ns)
    S = Matrix{Any}(nothing, npb, ns)

    if verbose
        @printf("%-22s", "problem")
        for c in configs
            @printf("%12s", first(c.label, 12))
        end
        println()
        println("-"^(22 + 12ns))
    end

    for (i, mk) in enumerate(problems)
        nlp = mk()
        pname = nlp.meta.name
        for (j, cfg) in enumerate(configs)
            # Fresh state per run: both the rule and the model are mutable.
            rule  = deepcopy(cfg.rule)
            model = deepcopy(cfg.model)
            sub   = deepcopy(cfg.subsolver)
            try
                NLPModels.reset!(nlp)
                stats = trust_region_radius(nlp; rule = rule, model = model,
                                            subsolver = sub, params = cfg.params)
                S[i, j] = stats
                if stats.status == :first_order
                    T[i, j] = cost === :iter  ? stats.iter          :
                              cost === :obj   ? neval_obj(nlp)      :
                              cost === :grad  ? neval_grad(nlp)     :
                              cost === :hprod ? neval_hprod(nlp)    :
                                                stats.elapsed_time
                end
            catch err
                err isa InterruptException && rethrow()
                verbose && @warn "run failed" problem = pname config = cfg.label err
            end
        end
        if verbose
            @printf("%-22s", first(pname, 22))
            for j in 1:ns
                isfinite(T[i, j]) ? @printf("%12.4g", T[i, j]) : @printf("%12s", "-")
            end
            println()
        end
        finalize(nlp)
    end
    return T, S
end

"""
    summarise(T, labels; io = stdout)

Print solved counts, median and mean cost per configuration.

The median is over solved problems only, so it is not comparable across columns
with very different reliability — read it together with the solved count, or
use a performance profile, which handles the two jointly.
"""
function summarise(T::AbstractMatrix, labels::AbstractVector; io::IO = stdout)
    npb, ns = size(T)
    length(labels) == ns || throw(DimensionMismatch(
        "summarise: $(length(labels)) labels for $ns columns"))

    @printf(io, "%-28s %10s %12s %12s\n", "configuration", "solved", "median", "mean")
    println(io, "-"^66)
    for (j, lab) in enumerate(labels)
        fin = filter(isfinite, collect(@view T[:, j]))
        if isempty(fin)
            @printf(io, "%-28s %5d/%-4d %12s %12s\n", lab, 0, npb, "--", "--")
        else
            @printf(io, "%-28s %5d/%-4d %12.4g %12.4g\n",
                    lab, length(fin), npb, _median(fin), sum(fin) / length(fin))
        end
    end
    return nothing
end

"Median without a hard dependency on Statistics."
function _median(v::AbstractVector)
    s = sort(collect(v))
    n = length(s)
    n == 0 && return NaN
    isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

# -----------------------------------------------------------------------------
# Convenience: sweep one parameter of one rule
# -----------------------------------------------------------------------------

"""
    sweep_configs(name, values, mkrule; model, subsolver, params) -> Vector{TRConfig}

Build one `TRConfig` per parameter value, labelled `name = value`.

`mkrule(v)` must return a fresh rule for the value `v`.

```julia
configs = sweep_configs("ζ", [0.1, 1.0, 10.0], ζ -> RDFO(0.25, 0.5, 2.0, ζ))
T, _    = run_matrix(problems, configs)
```

A sweep of this kind is the right way to study `ζ` or `μ_max`: the survey's
thresholds involve `κ̄ = 4/λ*_min`, which is a property of the solution and so
cannot be chosen a priori, and only a sweep reveals where the transition sits
on a given family of problems.
"""
function sweep_configs(name::AbstractString, values, mkrule;
                       model::ModelHessian = ExactHessian(),
                       subsolver::AbstractTRSubproblemSolver = SteihaugTointCG(),
                       params = TRSolverParams())
    [TRConfig(string(name, " = ", v); rule = mkrule(v), model = model,
              subsolver = subsolver, params = params) for v in values]
end
