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
#
# ---------------------------------------------------------------------------
# The thunk is also where the problem CLASS is chosen
#
# `tr_solve` now dispatches on the oracle, so a thunk returning a plain NLP gets
# `DeterministicTRSolver`, one returning a `FiniteSumNLP` gets
# `FiniteSumTRSolver`, and one returning an `ExpectationNLP` gets
# `ExpectationTRSolver`. Nothing here needs to know which — but two consequences
# do land in this file:
#
#   * the sampling rule lives on the ORACLE, not on `TRConfig`, so a sweep over
#     sampling rules varies the *problem* thunk rather than the configuration:
#         [() -> FiniteSumNLP(prob, r) for r in rules]
#   * iteration and evaluation counts are not comparable across sampling rules;
#     `cost = :samples` is the measure that is. See the `cost` table below.
# =============================================================================

"""
    TRConfig(label; rule, model, subsolver, params)

One cell of the experiment matrix: a named triple of radius rule, model
Hessian, and subproblem solver, together with the solver parameters.

`rule` and `model` carry mutable state (μ for `RGrad`, the operator for
L-BFGS/SR1), so `run_matrix` deep-copies them per run; a single `TRConfig` can
therefore be reused across problems without contamination.

The solver constructor deep-copies its three axes as well, so the copy here is
now belt-and-braces rather than load-bearing. It is kept deliberately: it makes
this file correct on its own terms rather than by relying on an invariant
maintained two modules away, and one `deepcopy` of a rule is free beside a solve.

The **sampling** rule is not a field here. It lives on the oracle, because it
decides what the oracle returns — so a sweep over sampling rules varies the
problem thunks, not the configurations:

```julia
problems = [() -> FiniteSumNLP(prob, r) for r in (FixedSample(64),
                                                  RadiusProportional(),
                                                  NormTest())]
configs  = [TRConfig("R-delta"; rule = RDelta())]
T, _ = run_matrix(problems, configs; cost = :samples)
```

Note that this puts the sampling rules down the *rows*, where `performance_profile`
treats them as different problems rather than different solvers. To profile them
as solvers, transpose `T` — or build one thunk per problem and one config per
rule and accept that the sampling rule is then fixed across the matrix.
"""
struct TRConfig{R <: RadiusRule, M <: ModelHessian,
                S <: SubproblemSolver, P}
    label::String
    rule::R
    model::M
    subsolver::S
    params::P
end

function TRConfig(label::AbstractString;
                  rule::RadiusRule = RDelta(),
                  model::ModelHessian = ExactHessian(),
                  subsolver::SubproblemSolver = SteihaugCG(),
                  params = TRParams())
    TRConfig(String(label), rule, model, subsolver, params)
end

Base.show(io::IO, c::TRConfig) = print(io, "TRConfig(", c.label, ")")

"""
    _SOLVED_STATUSES

Statuses that count as having solved the problem.

`:second_order` joins `:first_order` here. A run with `tol_H > 0` that reaches a
certified second-order point reports `:second_order`, and treating that as a
failure would have scored the *strongest* possible outcome as `Inf` — silently
zeroing the reliability of every second-order column in a profile.
"""
const _SOLVED_STATUSES = (:first_order, :second_order)

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
- `cost`:    which quantity fills the matrix.

  | `cost` | measures | valid on |
  |:--|:--|:--|
  | `:iter` | `stats.iter` | any |
  | `:obj`, `:grad`, `:hprod` | `neval_*(nlp)` | any |
  | `:time` | `stats.elapsed_time` | any |
  | `:samples` | cumulative term evaluations | sampled oracles only |

  **`:iter` is proportional to work only under `FixedSample`.** Under any
  adaptive sampling rule, `N_k` varies per iteration and the same runs order
  differently by the two measures — which is the whole point of the sampling
  axis. Use `:samples` whenever the columns differ in sampling rule; it raises
  on a deterministic problem rather than returning a meaningless zero.
- `verbose`: print one line per problem as it completes.

# Returns
- `T[p, c]`: the cost, or `Inf` where the run did not reach `:first_order`.
- `stats_matrix[p, c]`: the `GenericExecutionStats`, or `nothing` on error.

Runs terminating `:first_order` **or `:second_order`** count as solved;
`:max_iter`, `:stalled`, `:user`, `:exception` and any thrown error give `Inf`.
That convention keeps the profile honest: a solver that stops early with a large
gradient has not solved the problem, whatever its iteration count.

A configuration rejected by the class checks — `BHHHModel` on a problem that is
not a likelihood, a rule carrying `N_max` on a finite sum — raises from the
solver constructor and is recorded as `Inf` with a warning, so one illegal cell
does not abort the matrix.

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
    cost in (:iter, :obj, :grad, :hprod, :time, :samples) || throw(ArgumentError(
        "run_matrix: cost must be one of :iter, :obj, :grad, :hprod, :time, :samples"))

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
                nlp isa SampledNLP && reset_sampling!(nlp)   # batches, RNG, counters
                stats = tr_solve(nlp; rule = rule, model = model,
                                 subsolver = sub, params = cfg.params)
                S[i, j] = stats
                if stats.status in _SOLVED_STATUSES
                    T[i, j] = _cost_of(cost, stats, nlp)
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
    _cost_of(cost, stats, nlp) -> Float64

The requested cost of one completed run.

`:samples` is the only entry that inspects the oracle rather than the stats, and
the only one restricted by problem class: term evaluations exist only where terms
are sampled, so asking for it on a deterministic problem is an error rather than
a zero — a column of zeros in a cost matrix reads as "free", which is the
opposite of what it would mean.
"""
function _cost_of(cost::Symbol, stats, nlp)
    cost === :iter  && return float(stats.iter)
    cost === :obj   && return float(neval_obj(nlp))
    cost === :grad  && return float(neval_grad(nlp))
    cost === :hprod && return float(neval_hprod(nlp))
    cost === :time  && return float(stats.elapsed_time)
    # :samples
    nlp isa SampledNLP || throw(ArgumentError(
        "run_matrix: cost = :samples needs a sampled oracle, but this problem is " *
        "$(nameof(typeof(nlp))) ($(problem_class(nlp))). Use :iter, :grad or " *
        ":time for a deterministic run."))
    return float(samples_used(nlp).total)
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
configs = sweep_configs("ζ", [0.1, 1.0, 10.0], ζ -> RDFO(ζ = ζ))
T, _    = run_matrix(problems, configs)
```

A sweep of this kind is the right way to study `ζ` or `μ_max`: the survey's
thresholds involve `κ̄ = 4/λ*_min`, which is a property of the solution and so
cannot be chosen a priori, and only a sweep reveals where the transition sits
on a given family of problems.

`mkrule` may return any `RadiusRule`, including a `SecondOrder` wrapper — in which
case pair it with `params = TRParams(tol_H = 1e-6)` and a model that can report
negative curvature, or the sweep measures an expensive no-op. Runs reaching a
certified second-order point report `:second_order`, which `run_matrix` counts as
solved.
"""
function sweep_configs(name::AbstractString, values, mkrule;
                       model::ModelHessian = ExactHessian(),
                       subsolver::SubproblemSolver = SteihaugCG(),
                       params = TRParams())
    [TRConfig(string(name, " = ", v); rule = mkrule(v), model = model,
              subsolver = subsolver, params = params) for v in values]
end
