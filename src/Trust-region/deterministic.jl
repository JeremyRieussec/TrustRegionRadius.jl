# =============================================================================
# src/Trust-region/deterministic.jl
#
# The deterministic solver: f evaluated exactly, no batch, no sampling rule.
#
# There is deliberately no `sampling` keyword and no resampling branch. A
# deterministic run should not carry the machinery for a regime it is not in, and
# passing a sampling rule here should be a constructor error rather than a silently
# ignored argument.
# =============================================================================

"""
    DeterministicTRSolver(nlp; rule, model, subsolver, params)

Trust-region solver for a problem evaluated exactly: any `AbstractNLPModel`, or the
full-batch view [`FullBatchNLP`](@ref) of a finite sum.

Accepts no sampling rule. It has no `resample!`, no variance estimate and no sample
counters, and any `TRParams(true_stop = …)` other than `:none` is rejected — the
batch test *is* the true test here, so confirming it against the truth is
meaningless rather than merely redundant.

```julia
stats = tr_solve(nlp; rule = RGrad(), model = SR1Model(mem = 5),
                 subsolver = SteihaugCG(), params = TRParams(tol = 1e-8))
```
"""
mutable struct DeterministicTRSolver{T, V, R, M, S} <: AbstractTRSolver
    core::TRCore{T, V, R, M, S}
end

function DeterministicTRSolver(nlp::AbstractNLPModel{T, V};
                               rule::RadiusRule            = RDelta(),
                               model::ModelHessian         = ExactHessian(),
                               subsolver::SubproblemSolver = SteihaugCG(),
                               params::TRParams            = TRParams{T}(),
                               x_ref::Union{Nothing, AbstractVector} = nothing,
                               true_curvature::Bool        = false) where {T, V}
    nlp isa SampledNLP && throw(ArgumentError(
        "DeterministicTRSolver was given a $(nameof(typeof(nlp))), which is a " *
        "$(problem_class(nlp)) oracle. Use $(nlp isa ExpectationNLP ?
        "ExpectationTRSolver" : "FiniteSumTRSolver"), or wrap the problem in " *
        "FullBatchNLP to solve it exactly."))
    confirms_stop(params) && throw(ArgumentError(
        "TRParams(true_stop = :$(params.true_stop)) has no meaning for a " *
        "deterministic problem: the stopping test is already the true one. " *
        "Use true_stop = :none."))
    c = TRCore(nlp, rule, model, subsolver, params;
               x_ref = x_ref, true_curvature = true_curvature)
    return DeterministicTRSolver{T, V, typeof(c.rule), typeof(c.model),
                                 typeof(c.subsolver)}(c)
end

Base.getproperty(s::DeterministicTRSolver, f::Symbol) =
    f === :core ? getfield(s, :core) : getproperty(getfield(s, :core), f)

SolverCore.reset!(s::AbstractTRSolver) =
    (reset_rule!(s.core.rule); reset_model!(s.core.model, length(s.core.x)); s)
SolverCore.reset!(s::AbstractTRSolver, ::AbstractNLPModel) = SolverCore.reset!(s)

"""
    solve!(solver::DeterministicTRSolver, nlp, stats; callback, trace) -> TRResult

# Statuses
- `:first_order`  — `‖g‖ ≤ tol`
- `:second_order` — additionally `λ_min(B) ≥ −tol_H`
- `:max_iter`, `:max_time` — budget exhausted
- `:stalled` — the step fell below the level at which `f(x) − f(x+s)` carries
  information, so ρ became noise
- `:exception` — the model could not be built at the current iterate
- `:user` — stopped by the callback
"""
function SolverCore.solve!(solver::DeterministicTRSolver{T, V},
                           nlp::AbstractNLPModel{T, V},
                           stats::GenericExecutionStats{T, V};
                           callback = (args...) -> nothing,
                           trace::Bool = false) where {T, V}
    c = solver.core; p = c.params
    t0 = time()
    st = _init_state!(c, nlp)
    tr = TRTrace(trace, c.want_curv)
    trace_pre!(tr, st)

    _record!(stats, c, st, 0, t0)
    set_status!(stats, :unknown)

    k = 0
    while true
        _first_order_ok(st, p) && (set_status!(stats, _final_status(p)); break)
        k >= p.max_iterations   && (set_status!(stats, :max_iter);  break)
        time() - t0 > p.max_time && (set_status!(stats, :max_time); break)
        k += 1

        _tr_step!(c, nlp, st)
        st.failed  && (set_status!(stats, :exception); k -= 1; break)
        st.stalled && (set_status!(stats, :stalled);   break)

        trace_post!(tr, st)
        _record!(stats, c, st, k, t0)
        callback(nlp, solver, stats)
        stats.status == :user && break
    end

    _finish!(stats, c, st, k, t0)
    attach_trace!(stats, tr, st.Δ)
    return stats
end
