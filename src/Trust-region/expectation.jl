# =============================================================================
# src/Trust-region/expectation.jl
#
# The expectation solver: f(x) = E_ξ[F(x,ξ)], population unbounded.
#
# The loop adds three things to the deterministic one:
#
#   1. resample before every iteration, and re-evaluate f and g at the incumbent,
#      so ared and pred come from the same realisations;
#   2. refresh the variance estimates the adaptive rules consume;
#   3. hand the predicted reduction back to the rule.
#
# What it does NOT have is a full-batch limit. There is no N at which the estimate
# becomes exact, so the accuracy hypotheses of the stochastic convergence theory can
# never be discharged, and `:samples_total` is the only honest cost measure.
# =============================================================================

"""
    ExpectationTRSolver(nlp; rule, model, subsolver, params, budget)

Trust-region solver for an [`ExpectationNLP`](@ref).

The sampling rule lives on the oracle, not here — it decides what the *oracle*
returns, so it belongs with the oracle. `budget` is a safety cap on `N_k` applied on
top of the rule's own `N_max`; the accuracy theory imposes none, so without a cap a
criticality-anchored mechanism will ask for `Θ(Δ_k^{-4})` objective terms and get it.

`TRParams(true_stop = true)` requires `has_truth(prob)`. That is `true` for
[`PerturbedExpectation`](@ref), which supplies the exact mean by construction, and
`false` for a general expectation — where the request is not conservative but
impossible, and is refused rather than silently downgraded to the batch test.
"""
mutable struct ExpectationTRSolver{T, V, R, M, S} <: AbstractTRSolver
    core::TRCore{T, V, R, M, S}
end

function ExpectationTRSolver(nlp::ExpectationNLP;
                             rule::RadiusRule            = RDelta(),
                             model::ModelHessian         = ExactHessian(),
                             subsolver::SubproblemSolver = SteihaugCG(),
                             params::TRParams            = TRParams{Float64}())
    params.true_stop && !has_truth(nlp) && throw(ArgumentError(
        "TRParams(true_stop = true) needs an exact gradient, but " *
        "$(nameof(typeof(nlp.prob))) is an expectation with no closed-form mean. " *
        "Use PerturbedExpectation, or a finite sum, or drop true_stop — the batch " *
        "test is then the only test available and should be reported as such."))
    c = TRCore(nlp, rule, model, subsolver, params)
    return ExpectationTRSolver{Float64, Vector{Float64}, typeof(c.rule),
                               typeof(c.model), typeof(c.subsolver)}(c)
end

Base.getproperty(s::ExpectationTRSolver, f::Symbol) =
    f === :core ? getfield(s, :core) : getproperty(getfield(s, :core), f)

"""
    solve!(solver::ExpectationTRSolver, nlp, stats; callback, trace) -> TRResult

Adds to the deterministic trace, when `trace = true`:

| key | length | entry `i` |
|:--|:--|:--|
| `:grad_sample_trajectory` | `k` | `N_i^grad` |
| `:obj_sample_trajectory` | `k` | `N_i^obj` |
| `:samples_total` | — | cumulative term evaluations |
| `:sample_cap_hits` | — | iterations that hit the budget |
| `:true_grad_trajectory` | `k+1` | `‖∇f(x_i)‖`, only when `has_truth` |

`:grad_trajectory` holds `‖ĝ_k‖` — one batch's estimate. A mechanism that shrinks
the radius fast enough will drive it below `tol` on noise alone, which is why
`:true_grad_trajectory` exists and why every reported number should come from it.
"""
function SolverCore.solve!(solver::ExpectationTRSolver,
                           nlp::ExpectationNLP,
                           stats::GenericExecutionStats;
                           callback = (args...) -> nothing,
                           trace::Bool = false)
    c = solver.core; p = c.params
    t0 = time()
    reset_sampling!(nlp)
    st = _init_state!(c, nlp)
    truth = has_truth(nlp)
    tr = TRTrace(trace, c.want_curv, trace && truth)
    trace_pre!(tr, st, truth ? norm(true_gradient(nlp, c.x)) : NaN)

    _record!(stats, c, st, 0, t0)
    set_status!(stats, :unknown)

    k = 0
    while true
        if _first_order_ok(st, p) && _confirm_stop(nlp, c.x, p)
            set_status!(stats, _final_status(p)); break
        end
        k >= p.max_iterations    && (set_status!(stats, :max_iter);  break)
        time() - t0 > p.max_time && (set_status!(stats, :max_time); break)
        k += 1

        # A fresh draw. The incumbent f and g belong to the previous batch, so they
        # are re-evaluated before anything is compared against them.
        resample!(nlp, k, Float64(st.Δ), Float64(st.g_norm))
        _refresh_state!(c, nlp, st)
        update_variances!(nlp, c.x, c.g)

        _tr_step!(c, nlp, st)
        st.failed  && (set_status!(stats, :exception); k -= 1; break)
        st.stalled && (set_status!(stats, :stalled);   break)
        record_prediction!(nlp, Float64(st.predicted))

        trace_post!(tr, st, truth ? norm(true_gradient(nlp, c.x)) : NaN)
        _record!(stats, c, st, k, t0)
        callback(nlp, solver, stats)
        stats.status == :user && break
    end

    _finish!(stats, c, st, k, t0)
    attach_trace!(stats, tr, st.Δ)
    _attach_sampling!(stats, nlp, trace)
    return stats
end

"""
    _confirm_stop(nlp, x, params) -> Bool

Whether a satisfied stopping test should be believed. Always, unless
`true_stop = true`, in which case `‖ĝ_k‖ ≤ tol` is confirmed against the true
gradient before the run is declared converged.
"""
_confirm_stop(::AbstractNLPModel, x, ::TRParams) = true
function _confirm_stop(nlp::SampledNLP, x, p::TRParams)
    p.true_stop || return true
    return norm(true_gradient(nlp, x)) <= p.tol
end

function _attach_sampling!(stats, nlp::SampledNLP, trace::Bool)
    trace || return nothing
    set_solver_specific!(stats, :grad_sample_trajectory, copy(nlp.Ng_hist))
    set_solver_specific!(stats, :obj_sample_trajectory,  copy(nlp.Nf_hist))
    set_solver_specific!(stats, :samples_total,   samples_used(nlp).total)
    set_solver_specific!(stats, :sample_cap_hits, nlp.capped)
    set_solver_specific!(stats, :problem_class,   problem_class(nlp))
    return nothing
end
