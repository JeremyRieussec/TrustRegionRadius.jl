# =============================================================================
# src/Trust-region/finitesum.jl
#
# The finite-sum solver: f(x) = (1/M) Σ f_i(x), M finite and known.
#
# Structurally the expectation loop, with two things the expectation cannot have:
#
#   1. N_k ≤ M, imposed by the problem rather than chosen by the user. A rule
#      carrying its own N_max is rejected at oracle construction; a deliberate
#      sub-population limit goes in `budget` on the oracle.
#   2. A full-batch iteration. At N_k = M the iteration is *exactly* deterministic:
#      ρ̂ = ρ, the stopping test is the real one, and the accuracy hypotheses are
#      discharged rather than assumed. `:full_batch_trajectory` records which
#      iterations those were, and `FullBatch()` makes every one of them so — under
#      which this solver must reproduce `DeterministicTRSolver` exactly.
# =============================================================================

"""
    FiniteSumTRSolver(nlp; rule, model, subsolver, params)

Trust-region solver for a [`FiniteSumNLP`](@ref).

Every `TRParams(true_stop = …)` mode is available here: `has_truth` holds for the
whole class, so `:full` and `:both` can confirm exactly, at the cost of one full
pass per confirmation. `:both` pays that cost only when the batch cannot settle the
question on its own, and is the mode to prefer.

!!! note "The full-batch reference"
    `FiniteSumNLP(prob, FullBatch())` under this solver is numerically identical to
    `FullBatchNLP(prob)` under [`DeterministicTRSolver`](@ref). That is worth
    knowing, and worth testing: it is the only place where the sampled and exact
    code paths can be compared iterate for iterate, so it pins the resampling and
    re-evaluation logic against a reference that has none.
"""
mutable struct FiniteSumTRSolver{T, V, R, M, S} <: AbstractTRSolver
    core::TRCore{T, V, R, M, S}
end

function FiniteSumTRSolver(nlp::FiniteSumNLP;
                           rule::RadiusRule            = RDelta(),
                           model::ModelHessian         = ExactHessian(),
                           subsolver::SubproblemSolver = SteihaugCG(),
                           params::TRParams            = TRParams{Float64}(),
                           x_ref::Union{Nothing, AbstractVector} = nothing,
                           true_curvature::Bool        = false)
    c = TRCore(nlp, rule, model, subsolver, params;
               x_ref = x_ref, true_curvature = true_curvature)
    return FiniteSumTRSolver{Float64, Vector{Float64}, typeof(c.rule),
                             typeof(c.model), typeof(c.subsolver)}(c)
end

Base.getproperty(s::FiniteSumTRSolver, f::Symbol) =
    f === :core ? getfield(s, :core) : getproperty(getfield(s, :core), f)

"""
    solve!(solver::FiniteSumTRSolver, nlp, stats; callback, trace) -> TRResult

Adds `:full_batch_trajectory` (length `k`, `Bool`) to the expectation trace: the
iterations at which `N_k = M` and the step was therefore exact.

A run whose tail is all `true` has entered the deterministic regime and its final
`‖ĝ‖` is the real one. A run whose tail is all `false` never did, and its status is
a statement about one batch.
"""
function SolverCore.solve!(solver::FiniteSumTRSolver,
                           nlp::FiniteSumNLP,
                           stats::GenericExecutionStats;
                           callback = (args...) -> nothing,
                           trace::Bool = false)
    c = solver.core; p = c.params
    t0 = time()
    reset_sampling!(nlp)
    st = _init_state!(c, nlp)
    tr = TRTrace(trace, c.want_curv)
    sr = SampleTrace(trace, trace)
    trace_pre!(tr, st)
    sample_pre!(sr, norm(true_gradient(nlp, c.x)))

    _record!(stats, c, st, 0, t0)
    set_status!(stats, :unknown)

    k = 0
    while true
        if _first_order_ok(st, p) && _confirm_stop(nlp, c.x, Float64(st.g_norm), p)
            set_status!(stats, _final_status(p)); break
        end
        k >= p.max_iterations    && (set_status!(stats, :max_iter);  break)
        time() - t0 > p.max_time && (set_status!(stats, :max_time); break)
        k += 1

        resample!(nlp, k, Float64(st.Δ), Float64(st.g_norm))
        _refresh_state!(c, nlp, st)
        update_variances!(nlp, c.x, c.g)

        _tr_step!(c, nlp, st)
        st.failed  && (set_status!(stats, :exception); k -= 1; break)
        st.stalled && (set_status!(stats, :stalled);   break)
        record_prediction!(nlp, Float64(st.predicted))

        trace_post!(tr, st)
        sample_post!(sr, c, nlp, norm(true_gradient(nlp, c.x)))
        _record!(stats, c, st, k, t0)
        callback(nlp, solver, stats)
        stats.status == :user && break
    end

    _finish!(stats, c, st, k, t0)
    attach_trace!(stats, tr, st.Δ)
    attach_sample_trace!(stats, sr)
    _attach_sampling!(stats, nlp, trace)
    if trace
        set_solver_specific!(stats, :full_batch_trajectory, copy(nlp.full_hist))
        set_solver_specific!(stats, :population, population(nlp.prob))
    end
    return stats
end
