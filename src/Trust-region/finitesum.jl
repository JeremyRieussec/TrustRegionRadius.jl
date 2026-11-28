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
pass per confirmation. `:both` pays that cost when the `:test` bound fails or when the
batch is below `stop_N_min`, and it accepts without a pass whatever `:test` accepts.

Without a `budget` the largest batch is the population, and so is the default
`stop_N_min`. A refused stop, under the default `on_refusal = :cap`, and a stall on a
smaller batch move the run to the population for good. With that default no
confirming mode accepts a stop that the exact gradient does not satisfy, and under
every confirming mode `:stalled` is reported only on the population, after its batch
has failed the stopping test. [`TRParams`](@ref) describes the wrong stop this
prevents and what it costs.

A confirming mode is refused on an oracle with `scheme = :nested` and
`replace = true` whose largest batch is the population, because a batch of `M` drawn
that way repeats some observations and misses others.

!!! note "The full-batch reference"
    `FiniteSumNLP(prob, FullBatch())` under this solver reproduces `FullBatchNLP(prob)`
    under [`DeterministicTRSolver`](@ref), in status, iteration count and iterates,
    under every `true_stop` mode, with exceptions that all come from the initial
    batch. The sampled solver evaluates `x₀` on its first `N_init` terms before the
    first resample. The first entry of `:grad_trajectory` differs for that reason, and
    that of `:obj_trajectory` can. So do the iterates, and with them possibly the
    iteration count, under a radius rule that sets its first radius from `‖g(x₀)‖`,
    such as `RGrad` and `RAdaptiveGrad`, and a run whose
    `x₀` already meets the stopping test takes one iteration where the deterministic
    solver takes none. With `N_init = M` these differences vanish too. That is worth
    knowing, and worth testing: it is the only place where the sampled and exact code
    paths can be compared iterate for iterate, so it pins the resampling and
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
                           true_curvature::Bool        = false,
                           hessian_norm::Bool          = false)
    _check_largest_batch(nlp, params)
    c = TRCore(nlp, rule, model, subsolver, params;
               x_ref = x_ref, true_curvature = true_curvature,
               hessian_norm = hessian_norm)
    return FiniteSumTRSolver{Float64, Vector{Float64}, typeof(c.rule),
                             typeof(c.model), typeof(c.subsolver)}(c)
end

# A confirming mode concludes on the largest batch and moves the run there, and on a
# finite sum without a budget that batch has to be the population. Under :nested with
# replacement it is not. The pool grows by appending draws with replacement, so a batch
# of M repeats some observations and misses others, while `grad_standard_error` and
# `:full_batch_trajectory` treat it as the population.
function _check_largest_batch(nlp::FiniteSumNLP, p::TRParams)
    (confirms_stop(p) && nlp.scheme === :nested && nlp.replace &&
     _largest_batch(nlp) >= population(nlp.prob)) || return nothing
    throw(ArgumentError(
        "TRParams(true_stop = :$(p.true_stop)) concludes on the largest batch, " *
        "which on this oracle should be the population of M = " *
        "$(population(nlp.prob)). Under scheme = :nested with replace = true a batch " *
        "of M draws repeats some observations and misses others. Pass " *
        "replace = false to FiniteSumNLP, or use scheme = :independent or :prefix."))
end

Base.getproperty(s::FiniteSumTRSolver, f::Symbol) =
    f === :core ? getfield(s, :core) : getproperty(getfield(s, :core), f)

"""
    solve!(solver::FiniteSumTRSolver, nlp, stats; callback, trace) -> TRResult

Adds `:full_batch_trajectory` (length `k`, `Bool`) to the expectation trace: the
iterations at which `N_k = M` and the step was therefore exact.

A run whose tail is all `true` has entered the deterministic regime and its final
`‖ĝ‖` is the real one. A run whose tail is all `false` never did, and its status is
a statement about one batch. Under a confirming mode `:sample_floor_from` is the first
iteration drawn under the floor that a refused stop or a stall raised, and 0 if no
batch was drawn under one, even when the rule itself drew the population.
"""
function SolverCore.solve!(solver::FiniteSumTRSolver,
                           nlp::FiniteSumNLP,
                           stats::GenericExecutionStats;
                           callback = (args...) -> nothing,
                           trace::Bool = false)
    c = solver.core; p = c.params
    _check_largest_batch(nlp, p)       # the oracle may not be the one the solver was built on
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
    stalled = false                    # the step on the batch in hand has just stalled
    while true
        if _first_order_ok(st, p)
            confirmed = stalled ? _confirm_stalled(nlp, c.x, Float64(st.g_norm), p) :
                                  _confirm_stop(nlp, c.x, Float64(st.g_norm), p)
            if confirmed
                set_status!(stats, _final_status(p)); break
            end
            # A refusal, which only a confirming mode can produce.
            p.on_refusal === :cap && _escalate!(nlp)
        end
        stalled = false
        k >= p.max_iterations    && (set_status!(stats, :max_iter);  break)
        time() - t0 > p.max_time && (set_status!(stats, :max_time); break)
        k += 1

        resample!(nlp, k, Float64(st.Δ), Float64(st.g_norm))
        _refresh_state!(c, nlp, st)
        update_variances!(nlp, c.x, c.g)

        _tr_step!(c, nlp, st)
        st.failed  && (set_status!(stats, :exception); k -= 1; break)
        # Traced before the break, unlike the :exception path above, which winds `k`
        # back instead. A stalled iteration produced a real ρ and a real step norm;
        # discarding them would leave the per-iteration trajectories one entry short
        # of `st.iter` and silently misalign them against the per-state ones.
        if st.stalled
            if confirms_stop(p) && _below_largest_batch(nlp)
                # Not final under a confirming mode. The iteration stays as a
                # rejected step, the run moves to the largest batch, and the stalled
                # batch is the next candidate, judged as a stalled batch.
                _escalate!(nlp)
                stalled = true
            else
                set_status!(stats, _stall_status(nlp, c.x, st, p))
                trace_post!(tr, st)
                sample_post!(sr, c, nlp, norm(true_gradient(nlp, c.x)))
                break
            end
        end
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
    _finish_floor!(stats, nlp, p, trace)
    if trace
        set_solver_specific!(stats, :full_batch_trajectory, copy(nlp.full_hist))
        set_solver_specific!(stats, :population, population(nlp.prob))
    end
    return stats
end
