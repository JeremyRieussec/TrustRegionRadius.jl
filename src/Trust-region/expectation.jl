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
    ExpectationTRSolver(nlp; rule, model, subsolver, params, x_ref,
                        true_curvature, hessian_norm)

Trust-region solver for an [`ExpectationNLP`](@ref).

The sampling rule lives on the oracle, not here — it decides what the *oracle*
returns, so it belongs with the oracle. So does the budget: `budget` is a keyword of
[`ExpectationNLP`](@ref), not of this constructor. It is a safety cap on `N_k` applied
on top of the rule's own `N_max`; the accuracy theory imposes none, so without a cap a
criticality-anchored mechanism will ask for `Θ(Δ_k^{-4})` objective terms and get it.

`TRParams(true_stop = :full)` and `:both` require `has_truth(prob)`. That is `true`
for [`PerturbedExpectation`](@ref), which supplies the exact mean by construction,
and `false` for a general expectation — where the request is not conservative but
impossible, and is refused rather than silently downgraded to the batch test.

`:test` is available on every expectation, since it consults only `σ_g` and `N_k`,
both of which the oracle already maintains. Its verdict is only as good as that
variance estimate. [`TRParams`](@ref) describes the batch on which the bound fails
and the minimum batch size, `stop_N_min`, that guards against it.
"""
mutable struct ExpectationTRSolver{T, V, R, M, S} <: AbstractTRSolver
    core::TRCore{T, V, R, M, S}
end

function ExpectationTRSolver(nlp::ExpectationNLP;
                             rule::RadiusRule            = RDelta(),
                             model::ModelHessian         = ExactHessian(),
                             subsolver::SubproblemSolver = SteihaugCG(),
                             params::TRParams            = TRParams{Float64}(),
                             x_ref::Union{Nothing, AbstractVector} = nothing,
                             true_curvature::Bool        = false,
                             hessian_norm::Bool          = false)
    confirmation_needs_truth(params) && !has_truth(nlp) && throw(ArgumentError(
        "TRParams(true_stop = :$(params.true_stop)) needs an exact gradient, but " *
        "$(nameof(typeof(nlp.prob))) is an expectation with no closed-form mean. " *
        "Use true_stop = :test, which needs only the variance estimate, or " *
        "PerturbedExpectation, or a finite sum, or true_stop = :none — the batch " *
        "test is then the only test available and should be reported as such."))
    c = TRCore(nlp, rule, model, subsolver, params;
               x_ref = x_ref, true_curvature = true_curvature,
               hessian_norm = hessian_norm)
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
| `:sigma_g2_trajectory` | `k` | `σ̂_g²` of the batch that produced the step |
| `:sigma_f2_trajectory` | `k` | `σ̂_f²` likewise |
| `:paired_decrease_trajectory` | `k` | `δ̂_N`, under a rule that pairs |
| `:paired_variance_trajectory` | `k` | `σ̂_N²`, likewise |

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
    tr = TRTrace(trace, c.want_curv)
    sr = SampleTrace(trace, trace && truth)
    trace_pre!(tr, st)
    sample_pre!(sr, truth ? norm(true_gradient(nlp, c.x)) : NaN)

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

        # A fresh draw. The incumbent f and g belong to the previous batch, so they
        # are re-evaluated before anything is compared against them.
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
            set_status!(stats, :stalled)
            trace_post!(tr, st)
            sample_post!(sr, c, nlp, truth ? norm(true_gradient(nlp, c.x)) : NaN)
            break
        end
        record_prediction!(nlp, Float64(st.predicted))

        trace_post!(tr, st)
        sample_post!(sr, c, nlp, truth ? norm(true_gradient(nlp, c.x)) : NaN)
        _record!(stats, c, st, k, t0)
        callback(nlp, solver, stats)
        stats.status == :user && break
    end

    _finish!(stats, c, st, k, t0)
    attach_trace!(stats, tr, st.Δ)
    attach_sample_trace!(stats, sr)
    _attach_sampling!(stats, nlp, trace)
    return stats
end

"""
    _confirm_stop(nlp, x, g_norm, params) -> Bool

Whether a satisfied stopping test should be believed, under the mode named by
`params.true_stop`. Called only once `‖ĝ_k‖ ≤ tol` already holds, so `g_norm ≤ tol`
on entry.

# The three modes

`:test` is free. `σ_g/√N_k` is the plug-in standard error of the batch gradient, so

    ‖ĝ_k‖ + z·σ_g/√N_k ≤ tol

is a one-sided confidence statement that the *true* `‖∇f‖` is below `tol`, valid to
the extent that `σ_g`, estimated on that same batch, is close to the population
value. When the scores of the batch cancel in the mean while each of them stays
large, `σ_g` is large and the bound cannot be met however small `‖ĝ_k‖` happens to
be. That is the case the unguarded test gets wrong, and `:test` refuses it. When the
scores of the batch have each vanished, `σ_g` vanishes with `‖ĝ_k‖` and the bound
passes whatever `∇f` is doing. [`TRParams`](@ref) gives a measured instance. For that
reason the bound is read only when `_batch_can_conclude` allows it, on a batch of at
least `params.stop_N_min` observations. On a smaller batch `:test`
refuses. The mode can therefore withhold a correct stop until the sampling rule has
grown `N_k`.

`:full` is exact and unconditional, through [`confirm_gradient_norm!`](@ref) at the
cap. It is the only mode that does not depend on the batch.

`:both` tries the cheap bound first and pays for the cap-sized pass when the bound
fails or may not be read. It costs fewer evaluations than `:full` whenever the bound
settles a candidate. It does not carry the same guarantee, because it accepts
whatever `:test` accepts.

!!! note "Screening on `‖∇f‖ = 0` would be the wrong screen"
    The natural-sounding alternative is to test `H₀: ∇f = 0` and reject the stop
    when `H₀` is refuted. That is not the hypothesis the solver cares about. A
    well-resolved iterate with `‖ĝ_k‖` comfortably below `tol` refutes `∇f = 0` —
    the gradient is significantly non-zero and significantly small — and blocking
    there would refuse every correct stop the cheap path is meant to accept. The
    test above screens on `‖∇f‖ ≤ tol`, which is the stopping criterion itself.
"""
_confirm_stop(::AbstractNLPModel, x, g_norm, ::TRParams) = true

function _confirm_stop(nlp::SampledNLP, x, g_norm, p::TRParams)
    mode = p.true_stop
    mode === :none && return true
    mode === :full && return confirm_gradient_norm!(nlp, x) <= p.tol

    resolved = _batch_can_conclude(nlp, p) &&
               g_norm + p.stop_z * grad_standard_error(nlp) <= p.tol
    mode === :test && return resolved
    return resolved || confirm_gradient_norm!(nlp, x) <= p.tol   # :both
end

"""
    _batch_can_conclude(nlp, params) -> Bool

Whether the current gradient batch is large enough for the statistical bound of
`:test` and `:both` to be read. It is when the batch holds at least
`params.stop_N_min` observations, or when it is the largest batch the oracle allows
and that cap is smaller.

A plug-in variance says nothing about the observations outside the batch. On a
batch of two whose scores have both vanished it is zero, and the bound then passes
at a point that is not critical. The minimum does not remove that possibility. It
makes it require `stop_N_min` vanished scores in one draw.
"""
_batch_can_conclude(nlp::SampledNLP, p::TRParams) =
    nlp.Ng >= min(p.stop_N_min, population_cap(nlp))

function _attach_sampling!(stats, nlp::SampledNLP, trace::Bool)
    trace || return nothing
    set_solver_specific!(stats, :grad_sample_trajectory, copy(nlp.Ng_hist))
    set_solver_specific!(stats, :obj_sample_trajectory,  copy(nlp.Nf_hist))
    set_solver_specific!(stats, :samples_total,   samples_used(nlp).total)
    set_solver_specific!(stats, :sample_cap_hits, nlp.capped)
    set_solver_specific!(stats, :problem_class,   problem_class(nlp))
    return nothing
end
