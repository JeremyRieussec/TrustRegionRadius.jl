# =============================================================================
# src/Trust-region/solver.jl
#
# The JSO-compatible solver.
#
# One parameter type (`TRParams`) and one result type (`TRResult`, an alias of
# the JSO `GenericExecutionStats`), rather than two parallel sets of names.
#
# Three orthogonal axes are threaded through the loop:
#   rule      :: RadiusRule            -- how Δ moves
#   model     :: ModelHessian          -- what curvature the model reports
#   subsolver :: SubproblemSolver      -- how the subproblem is solved
# =============================================================================

# -----------------------------------------------------------------------------
# TRParams
# -----------------------------------------------------------------------------

"""
    TRParams{T}(; η, η₁, η₂, Δ₀, Δmax, max_iterations, tol, max_time)

Solver parameters, parametric on the element type so `Float32` and `BigFloat`
models keep their precision.

# Fields
- `η`:   **acceptance** threshold; the step is taken when `ρ ≥ η`  (default `η₁`)
- `η₁`:  first **scaling** threshold, passed to the radius rule    (default 0.1)
- `η₂`:  second scaling threshold, "very successful"               (default 0.9)
- `Δ₀`:  initial radius; ignored by rules of the form `Δ = μ‖g‖`   (default 1)
- `Δmax`: hard cap on the radius                                  (default `Inf`)
- `max_iterations`:                                               (default 10 000)
- `tol`: first-order tolerance on `‖g‖`                           (default `√eps`)
- `tol_H`: second-order tolerance on `λ_min(B)`; `-1` disables it  (default `-1`)
- `max_time`: wall-clock budget in seconds                        (default `Inf`)

The thresholds satisfy `0 ≤ η ≤ η₁ ≤ η₂ < 1`.

# Acceptance is decoupled from scaling

`η` governs whether the trial point is taken; `η₁` and `η₂` govern only how the
radius is scaled. Taking `η < η₁` opens a middle regime, `ρ ∈ [η, η₁)`, in which
the step is accepted and the radius nonetheless contracts — a combination the
coupled formulation cannot express. `η = 0` is admissible and accepts every
non-increase in `f`; it is covered by the first-order framework of Part I but
not by the Curtis-Scheinberg analysis, so it is worth a column of its own in any
comparison.

`η` defaults to `η₁`, which reproduces the coupled behaviour exactly. Pass `η`
explicitly to decouple.

!!! note "Hold these fixed across mechanisms"
    In any comparison the thresholds and factors must be identical for every
    rule. Tuning them per rule measures tuning effort rather than algorithmic
    merit and makes a performance profile uninterpretable. The one thing that
    may legitimately vary is `η`, because varying it *is* an experiment.
"""
struct TRParams{T}
    η::T
    η₁::T
    η₂::T
    Δ₀::T
    Δmax::T
    max_iterations::Int
    tol::T
    tol_H::T
    max_time::Float64

    function TRParams{T}(; η₁::Real = T(0.1),
                           η₂::Real = T(0.9),
                           η::Union{Real, Nothing} = nothing,
                           Δ₀::Real = T(1),
                           Δmax::Real = T(Inf),
                           max_iterations::Int = 10_000,
                           tol::Real = sqrt(eps(T)),
                           tol_H::Real = T(-1),
                           max_time::Real = Inf) where {T}
        ηa = η === nothing ? η₁ : η
        0 <= ηa <= η₁ <= η₂ < 1 || throw(ArgumentError(
            "TRParams: need 0 ≤ η ≤ η₁ ≤ η₂ < 1, got η = $ηa, η₁ = $η₁, η₂ = $η₂"))
        Δ₀ > 0     || throw(ArgumentError("TRParams: need Δ₀ > 0, got $Δ₀"))
        Δmax >= Δ₀ || throw(ArgumentError("TRParams: need Δmax ≥ Δ₀"))
        max_iterations > 0 || throw(ArgumentError("TRParams: need max_iterations > 0"))
        tol > 0    || throw(ArgumentError("TRParams: need tol > 0"))
        tol_H == -1 || tol_H > 0 || throw(ArgumentError(
            "TRParams: need tol_H > 0 to request the second-order test, " *
            "or tol_H = -1 to disable it; got $tol_H"))
        new{T}(T(ηa), T(η₁), T(η₂), T(Δ₀), T(Δmax), max_iterations,
               T(tol), T(tol_H), Float64(max_time))
    end
end

TRParams(; kwargs...) = TRParams{Float64}(; kwargs...)

function Base.show(io::IO, p::TRParams{T}) where {T}
    println(io, "TRParams{$T}:")
    println(io, "  η  = ", p.η, "   (acceptance)")
    println(io, "  η₁ = ", p.η₁, ",  η₂ = ", p.η₂, "   (scaling)")
    p.η < p.η₁ && println(io, "  acceptance decoupled from scaling: ρ ∈ [",
                          p.η, ", ", p.η₁, ") accepts but contracts")
    println(io, "  Δ₀ = ", p.Δ₀, ",  Δmax = ", p.Δmax)
    println(io, "  max_iterations = ", p.max_iterations)
    println(io, "  tol = ", p.tol, ",  max_time = ", p.max_time)
    p.tol_H > 0 ?
        println(io, "  tol_H = ", p.tol_H, "   (second-order test: λ_min ≥ −tol_H)") :
        println(io, "  tol_H disabled: stops at ‖g‖ ≤ tol, whatever the curvature")
end

"""
    TRResult

The result type: an alias of JSO's `GenericExecutionStats`, so anything in the
JSO ecosystem (`SolverBenchmark`, `solve!`, callbacks) accepts it directly.

Useful fields: `status`, `solution`, `objective`, `dual_feas`, `iter`,
`elapsed_time`. Evaluation counts live on the model — `neval_obj(nlp)`,
`neval_grad(nlp)`, `neval_hprod(nlp)` — not on the result.

Per-iteration trajectories are attached under `stats.solver_specific` when
`trace = true`; see [`tr_solve`](@ref).
"""
const TRResult = GenericExecutionStats

# -----------------------------------------------------------------------------
# TRSolver
# -----------------------------------------------------------------------------

"""
    TRSolver{T, V, R, M, S} <: AbstractOptimizationSolver

Trust-region solver with pre-allocated workspace.

Type parameters cover the element type `T`, vector type `V`, and the three
axes `R <: RadiusRule`, `M <: ModelHessian`, `S <: SubproblemSolver`, so every
dispatch in the loop is resolved at compile time.

`rule`, `model` and `subsolver` are deep-copied at construction: all three may
carry mutable state (μ, quasi-Newton operators), and the solver owns its copy
so a single configuration object can be reused across problems.
"""
mutable struct TRSolver{T, V <: AbstractVector{T},
                        R <: RadiusRule,
                        M <: ModelHessian,
                        S <: SubproblemSolver} <: AbstractOptimizationSolver
    x::V
    g::V
    g_old::V
    s::V
    x_cand::V
    Hs::V
    Hs_new::V           # H_{k+1}·s_k, only allocated for retrospective rules
    rule::R
    model::M
    subsolver::S
    params::TRParams{T}
end

"""
    TRSolver(nlp; rule, model, subsolver, params)

Allocate a solver for `nlp`.

# Keyword arguments
- `rule`:      [`RadiusRule`](@ref), default [`RDelta`](@ref)
- `model`:     [`ModelHessian`](@ref), default [`ExactHessian`](@ref)
- `subsolver`: [`SubproblemSolver`](@ref), default [`SteihaugCG`](@ref)
- `params`:    [`TRParams`](@ref)
"""
function TRSolver(nlp::AbstractNLPModel{T, V};
                  rule::RadiusRule           = RDelta(),
                  model::ModelHessian        = ExactHessian(),
                  subsolver::SubproblemSolver = SteihaugCG(),
                  params::TRParams           = TRParams{T}()) where {T, V}
    n      = nlp.meta.nvar
    x      = similar(nlp.meta.x0)
    g      = similar(nlp.meta.x0)
    g_old  = similar(nlp.meta.x0)
    s      = similar(nlp.meta.x0)
    x_cand = similar(nlp.meta.x0)
    Hs     = similar(nlp.meta.x0)
    rule_c = deepcopy(rule)
    mod_c  = deepcopy(model)
    sub_c  = deepcopy(subsolver)
    # Some rules need more of (η, η₁, η₂) than the ordering: the step-driven
    # ones require η₁ > 0. Check once, here, rather than per iteration.
    validate_thresholds(rule_c, params.η, params.η₁, params.η₂)
    # Only retrospective rules need the second Hessian-vector buffer.
    Hs_new = needs_retrospective(rule_c) ? similar(nlp.meta.x0) : similar(nlp.meta.x0, 0)
    reset_model!(mod_c, n)
    return TRSolver{T, V, typeof(rule_c), typeof(mod_c), typeof(sub_c)}(
        x, g, g_old, s, x_cand, Hs, Hs_new, rule_c, mod_c, sub_c, params)
end

function SolverCore.reset!(solver::TRSolver)
    reset_rule!(solver.rule)
    reset_model!(solver.model, length(solver.x))
    return solver
end

SolverCore.reset!(solver::TRSolver, ::AbstractNLPModel) = SolverCore.reset!(solver)

# -----------------------------------------------------------------------------
# solve!
# -----------------------------------------------------------------------------

"""
    solve!(solver, nlp, stats; callback, trace) -> TRResult

In-place JSO solve; `stats` is mutated throughout.

`callback(nlp, solver, stats)` runs at the end of every iteration; set
`stats.status = :user` inside it to stop.

With `trace = true` the per-iteration trajectories are collected into
`stats.solver_specific`:

| key                      | length | entry `i`                          |
|:-------------------------|:-------|:-----------------------------------|
| `:delta_trajectory`      | `k+1`  | `Δ_{i-1}`, starting at `Δ_0`       |
| `:grad_trajectory`       | `k+1`  | `‖g_{i-1}‖`                        |
| `:obj_trajectory`        | `k+1`  | `f_{i-1}`                          |
| `:ratio_trajectory`      | `k`    | `ρ_i`                              |
| `:step_trajectory`       | `k`    | `‖s_i‖`                            |
| `:active_trajectory`     | `k`    | `‖s_i‖ = Δ_i`                      |
| `:accepted_trajectory`   | `k`    | `ρ_i ≥ η`                          |
| `:lambda_min_trajectory` | `k+1`  | `λ_min(B_{i-1})`                   |
| `:tau_trajectory`        | `k+1`  | `τ_{i-1}`, the measure the rule saw |

The last two appear only when a curvature estimate was computed at all, i.e. when
the rule is wrapped in `SecondOrder` or `tol_H > 0`. Comparing them is the cheapest
second-order diagnostic there is: wherever `τ > ‖g‖` the run is somewhere the
gradient alone calls critical and the curvature does not.

The first three are one entry longer than the rest, since they have a value
before the first iteration; align on the tail when plotting them together.

Two of these are worth recording even when nothing else is. The activity flag
decides whether the constraint eventually stops binding, which is the
distinction between mechanisms that a first-order convergence test cannot see.
The acceptance flag cannot be reconstructed from `:ratio_trajectory` once
`η < η₁`, because the rule's own thresholds no longer coincide with the one that
decided the step; it is the only record of which iterations belong to `𝒮`.

# Statuses
- `:first_order` — `‖g‖ ≤ tol`
- `:max_iter`    — iteration budget exhausted
- `:max_time`    — wall-clock budget exhausted
- `:stalled`     — the step fell below the level at which `f(x) − f(x+s)`
                   carries information, so ρ is numerical noise
- `:exception`   — the model could not be built at the current iterate
- `:user`        — stopped by the callback
"""
function SolverCore.solve!(solver::TRSolver{T, V, R, M, S},
                           nlp::AbstractNLPModel{T, V},
                           stats::GenericExecutionStats{T, V};
                           callback = (args...) -> nothing,
                           trace::Bool = false) where {T, V, R, M, S}

    p = solver.params
    reset_rule!(solver.rule)
    reset_model!(solver.model, nlp.meta.nvar)
    nlp isa SampledNLP && reset_sampling!(nlp)
    retro = needs_retrospective(solver.rule)

    t0 = time()
    copyto!(solver.x, nlp.meta.x0)
    f = obj(nlp, solver.x)
    grad!(nlp, solver.x, solver.g)
    g_norm = norm(solver.g)

    # λ_min(B_k) is needed by a τ-anchored rule, by the second-order stopping
    # test, or by neither — in which case it is never computed. `crit` is what
    # the rule measures its radius against: ‖g_k‖ for a first-order rule,
    # τ_k = max{‖g_k‖, −λ_min} for one wrapped in `SecondOrder`.
    want_curv = needs_curvature(solver.rule) || p.tol_H > 0
    λmin = want_curv ? T(lambda_min_estimate(solver.model, nlp, solver.x)) : T(NaN)
    crit = T(criticality(solver.rule, Float64(g_norm),
                         want_curv ? Float64(λmin) : 0.0))

    Δ = T(initial_radius(solver.rule, Float64(p.Δ₀), Float64(crit)))
    Δ = min(Δ, p.Δmax)

    Δ_tr = trace ? Float64[Δ]      : Float64[]
    g_tr = trace ? Float64[g_norm] : Float64[]
    f_tr = trace ? Float64[f]      : Float64[]
    λ_tr = (trace && want_curv) ? Float64[λmin] : Float64[]
    τ_tr = (trace && want_curv) ? Float64[crit] : Float64[]
    ρ_tr = Float64[]
    s_tr = Float64[]
    a_tr = Bool[]
    acc_tr = Bool[]

    set_iter!(stats, 0)
    set_objective!(stats, f)
    set_dual_residual!(stats, g_norm)
    set_solution!(stats, solver.x)
    set_status!(stats, :unknown)

    k = 0
    while true
        if g_norm <= p.tol
            # With tol_H set, a critical point is only a stopping point when the
            # model reports no usable negative curvature; otherwise the run
            # continues and the anchored rules see τ = −λ_min > 0.
            if p.tol_H <= 0 || λmin >= -p.tol_H
                set_status!(stats, p.tol_H > 0 ? :second_order : :first_order); break
            end
        end
        if k >= p.max_iterations
            set_status!(stats, :max_iter); break
        end
        if time() - t0 > p.max_time
            set_status!(stats, :max_time); break
        end
        k += 1

        # --- resample, if the model is stochastic ---
        # `prepare_iteration!` is a no-op for an ordinary NLP. For a SampledNLP it
        # chooses N_k from the sampling rule and draws the batch, after which the
        # incumbent f and g belong to the previous batch and must be recomputed:
        # ared and pred have to be formed from the same realisations, or ρ̂ is
        # dominated by the difference between two batches rather than by the step.
        if prepare_iteration!(nlp, k, Float64(Δ), Float64(g_norm))
            f = obj(nlp, solver.x)
            grad!(nlp, solver.x, solver.g)
            g_norm = norm(solver.g)
            update_variances!(nlp, solver.x)
            if want_curv
                λmin = T(lambda_min_estimate(solver.model, nlp, solver.x))
            end
            crit = T(criticality(solver.rule, Float64(g_norm),
                                 want_curv ? Float64(λmin) : 0.0))
        end

        # --- subproblem ---
        local active::Bool
        try
            active = solve_subproblem!(solver.subsolver, solver.model, nlp,
                                       solver.x, solver.g, Δ, solver.s, solver.Hs)
        catch err
            err isa DomainError || rethrow()
            set_status!(stats, :exception); k -= 1; break
        end
        s_norm = norm(solver.s)

        # --- ratio ---
        @. solver.x_cand = solver.x + solver.s
        f_cand = obj(nlp, solver.x_cand)
        model_hprod!(solver.model, nlp, solver.x, solver.s, solver.Hs)
        predicted = -dot(solver.g, solver.s) - T(0.5) * dot(solver.s, solver.Hs)
        actual = f - f_cand
        ρ = (isfinite(f_cand) && predicted > 0) ? actual / predicted : T(-Inf)

        # Hand the predicted reduction back to the sampling rule. Only
        # SequentialEstimation reads it — it sizes the next batch so the noise in
        # the estimated decrease stays small beside the decrease the model claims.
        record_prediction!(nlp, Float64(predicted))

        # Stagnation: |actual| at the rounding level of f and a numerically nil
        # step. ρ is then noise, every step is rejected, and the radius
        # collapses for ever; report it rather than spin.
        if abs(actual) <= eps(T) * max(one(T), abs(f)) && s_norm < eps(T)^T(0.75)
            set_status!(stats, :stalled); break
        end

        g_norm_old = g_norm
        crit_old = crit
        # Acceptance is decided by η alone. The rule receives η₁ and η₂ and
        # scales the radius on its own reading of ρ, so an accepted step with
        # ρ ∈ [η, η₁) still contracts.
        accepted = ρ >= p.η

        if accepted
            copyto!(solver.g_old, solver.g)
            copyto!(solver.x, solver.x_cand)
            f = f_cand          # same batch as ared: consistent within the iteration
            grad!(nlp, solver.x, solver.g)
            g_norm = norm(solver.g)
            # quasi-Newton update with y_k = g_{k+1} − g_k
            @. solver.g_old = solver.g - solver.g_old
            update_model!(solver.model, solver.s, solver.g_old)
        end

        # The curvature belongs to the model at the *new* iterate, so it is
        # re-estimated after the model has been updated — on a rejected step
        # neither has moved and the previous value still stands.
        if want_curv && accepted
            λmin = T(lambda_min_estimate(solver.model, nlp, solver.x))
        end
        crit = T(criticality(solver.rule, Float64(g_norm),
                             want_curv ? Float64(λmin) : 0.0))

        # --- the ratio that drives the radius ---
        # Retrospective rules judge the NEW model on the SAME step, so they need
        # H_{k+1}·s_k. On a rejected step the model has not changed and ρ̃ ≡ ρ.
        ρ_rule = ρ
        if retro && accepted
            model_hprod!(solver.model, nlp, solver.x, solver.s, solver.Hs_new)
            ρ_rule = retrospective_ratio(actual, solver.s, solver.g, solver.Hs_new)
        end

        Δ = T(update_radius!(solver.rule, Float64(Δ), Float64(ρ_rule), accepted,
                             Float64(p.η₁), Float64(p.η₂),
                             Float64(s_norm), Float64(crit_old), Float64(crit)))
        Δ = min(Δ, p.Δmax)

        if trace
            push!(Δ_tr, Δ); push!(g_tr, g_norm); push!(f_tr, f)
            want_curv && (push!(λ_tr, Float64(λmin)); push!(τ_tr, Float64(crit)))
            push!(ρ_tr, ρ); push!(s_tr, s_norm); push!(a_tr, active)
            push!(acc_tr, accepted)
        end

        set_iter!(stats, k)
        set_objective!(stats, f)
        set_dual_residual!(stats, g_norm)
        set_solution!(stats, solver.x)
        set_time!(stats, time() - t0)

        callback(nlp, solver, stats)
        stats.status == :user && break
    end

    set_solution!(stats, solver.x)
    set_objective!(stats, f)
    set_dual_residual!(stats, g_norm)
    set_iter!(stats, k)
    set_time!(stats, time() - t0)

    if trace
        set_solver_specific!(stats, :delta_trajectory,  Δ_tr)
        set_solver_specific!(stats, :grad_trajectory,   g_tr)
        set_solver_specific!(stats, :obj_trajectory,    f_tr)
        set_solver_specific!(stats, :ratio_trajectory,  ρ_tr)
        set_solver_specific!(stats, :step_trajectory,   s_tr)
        set_solver_specific!(stats, :active_trajectory, a_tr)
        set_solver_specific!(stats, :accepted_trajectory, acc_tr)
        if nlp isa SampledNLP
            set_solver_specific!(stats, :grad_sample_trajectory, copy(nlp.Ng_hist))
            set_solver_specific!(stats, :obj_sample_trajectory,  copy(nlp.Nf_hist))
            set_solver_specific!(stats, :samples_total, samples_used(nlp).total)
            set_solver_specific!(stats, :sample_cap_hits, nlp.capped)
        end
        if want_curv
            set_solver_specific!(stats, :lambda_min_trajectory, λ_tr)
            set_solver_specific!(stats, :tau_trajectory,        τ_tr)
        end
        set_solver_specific!(stats, :final_delta,       Float64(Δ))
    end
    return stats
end

function SolverCore.solve!(solver::TRSolver{T, V}, nlp::AbstractNLPModel{T, V};
                           kwargs...) where {T, V}
    stats = GenericExecutionStats(nlp)
    return solve!(solver, nlp, stats; kwargs...)
end

# -----------------------------------------------------------------------------
# Functional interface
# -----------------------------------------------------------------------------

"""
    tr_solve(nlp; rule, model, subsolver, params, trace, callback) -> TRResult

Solve `nlp` by the trust-region method. Constructs a [`TRSolver`](@ref) and
runs it.

```julia
using TrustRegionRadius, ADNLPModels

nlp = ADNLPModel(x -> (1 - x[1])^2 + 100(x[2] - x[1]^2)^2, [-1.2, 1.0])

stats = tr_solve(nlp;
    rule      = RGrad(),
    model     = SR1Model(mem = 5),
    subsolver = SteihaugCG(),
    params    = TRParams(tol = 1e-8),
    trace     = true)

stats.status                                        # :first_order
stats.solver_specific[:delta_trajectory]            # Δ per iteration
count(stats.solver_specific[:active_trajectory])    # iterations with ‖s‖ = Δ
```
"""
function tr_solve(nlp::AbstractNLPModel{T, V};
                  rule::RadiusRule            = RDelta(),
                  model::ModelHessian         = ExactHessian(),
                  subsolver::SubproblemSolver = SteihaugCG(),
                  params::TRParams            = TRParams{T}(),
                  kwargs...) where {T, V}
    solver = TRSolver(nlp; rule = rule, model = model,
                      subsolver = subsolver, params = params)
    return solve!(solver, nlp; kwargs...)
end
