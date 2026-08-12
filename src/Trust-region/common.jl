# =============================================================================
# src/Trust-region/common.jl
#
# What the three solvers share: the parameters, the workspace, and one
# iteration.
#
# The three loops differ only in what happens *before* the subproblem is solved
# and *after* the radius is updated — resampling, sample accounting, full-batch
# detection. Everything between is identical, so it lives here as `_tr_step!` and
# the three loops in deterministic.jl, expectation.jl and finitesum.jl stay short
# enough to read at a glance.
# =============================================================================

# -----------------------------------------------------------------------------
# TRParams
# -----------------------------------------------------------------------------

"""
    TRParams{T}(; η, η1, η2, Δ0, Δmin, Δmax, max_iterations, tol, tol_H,
                  max_time, true_stop, stop_z)

Solver parameters, shared by all three solvers.

# Fields
- `η`:   **acceptance** threshold; the step is taken when `ρ ≥ η`  (default `η1`)
- `η1`:  first **scaling** threshold, passed to the radius rule    (default 0.1)
- `η2`:  second scaling threshold, "very successful"               (default 0.9)
- `Δ0`:  initial radius; ignored by rules of the form `Δ = μ·crit` (default 1)
- `Δmin`, `Δmax`: hard floor and cap                               (default 0, `Inf`)
- `max_iterations`                                                 (default 10 000)
- `tol`: first-order tolerance on `‖g‖`                            (default `√eps`)
- `tol_H`: second-order tolerance on `λ_min(B)`; `-1` disables it   (default `-1`)
- `max_time`: wall-clock budget in seconds                         (default `Inf`)
- `true_stop`: how a satisfied stopping test is confirmed          (default `:none`)
- `stop_z`: confidence multiplier for the statistical modes        (default 2)

# Confirming the stopping test

`‖ĝ_k‖ ≤ tol` is a statement about **one batch**. On a small batch it can be met on
sampling noise alone, and then the solver reports `:first_order` at a point that is
not critical — a batch of `N = 2` with an exact model puts the step on the batch
minimiser, so `‖ĝ‖` is zero to rounding whatever `∇f` is doing. `true_stop` chooses
what has to hold before that stop is believed:

| value | cost per candidate stop | accepts the stop when |
|:--|:--|:--|
| `:none` | none | always — the batch test is the test |
| `:test` | none | `‖ĝ_k‖ + z·σ_g/√N_k ≤ tol` |
| `:full` | one cap-sized pass | `‖g‖ ≤ tol` on the largest batch allowed |
| `:both` | a pass only when `:test` is inconclusive | `:test`, else `:full` |

`:test` is free: it asks whether the batch can *resolve* `tol` at all, by putting a
one-sided confidence bound on `‖∇f‖` from the plug-in standard error `σ_g/√N_k`. It
never stops on noise, but it can refuse to stop for a long time when `N_k` is held
down by the sampling rule.

`:full` is exact and always pays. `:both` is the one to reach for: the cheap bound
settles the well-resolved case with no extra evaluations, and the expensive pass is
spent only when the batch genuinely cannot tell. See [`confirm_gradient_norm!`](@ref)
for what "the largest batch allowed" means and how its cost is accounted.

`true_stop = false` and `true_stop = true` remain accepted, and mean `:none` and
`:full`. A deterministic solver rejects anything but `:none` — the batch test *is*
the true test there. `:full` and `:both` on an expectation need [`has_truth`](@ref);
`:test` does not, since it consults only the variance the oracle already tracks.

!!! warning "The default is not the safe choice"
    `:none` reproduces the literature, and every benchmark result recorded in this
    repository was produced under it. It is the right default for a survey whose
    subject is the mechanisms, but a `:first_order` status obtained under it is a
    claim about a batch, not about `∇f`. Report which mode produced a number.

`0 ≤ η ≤ η1 ≤ η2 < 1`. `η₁`, `η₂` and `Δ₀` work as keywords and as property names
too: `TRParams(η₁ = 0.2).η1 == 0.2`.

# Acceptance is decoupled from scaling

`η` governs whether the trial point is taken; `η1` and `η2` govern only how the
radius is scaled, and are the only two a rule ever sees. `η < η1` opens a middle
band, `ρ ∈ [η, η1)`, in which the step is accepted and the radius nonetheless
contracts. `η` defaults to `η1`, reproducing the coupled behaviour exactly.

!!! note "Hold these fixed across mechanisms"
    In any comparison the thresholds and factors must be identical for every rule.
    Tuning them per rule measures tuning effort rather than algorithmic merit. `η`
    is the exception worth varying deliberately: it belongs to the algorithm rather
    than to any one mechanism.
"""
struct TRParams{T}
    η::T
    η1::T
    η2::T
    Δ0::T
    Δmin::T
    Δmax::T
    max_iterations::Int
    tol::T
    tol_H::T
    max_time::Float64
    true_stop::Symbol
    stop_z::T
end

"""
    _stop_mode(v) -> Symbol

Normalise the `true_stop` argument. `false`/`true` are kept as spellings of
`:none`/`:full` so existing scripts and the archived experiment configs still load.
"""
_stop_mode(b::Bool) = b ? :full : :none
function _stop_mode(s::Symbol)
    s in (:none, :test, :full, :both) || throw(ArgumentError(
        "TRParams: true_stop must be one of :none, :test, :full, :both " *
        "(or a Bool, where false = :none and true = :full), got :$s"))
    return s
end
"""
    confirms_stop(p) -> Bool

Whether `p` asks for any confirmation of a satisfied stopping test.
"""
@inline confirms_stop(p::TRParams) = p.true_stop !== :none

"""
    confirmation_needs_truth(p) -> Bool

Whether the confirmation mode can require an exact gradient. True for `:full` and
`:both`, false for `:test`, which reads only the variance the oracle already has.
"""
@inline confirmation_needs_truth(p::TRParams) = p.true_stop in (:full, :both)

@inline function _pick(name::String, ascii, unicode, default)
    ascii !== nothing && unicode !== nothing && ascii != unicode &&
        throw(ArgumentError("TRParams: $name given twice with different values"))
    ascii   !== nothing && return ascii
    unicode !== nothing && return unicode
    return default
end

function TRParams{T}(; η::Union{Real, Nothing} = nothing,
                       η1::Union{Real, Nothing} = nothing,
                       η2::Union{Real, Nothing} = nothing,
                       Δ0::Union{Real, Nothing} = nothing,
                       η₁::Union{Real, Nothing} = nothing,
                       η₂::Union{Real, Nothing} = nothing,
                       Δ₀::Union{Real, Nothing} = nothing,
                       Δmin::Real = T(0), Δmax::Real = T(Inf),
                       max_iterations::Int = 10_000,
                       tol::Real = sqrt(eps(T)), tol_H::Real = T(-1),
                       max_time::Real = Inf,
                       true_stop::Union{Bool, Symbol} = :none,
                       stop_z::Real = 2) where {T}
    e1 = _pick("η1", η1, η₁, T(0.1))
    e2 = _pick("η2", η2, η₂, T(0.9))
    d0 = _pick("Δ0", Δ0, Δ₀, T(1))
    ea = η === nothing ? e1 : η
    0 <= ea <= e1 <= e2 < 1 || throw(ArgumentError(
        "TRParams: need 0 ≤ η ≤ η1 ≤ η2 < 1, got η = $ea, η1 = $e1, η2 = $e2"))
    d0 > 0       || throw(ArgumentError("TRParams: need Δ0 > 0, got $d0"))
    Δmin >= 0    || throw(ArgumentError("TRParams: need Δmin ≥ 0, got $Δmin"))
    Δmin <= Δmax || throw(ArgumentError("TRParams: need Δmin ≤ Δmax"))
    Δmax >= d0   || throw(ArgumentError("TRParams: need Δmax ≥ Δ0"))
    max_iterations > 0 || throw(ArgumentError("TRParams: need max_iterations > 0"))
    tol > 0      || throw(ArgumentError("TRParams: need tol > 0"))
    tol_H == -1 || tol_H > 0 || throw(ArgumentError(
        "TRParams: need tol_H > 0 to request the second-order test, or " *
        "tol_H = -1 to disable it; got $tol_H"))
    mode = _stop_mode(true_stop)
    stop_z > 0 || throw(ArgumentError("TRParams: need stop_z > 0, got $stop_z"))
    return TRParams{T}(T(ea), T(e1), T(e2), T(d0), T(Δmin), T(Δmax),
                       max_iterations, T(tol), T(tol_H), Float64(max_time),
                       mode, T(stop_z))
end

TRParams(; kwargs...) = TRParams{Float64}(; kwargs...)

@inline function Base.getproperty(p::TRParams, s::Symbol)
    s === :η₁ && return getfield(p, :η1)
    s === :η₂ && return getfield(p, :η2)
    s === :Δ₀ && return getfield(p, :Δ0)
    return getfield(p, s)
end

Base.propertynames(::TRParams) =
    (:η, :η1, :η2, :Δ0, :Δmin, :Δmax, :max_iterations, :tol, :tol_H,
     :max_time, :true_stop, :stop_z, :η₁, :η₂, :Δ₀)

function Base.show(io::IO, p::TRParams{T}) where {T}
    println(io, "TRParams{$T}:")
    println(io, "  η  = ", p.η, "   (acceptance)")
    println(io, "  η1 = ", p.η1, ",  η2 = ", p.η2, "   (scaling)")
    p.η < p.η1 && println(io, "  decoupled: ρ ∈ [", p.η, ", ", p.η1,
                          ") accepts but contracts")
    println(io, "  Δ0 = ", p.Δ0, ",  Δmin = ", p.Δmin, ",  Δmax = ", p.Δmax)
    println(io, "  max_iterations = ", p.max_iterations, ",  tol = ", p.tol)
    p.tol_H > 0 ?
        println(io, "  tol_H = ", p.tol_H, "   (second-order test)") :
        println(io, "  tol_H disabled")
    if confirms_stop(p)
        println(io, "  true_stop = :", p.true_stop, "   (stopping test confirmed",
                p.true_stop === :full ? "" : ", z = $(p.stop_z)", ")")
    else
        println(io, "  true_stop disabled: ‖ĝ‖ ≤ tol is a claim about one batch")
    end
end

"""
    TRResult

An alias of JSO's `GenericExecutionStats`. Useful fields: `status`, `solution`,
`objective`, `dual_feas`, `iter`, `elapsed_time`. Evaluation counts live on the
model — `neval_obj(nlp)`, `neval_grad(nlp)`, `neval_hprod(nlp)`.
"""
const TRResult = GenericExecutionStats

# -----------------------------------------------------------------------------
# The shared core
# -----------------------------------------------------------------------------

"""
    AbstractTRSolver <: AbstractOptimizationSolver

Supertype of the three solvers. Which one applies is decided by the problem class,
not by a keyword:

| solver | problem | sampling rule |
|:--|:--|:--|
| [`DeterministicTRSolver`](@ref) | any `AbstractNLPModel`, [`FullBatchNLP`](@ref) | none — the constructor has no such argument |
| [`ExpectationTRSolver`](@ref) | [`ExpectationNLP`](@ref) | required, cap from `budget` |
| [`FiniteSumTRSolver`](@ref) | [`FiniteSumNLP`](@ref) | required, cap `min(budget, M)` |

[`tr_solve`](@ref) dispatches for you.
"""
abstract type AbstractTRSolver <: AbstractOptimizationSolver end

"""
    TRCore{T, V, R, M, S}

Everything the three solvers hold in common: the vector workspace, the three axes,
the parameters, and the four flags derived from them once at construction.

The type parameters cover the element type, the vector type, and the axes
`R <: RadiusRule`, `M <: ModelHessian`, `S <: SubproblemSolver`, so every dispatch
in the loop resolves at compile time.
"""
mutable struct TRCore{T, V <: AbstractVector{T}, R <: RadiusRule,
                      M <: ModelHessian, S <: SubproblemSolver}
    x::V; g::V; g_old::V; y::V; s::V; x_cand::V; Hs::V; Hs_new::V
    ws::SubWorkspace{V}
    rule::R
    model::M
    subsolver::S
    params::TRParams{T}
    retro::Bool
    want_curv::Bool
    want_vec::Bool
    sub_hs::Bool
    vmin::Vector{Float64}
end

function TRCore(nlp::AbstractNLPModel{T, V}, rule, model, subsolver,
                params::TRParams{T}) where {T, V}
    x0 = nlp.meta.x0
    rule_c = deepcopy(rule); mod_c = deepcopy(model); sub_c = deepcopy(subsolver)

    validate_thresholds(rule_c, params.η, params.η1, params.η2)
    check_model_problem(mod_c, nlp)

    let mt = model_eltype(mod_c)
        mt === nothing || mt === T || throw(ArgumentError(
            "TRSolver: the model was built for $mt but the problem has element " *
            "type $T. Construct it as $(nameof(typeof(mod_c)))(T = $T)."))
    end

    retro     = needs_retrospective(rule_c)
    want_vec  = needs_eigenvector(sub_c)
    want_curv = needs_curvature(rule_c) || params.tol_H > 0 || want_vec
    sub_hs    = returns_hprod(sub_c)

    if want_curv && !reports_negative_curvature(mod_c)
        @warn "τ ≡ ‖g‖: this model is positive (semi)definite by construction, so " *
              "λ_min(B) ≥ 0 always and the second-order machinery is a no-op. A " *
              ":second_order status from this run certifies nothing about ∇²f. " *
              "Use ExactHessian or SR1Model when the second-order question is the " *
              "question." model = nameof(typeof(mod_c)) maxlog = 1
    end

    reset_model!(mod_c, nlp.meta.nvar)
    Hs_new = retro ? similar(x0) : similar(x0, 0)

    return TRCore{T, V, typeof(rule_c), typeof(mod_c), typeof(sub_c)}(
        similar(x0), similar(x0), similar(x0), similar(x0), similar(x0),
        similar(x0), similar(x0), Hs_new, SubWorkspace(x0),
        rule_c, mod_c, sub_c, params, retro, want_curv, want_vec, sub_hs, Float64[])
end

"""
    TRState{T}

The per-iteration scalars, mutated in place by `_tr_step!` so the loops do
not have to thread a dozen return values.
"""
mutable struct TRState{T}
    f::T
    g_norm::T
    Δ::T
    crit::T
    λmin::Float64
    ρ::T
    ρ_rule::T
    predicted::T
    s_norm::T
    accepted::Bool
    active::Bool
    stalled::Bool
    failed::Bool
end

"""
    _init_state!(c, nlp) -> TRState

Evaluate `f`, `g`, the curvature if any rule wants it, and the initial radius.
"""
function _init_state!(c::TRCore{T}, nlp) where {T}
    reset_rule!(c.rule)
    reset_model!(c.model, nlp.meta.nvar)
    copyto!(c.x, nlp.meta.x0)
    f = obj(nlp, c.x)
    grad!(nlp, c.x, c.g)
    g_norm = norm(c.g)
    λmin = NaN
    if c.want_curv
        λmin, v = curvature_estimate(c.model, nlp, c.x, c.want_vec)
        c.vmin = v
    end
    crit = T(criticality(c.rule, Float64(g_norm), c.want_curv ? λmin : 0.0))
    Δ = clamp(T(initial_radius(c.rule, Float64(c.params.Δ0), Float64(crit))),
              c.params.Δmin, c.params.Δmax)
    return TRState{T}(f, g_norm, Δ, crit, λmin, T(0), T(0), T(0), T(0),
                      false, false, false, false)
end

"""
    _refresh_state!(c, nlp, st)

Re-evaluate `f`, `g`, the curvature and the criticality measure at the current
iterate, without moving it.

Called by the two sampled solvers after every resample: the incumbent values belong
to the *previous* batch, and `ared` and `pred` have to be formed from the same
realisations or ρ̂ is dominated by the difference between two batches rather than by
the step.
"""
function _refresh_state!(c::TRCore{T}, nlp, st::TRState{T}) where {T}
    st.f = obj(nlp, c.x)
    grad!(nlp, c.x, c.g)
    st.g_norm = norm(c.g)
    if c.want_curv
        st.λmin, c.vmin = curvature_estimate(c.model, nlp, c.x, c.want_vec)
    end
    st.crit = T(criticality(c.rule, Float64(st.g_norm), c.want_curv ? st.λmin : 0.0))
    return nothing
end

"""
    _tr_step!(c, nlp, st) -> nothing

One trust-region iteration: subproblem, ratio, acceptance, radius update.

Identical in all three regimes — which is the point of separating the solvers, since
what surrounds it is not. Sets `st.failed` on a `DomainError` from the model and
`st.stalled` when the achieved reduction has fallen to rounding, and leaves every
other field describing the iteration just taken.
"""
function _tr_step!(c::TRCore{T}, nlp, st::TRState{T}) where {T}
    p = c.params
    st.failed = false; st.stalled = false

    # --- subproblem ---
    try
        st.active = solve_subproblem!(c.subsolver, c.model, nlp, c.x, c.g, st.Δ,
                                      c.s, c.Hs, c.ws;
                                      curv = c.want_curv ? (st.λmin, c.vmin) : nothing)
    catch err
        err isa DomainError || rethrow()
        st.failed = true; return nothing
    end
    st.s_norm = norm(c.s)

    # --- ratio ---
    @. c.x_cand = c.x + c.s
    f_cand = obj(nlp, c.x_cand)
    # CG accumulated B·s while building the step; only a subsolver that did not
    # declare `returns_hprod` needs it computed here.
    c.sub_hs || model_hprod!(c.model, nlp, c.x, c.s, c.Hs)
    st.predicted = -dot(c.g, c.s) - T(0.5) * dot(c.s, c.Hs)
    actual = st.f - f_cand
    st.ρ = (isfinite(f_cand) && st.predicted > 0) ? actual / st.predicted : T(-Inf)

    # Stagnation: |actual| at the rounding level of f and a numerically nil step.
    # ρ is then noise, every step is rejected, and the radius collapses for ever.
    if abs(actual) <= eps(T) * max(one(T), abs(st.f)) && st.s_norm < eps(T)^T(0.75)
        st.stalled = true; return nothing
    end

    crit_old = st.crit
    st.accepted = st.ρ >= p.η          # acceptance is decided by η alone

    if st.accepted
        copyto!(c.g_old, c.g)
        copyto!(c.x, c.x_cand)
        st.f = f_cand                  # same batch as ared: consistent within the step
        grad!(nlp, c.x, c.g)
        st.g_norm = norm(c.g)
        @. c.y = c.g - c.g_old         # y_k in its own buffer, so g_old survives
        update_model!(c.model, c.s, c.y)
    end

    # The curvature belongs to the model at the *new* iterate; on a rejected step
    # neither has moved and the previous value still stands.
    if c.want_curv && st.accepted
        st.λmin, c.vmin = curvature_estimate(c.model, nlp, c.x, c.want_vec)
    end
    st.crit = T(criticality(c.rule, Float64(st.g_norm), c.want_curv ? st.λmin : 0.0))

    # --- the ratio that drives the radius ---
    st.ρ_rule = st.ρ
    if c.retro && st.accepted
        model_hprod!(c.model, nlp, c.x, c.s, c.Hs_new)
        st.ρ_rule = retrospective_ratio(actual, c.s, c.g, c.Hs_new)
    end

    st.Δ = clamp(T(update_radius!(c.rule, Float64(st.Δ), Float64(st.ρ_rule),
                                  st.accepted, Float64(p.η1), Float64(p.η2),
                                  Float64(st.s_norm), Float64(crit_old),
                                  Float64(st.crit))),
                 p.Δmin, p.Δmax)
    return nothing
end

"""
    _first_order_ok(st, p) -> Bool

The stopping test on the model's own quantities: `‖g‖ ≤ tol`, and `λ_min ≥ −tol_H`
when the second-order test is on.
"""
@inline _first_order_ok(st::TRState, p::TRParams) =
    st.g_norm <= p.tol && (p.tol_H <= 0 || st.λmin >= -p.tol_H)

"""
    _final_status(p) -> Symbol

`:second_order` when the second-order test was requested and met, `:first_order`
otherwise.

`:second_order` is not one of SolverCore's built-in statuses; the package registers it
with `SolverCore.STATUSES` in `__init__`. Without that registration `set_status!`
rejects it, and the failure lands on the one path that succeeded.
"""
@inline _final_status(p::TRParams) = p.tol_H > 0 ? :second_order : :first_order

# -----------------------------------------------------------------------------
# Tracing
# -----------------------------------------------------------------------------

"""
    TRTrace(on, want_curv, want_truth)

The per-iteration trajectories, collected only when `trace = true`.

`Δ`, `‖g‖` and `f` have a value before the first iteration and so are one entry
longer than the rest; align on the tail when plotting them together.
"""
mutable struct TRTrace
    on::Bool
    curv::Bool
    truth::Bool
    Δ::Vector{Float64}
    g::Vector{Float64}
    f::Vector{Float64}
    λ::Vector{Float64}
    τ::Vector{Float64}
    tg::Vector{Float64}
    ρ::Vector{Float64}
    s::Vector{Float64}
    active::Vector{Bool}
    accepted::Vector{Bool}
end

TRTrace(on::Bool, curv::Bool, truth::Bool) =
    TRTrace(on, curv, truth, Float64[], Float64[], Float64[], Float64[], Float64[],
            Float64[], Float64[], Float64[], Bool[], Bool[])

function trace_pre!(tr::TRTrace, st::TRState, truth_val)
    tr.on || return nothing
    push!(tr.Δ, st.Δ); push!(tr.g, st.g_norm); push!(tr.f, st.f)
    tr.curv && (push!(tr.λ, st.λmin); push!(tr.τ, Float64(st.crit)))
    tr.truth && push!(tr.tg, truth_val)
    return nothing
end

function trace_post!(tr::TRTrace, st::TRState, truth_val)
    tr.on || return nothing
    trace_pre!(tr, st, truth_val)
    push!(tr.ρ, st.ρ); push!(tr.s, st.s_norm)
    push!(tr.active, st.active); push!(tr.accepted, st.accepted)
    return nothing
end

function attach_trace!(stats, tr::TRTrace, Δ_final)
    tr.on || return nothing
    set_solver_specific!(stats, :delta_trajectory,    tr.Δ)
    set_solver_specific!(stats, :grad_trajectory,     tr.g)
    set_solver_specific!(stats, :obj_trajectory,      tr.f)
    set_solver_specific!(stats, :ratio_trajectory,    tr.ρ)
    set_solver_specific!(stats, :step_trajectory,     tr.s)
    set_solver_specific!(stats, :active_trajectory,   tr.active)
    set_solver_specific!(stats, :accepted_trajectory, tr.accepted)
    if tr.curv
        set_solver_specific!(stats, :lambda_min_trajectory, tr.λ)
        set_solver_specific!(stats, :tau_trajectory,        tr.τ)
    end
    tr.truth && set_solver_specific!(stats, :true_grad_trajectory, tr.tg)
    set_solver_specific!(stats, :final_delta, Float64(Δ_final))
    return nothing
end

function _finish!(stats, c::TRCore, st::TRState, k, t0)
    set_solution!(stats, c.x)
    set_objective!(stats, st.f)
    set_dual_residual!(stats, st.g_norm)
    set_iter!(stats, k)
    set_time!(stats, time() - t0)
    return stats
end

function _record!(stats, c::TRCore, st::TRState, k, t0)
    set_iter!(stats, k)
    set_objective!(stats, st.f)
    set_dual_residual!(stats, st.g_norm)
    set_solution!(stats, c.x)
    set_time!(stats, time() - t0)
    return nothing
end
