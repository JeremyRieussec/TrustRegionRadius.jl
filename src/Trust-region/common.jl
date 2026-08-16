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
    x_ref::Union{Nothing, V}
    want_true_curv::Bool
    want_paired::Bool
    paired_δ::Float64
    paired_σ²::Float64
end

"""
    TRCore(nlp, rule, model, subsolver, params; x_ref, true_curvature)

`x_ref` is a reference solution. When given, the trace carries
`‖x_k − x_ref‖`, which is what the superlinear rate theorem of Part~II is stated
in; without it only `‖g_k‖` is available, and the two differ by the condition
number of the Hessian at the solution, which is the constant under study.

`true_curvature = true` adds `λ_min(∇²f(x_k))` beside `λ_min(B_k)`. It costs one
extra eigenvalue estimate per iteration and is off by default. The pair is what
separates a saddle the model can see from one it cannot, which is the question
the second-order sections of Part~II ask.
"""
function TRCore(nlp::AbstractNLPModel{T, V}, rule, model, subsolver,
                params::TRParams{T};
                x_ref::Union{Nothing, AbstractVector} = nothing,
                true_curvature::Bool = false) where {T, V}
    x0 = nlp.meta.x0
    rule_c = deepcopy(rule); mod_c = deepcopy(model); sub_c = deepcopy(subsolver)

    validate_thresholds(rule_c, params.η, params.η1, params.η2)
    check_model_problem(mod_c, nlp)

    let mt = model_eltype(mod_c)
        mt === nothing || mt === T || throw(ArgumentError(
            "TRSolver: the model was built for $mt but the problem has element " *
            "type $T. Construct it as $(nameof(typeof(mod_c)))(T = $T)."))
    end

    # The paired decrease statistic is computed only when a sampling rule reads
    # it. A deterministic run has no batch to pair over and pays nothing.
    want_paired = nlp isa SampledNLP && needs_paired(nlp.rule)

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

    xr = x_ref === nothing ? nothing : convert(V, collect(x_ref))
    xr === nothing || length(xr) == nlp.meta.nvar || throw(ArgumentError(
        "TRCore: x_ref has length $(length(xr)) but the problem has " *
        "$(nlp.meta.nvar) variables."))

    return TRCore{T, V, typeof(rule_c), typeof(mod_c), typeof(sub_c)}(
        similar(x0), similar(x0), similar(x0), similar(x0), similar(x0),
        similar(x0), similar(x0), Hs_new, SubWorkspace(x0),
        rule_c, mod_c, sub_c, params, retro, want_curv, want_vec, sub_hs, Float64[],
        xr, true_curvature, want_paired, NaN, NaN)
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
    # --- diagnostics: hypotheses of Part II, measured rather than assumed ---
    γ::T                # ‖g_{k+1} − g_k − H_k s_k‖ / ‖s_k‖   (Dennis–Moré)
    ξ::T                # ‖H_k s_k + g_k‖ / ‖g_k‖             (realised inexactness)
    cos_cauchy::T       # cos(s_k, −g_k); 1 exactly at the Cauchy point
    cg_iters::Int       # inner iterations of the subproblem solver
    branch::Symbol      # which branch of the radius rule fired
    λtrue::Float64      # λ_min(∇²f(x_k)), when asked for
    dist::T             # ‖x_k − x_ref‖, when a reference solution was given
end

"""
    _blank_diagnostics(T) -> tuple

The diagnostic fields of [`TRState`](@ref) before anything has been measured.
`NaN` rather than zero: a quantity that was not computed must not be mistaken for
one that was measured to be zero.
"""
@inline _blank_diagnostics(::Type{T}) where {T} =
    (T(NaN), T(NaN), T(NaN), 0, :none, NaN, T(NaN))

"""
    _true_lambda_min(c, nlp, x) -> Float64

`λ_min(∇²f(x))`, the curvature of the *problem* rather than of the model.

Returns `NaN` when it was not requested, and also when the problem cannot supply
a Hessian: a diagnostic must not be able to fail a run.
"""
function _true_lambda_min(c::TRCore, nlp, x)
    c.want_true_curv || return NaN
    try
        return Float64(lambda_min_estimate(ExactHessian(), nlp, x))
    catch
        return NaN
    end
end

"""
    _distance_to_ref(c, x) -> T

`‖x − x_ref‖`, or `NaN` when no reference solution was given. Uses the
subproblem workspace as scratch, which is free at every point where this is
called.
"""
@inline function _distance_to_ref(c::TRCore{T}, x) where {T}
    c.x_ref === nothing && return T(NaN)
    @. c.ws.cand = x - c.x_ref
    return T(norm(c.ws.cand))
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
    st = TRState{T}(f, g_norm, Δ, crit, λmin, T(0), T(0), T(0), T(0),
                    false, false, false, false, _blank_diagnostics(T)...)
    st.λtrue = _true_lambda_min(c, nlp, c.x)
    st.dist  = _distance_to_ref(c, c.x)
    return st
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
    st.λtrue = _true_lambda_min(c, nlp, c.x)
    st.dist  = _distance_to_ref(c, c.x)
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
    st.cg_iters = c.ws.iters

    # --- ratio ---
    @. c.x_cand = c.x + c.s
    f_cand = obj(nlp, c.x_cand)

    # Paired differences, taken here because this is the only point at which both
    # x_k and the trial point are live and the batch has not moved. The rule is
    # the consumer, so the statistic is handed straight to it.
    if c.want_paired
        c.paired_δ, c.paired_σ², N_paired = paired_decrease_stats(nlp, c.x, c.x_cand)
        record_paired!(nlp.rule, c.paired_δ, c.paired_σ², N_paired)
    end
    # CG accumulated B·s while building the step; only a subsolver that did not
    # declare `returns_hprod` needs it computed here.
    c.sub_hs || model_hprod!(c.model, nlp, c.x, c.s, c.Hs)
    st.predicted = -dot(c.g, c.s) - T(0.5) * dot(c.s, c.Hs)

    # Two diagnostics taken here because this is the only point at which `c.g` is
    # still g_k and `c.Hs` already holds H_k s_k. Both reuse the subproblem
    # workspace, which the solver has finished with, so neither allocates.
    #
    #   ξ_k = ‖H_k s_k + g_k‖ / ‖g_k‖   is the *realised* inexactness of
    #   Condition C.Inx, which every result of Part II §7 assumes tends to zero.
    #   The forcing sequence a solver advertises is an upper bound on it.
    #
    #   cos(s_k, −g_k) = 1 exactly when the step is the Cauchy point, which with
    #   truncated CG and a small radius is the silent degeneration to gradient
    #   descent.
    if st.g_norm > 0
        @. c.ws.cand = c.Hs + c.g
        st.ξ = T(norm(c.ws.cand)) / st.g_norm
        st.cos_cauchy = st.s_norm > 0 ?
            -dot(c.g, c.s) / (st.g_norm * st.s_norm) : T(NaN)
    else
        st.ξ = T(NaN); st.cos_cauchy = T(NaN)
    end

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

        # γ_k = ‖y_k − H_k s_k‖ / ‖s_k‖ is the Dennis–Moré residual, the
        # hypothesis behind Proposition [steps are strictly decreasing] and behind
        # the superlinear rate. Everything it needs is already in the workspace,
        # and it has to be taken before `update_model!` replaces H_k by H_{k+1}.
        if st.s_norm > 0
            @. c.ws.cand = c.y - c.Hs
            st.γ = T(norm(c.ws.cand)) / st.s_norm
        else
            st.γ = T(NaN)
        end

        update_model!(c.model, c.s, c.y)
    end
    st.accepted || (st.γ = T(NaN))

    # The curvature belongs to the model at the *new* iterate; on a rejected step
    # neither has moved and the previous value still stands.
    if c.want_curv && st.accepted
        st.λmin, c.vmin = curvature_estimate(c.model, nlp, c.x, c.want_vec)
    end
    if st.accepted
        st.λtrue = _true_lambda_min(c, nlp, c.x)
        st.dist  = _distance_to_ref(c, c.x)
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
    st.branch = last_branch(c.rule)
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

# Alignment

Entry `j` of every trajectory describes iteration `j-1`, counting from the
initial point. The state trajectories `Δ`, `g`, `f` (and `λ`, `τ`, `dist`,
`λtrue`, `tg`) carry one **extra entry at the end**: the state left behind by the
final iteration.

To pair a state with the iteration it produced, drop that last entry:

```julia
Δ = stats.solver_specific[:delta_trajectory]
s = stats.solver_specific[:step_trajectory]
active = stats.solver_specific[:active_trajectory]
Δ[1:end-1] .>= s          # radius against the step taken inside it
```

Aligning on the *tail* instead pairs `Δ_k` with `‖s_{k-1}‖` and shifts every
activity, ratio and step-versus-radius plot by one iteration. Prefer
[`theta_trajectory`](@ref) and the other helpers in the diagnostics layer, which
do the alignment once and correctly.
"""
mutable struct TRTrace
    on::Bool
    curv::Bool
    Δ::Vector{Float64}
    g::Vector{Float64}
    f::Vector{Float64}
    λ::Vector{Float64}
    τ::Vector{Float64}
    ρ::Vector{Float64}
    s::Vector{Float64}
    active::Vector{Bool}
    accepted::Vector{Bool}
    # --- diagnostics ---
    ρ̃::Vector{Float64}
    γ::Vector{Float64}
    ξ::Vector{Float64}
    cosc::Vector{Float64}
    cgit::Vector{Int}
    branch::Vector{Symbol}
    dist::Vector{Float64}
    λtrue::Vector{Float64}
end

TRTrace(on::Bool, curv::Bool) =
    TRTrace(on, curv, Float64[], Float64[], Float64[], Float64[], Float64[],
            Float64[], Float64[], Bool[], Bool[],
            Float64[], Float64[], Float64[], Float64[], Int[], Symbol[],
            Float64[], Float64[])

function trace_pre!(tr::TRTrace, st::TRState)
    tr.on || return nothing
    push!(tr.Δ, st.Δ); push!(tr.g, st.g_norm); push!(tr.f, st.f)
    tr.curv && (push!(tr.λ, st.λmin); push!(tr.τ, Float64(st.crit)))
    push!(tr.dist, Float64(st.dist)); push!(tr.λtrue, st.λtrue)
    return nothing
end

function trace_post!(tr::TRTrace, st::TRState)
    tr.on || return nothing
    trace_pre!(tr, st)
    push!(tr.ρ, st.ρ); push!(tr.s, st.s_norm)
    push!(tr.active, st.active); push!(tr.accepted, st.accepted)
    push!(tr.ρ̃, Float64(st.ρ_rule)); push!(tr.γ, Float64(st.γ))
    push!(tr.ξ, Float64(st.ξ)); push!(tr.cosc, Float64(st.cos_cauchy))
    push!(tr.cgit, st.cg_iters); push!(tr.branch, st.branch)
    return nothing
end

"""
    attach_trace!(stats, tr, Δ_final)

Write the trajectories into `stats.solver_specific`.

Beyond the seven original keys, and on the same alignment convention:

| key | length | meaning |
|:--|:--|:--|
| `:rho_tilde_trajectory` | `k` | the ratio that drove the radius: `ρ_k`, or `ρ̃_{k+1}` under a retrospective rule |
| `:gamma_trajectory` | `k` | `‖g_{k+1} − g_k − H_k s_k‖/‖s_k‖`, the Dennis–Moré residual; `NaN` on rejected steps |
| `:xi_trajectory` | `k` | `‖H_k s_k + g_k‖/‖g_k‖`, the realised inexactness |
| `:cos_cauchy_trajectory` | `k` | `cos(s_k, −g_k)`; `1` at the Cauchy point |
| `:cg_iters_trajectory` | `k` | inner iterations of the subproblem solver |
| `:branch_trajectory` | `k` | which branch of the radius rule fired, see [`last_branch`](@ref) |
| `:dist_trajectory` | `k+1` | `‖x_k − x_ref‖`, only when a reference solution was given |
| `:lambda_min_true_trajectory` | `k+1` | `λ_min(∇²f(x_k))`, only when `true_curvature = true` |

The last two are attached only when they were actually measured, so their
absence is unambiguous rather than a column of `NaN`.
"""
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
    set_solver_specific!(stats, :rho_tilde_trajectory,  tr.ρ̃)
    set_solver_specific!(stats, :gamma_trajectory,      tr.γ)
    set_solver_specific!(stats, :xi_trajectory,         tr.ξ)
    set_solver_specific!(stats, :cos_cauchy_trajectory, tr.cosc)
    set_solver_specific!(stats, :cg_iters_trajectory,   tr.cgit)
    set_solver_specific!(stats, :branch_trajectory,     tr.branch)
    any(isfinite, tr.dist) &&
        set_solver_specific!(stats, :dist_trajectory, tr.dist)
    any(isfinite, tr.λtrue) &&
        set_solver_specific!(stats, :lambda_min_true_trajectory, tr.λtrue)
    set_solver_specific!(stats, :final_delta, Float64(Δ_final))
    return nothing
end

# -----------------------------------------------------------------------------
# The stochastic trace
# -----------------------------------------------------------------------------

"""
    SampleTrace(on, want_truth)

The trajectories that exist only under sampling, kept apart from
[`TRTrace`](@ref) because none of them means anything on a deterministic run: a
variance estimated from one exact evaluation is not a small number, it is a
category error.

The two solvers that sample own one of each and fill them at the same points.
Everything a deterministic run can also produce — the radius, the ratio, the
step, the residual diagnostics — stays in `TRTrace`.

Alignment follows `TRTrace`: `tg` is a state trajectory with `k+1` entries, the
rest are per-iteration with `k`.
"""
mutable struct SampleTrace
    on::Bool
    truth::Bool
    tg::Vector{Float64}
    σg²::Vector{Float64}
    σf²::Vector{Float64}
    δ::Vector{Float64}
    σ²D::Vector{Float64}
end

SampleTrace(on::Bool, truth::Bool) =
    SampleTrace(on, truth, Float64[], Float64[], Float64[], Float64[], Float64[])

function sample_pre!(sr::SampleTrace, truth_val)
    sr.on || return nothing
    sr.truth && push!(sr.tg, Float64(truth_val))
    return nothing
end

function sample_post!(sr::SampleTrace, c::TRCore, nlp, truth_val)
    sr.on || return nothing
    sample_pre!(sr, truth_val)
    push!(sr.σg², Float64(nlp.σg²)); push!(sr.σf², Float64(nlp.σf²))
    push!(sr.δ, c.paired_δ); push!(sr.σ²D, c.paired_σ²)
    return nothing
end

"""
    attach_sample_trace!(stats, sr)

| key | length | meaning |
|:--|:--|:--|
| `:true_grad_trajectory` | `k+1` | `‖∇f(x_k)‖`, only when the problem has a truth |
| `:sigma_g2_trajectory` | `k` | `σ̂_g²` of the batch that produced the step |
| `:sigma_f2_trajectory` | `k` | `σ̂_f²` likewise |
| `:paired_decrease_trajectory` | `k` | `δ̂_N`, the mean paired decrease |
| `:paired_variance_trajectory` | `k` | `σ̂_N²`, its sample variance |

The last two are attached only under a rule that asked for them, so their
absence means the statistic was never formed rather than that it was zero.
"""
function attach_sample_trace!(stats, sr::SampleTrace)
    sr.on || return nothing
    sr.truth && set_solver_specific!(stats, :true_grad_trajectory, sr.tg)
    set_solver_specific!(stats, :sigma_g2_trajectory, sr.σg²)
    set_solver_specific!(stats, :sigma_f2_trajectory, sr.σf²)
    if any(isfinite, sr.δ)
        set_solver_specific!(stats, :paired_decrease_trajectory, sr.δ)
        set_solver_specific!(stats, :paired_variance_trajectory, sr.σ²D)
    end
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
