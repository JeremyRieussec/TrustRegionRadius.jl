# =============================================================================
# src/Radius_updates/rules.jl
#
# Every trust-region radius update mechanism, under its primary name.
#
#   RDelta        criticality-blind, fixed factor        (Conn-Gould-Toint)
#   RStep         step-anchored                          (Powell, CGT 10.5.2)
#   RDFO          criticality-anchored, parameter ζ      (Scheinberg et al.)
#   RGrad         gradient-scaled, Δ = μ‖g‖, μ uncapped  (Fan-Yuan)
#   RGradCapped   gradient-scaled with μ ≤ μ_max
#   RAdaptiveStep step-anchored, exponential factor      (Hei 2003)
#   RAdaptiveGrad gradient-anchored, exponential factor
#   RAdaptiveFanYuan  μ *= R(ρ), Δ = μ‖g‖                (Hei + Fan-Yuan)
#   RRTR          retrospective                          (Bastin et al. 2010)
#   RRTRGrad      retrospective gradient-scaled          (Fan-Pan-Song 2016)
#
# All are concrete subtypes of `RadiusRule` and implement
#
#     initial_radius(rule, Δ₀, g_norm)                            -> Float64
#     update_radius!(rule, Δ, ρ, accepted, η₁, η₂,
#                    s_norm, g_norm_old, g_norm_new)              -> Float64
#     reset_rule!(rule)                                           -> nothing
#
# Two conventions are enforced throughout.
#
# ACCEPTANCE IS DECOUPLED FROM SCALING.  The solver decides acceptance with the
# threshold η (`ρ ≥ η`) and passes the outcome as the `accepted` flag.  The
# thresholds η₁ and η₂ passed to the rule are *scaling* thresholds only, and
# satisfy 0 ≤ η ≤ η₁ ≤ η₂ < 1.  A rule therefore sees three regimes, not two,
# and an iteration can be accepted (ρ ≥ η) while still contracting the radius
# (ρ < η₁).  Setting η = 0 is legitimate and is what the first-order framework
# of Part I covers but Curtis-Scheinberg does not.  Only the rules whose update
# genuinely branches on acceptance rather than on ρ alone read the flag: the
# retrospective family, for which ρ̃ is defined only on accepted steps.
#
# FACTORS SATISFY 0 < γ₁ ≤ γ₂ < 1 < γ₃.  γ₁ is the aggressive contraction, γ₂
# the mild one, γ₃ the expansion, uniformly across every rule; `check_factors`
# enforces it at construction.  Rules that do not use one of the three pass
# `nothing` for it.
# =============================================================================

# -----------------------------------------------------------------------------
# Interface
# -----------------------------------------------------------------------------

"""
    RadiusRule

Abstract supertype for trust-region radius update mechanisms.

A concrete `MyRule <: RadiusRule` must implement

    initial_radius(rule, Δ₀, g_norm) -> Float64
    update_radius!(rule, Δ, ρ, accepted, η₁, η₂,
                   s_norm, g_norm_old, g_norm_new) -> Float64

and, if it carries mutable state,

    reset_rule!(rule) -> nothing

# Arguments of `update_radius!`

- `Δ`:        the current radius Δ_k.
- `ρ`:        the ratio that drives the *scaling*. This is ρ_k for ordinary
              rules and the retrospective ratio ρ̃_k for those declaring
              [`needs_retrospective`](@ref). On a rejected step ρ̃ is undefined
              and the solver passes ρ_k, so a retrospective rule must branch on
              `accepted` before it compares `ρ` to its own thresholds.
- `accepted`: whether the step was taken, i.e. `ρ_k ≥ η`. Decided by the solver
              with the acceptance threshold η, which the rule never sees.
- `η₁, η₂`:   scaling thresholds, `η ≤ η₁ ≤ η₂ < 1`.
- `s_norm`:   ‖s_k‖, the realised trial step length.
- `g_norm_old`: ‖g_k‖, before the accept/reject decision. Used by rules
              anchored to criticality at the *current* iterate ([`RDFO`](@ref)).
- `g_norm_new`: ‖g_{k+1}‖, after it. Used by rules of the form `Δ = μ‖g‖`
              ([`RGrad`](@ref) and relatives). Equal to `g_norm_old` on a
              rejected step.

Every rule must contract when the iteration is unsuccessful — `Δ_{k+1} ≤ γ₂Δ_k`
whenever `ρ_k < η` — or it forfeits weak admissibility, and, since a rejected
step leaves both the model and the iterate unchanged, a rule that returns
`Δ_{k+1} = Δ_k` there will re-solve an identical subproblem for ever. Because
`η ≤ η₁`, branching on `ρ < η₁` is enough to guarantee it, which is why most
rules can ignore `accepted`.
"""
abstract type RadiusRule end

"""
    initial_radius(rule, Δ₀, g_norm) -> Float64

Radius at iteration 0. Most rules return `Δ₀`; those of the form
`Δ = μ‖g‖` return `μ₀ · g_norm` and so ignore `Δ₀` entirely.
"""
initial_radius(::RadiusRule, Δ₀::Float64, ::Float64) = Δ₀

"""
    update_radius!(rule, Δ, ρ, accepted, η₁, η₂, s_norm, g_norm_old, g_norm_new)

Radius for the next iteration. May mutate `rule` (e.g. the multiplier μ).
See [`RadiusRule`](@ref) for the meaning of each argument.
"""
function update_radius! end

"""
    reset_rule!(rule) -> nothing

Restore mutable state to its construction value. Default: no-op.
Called at the start of every solve and by `SolverCore.reset!`.
"""
reset_rule!(::RadiusRule) = nothing

"""
    needs_retrospective(rule) -> Bool

`true` if the rule expects the retrospective ratio ρ̃ in the `ρ` slot of
`update_radius!`. The solver computes ρ̃ only when this returns `true`, since it
costs one extra model evaluation at the previous iterate, and only on accepted
steps, since ρ̃ compares the new model against a step that was taken.
"""
needs_retrospective(::RadiusRule) = false

"""
    is_criticality_anchored(rule) -> Bool

`true` if the radius is tied to a criticality measure, so that `Δ_k → 0` along
a convergent run. For these rules eventual inactivity of the trust region
requires a parameter above the problem-dependent threshold κ̄ = 4/λ*_min.

The complement is not a single regime: `RDelta` keeps `liminf Δ_k > 0`, while
the step-anchored rules drive `Δ_k → 0` in the local regime with the stronger
summability `Σ_k Δ_k < ∞`, inherited from Powell's theorem on the step series.
Use [`asymptotic_regime`](@ref) when the three-way distinction matters.
"""
is_criticality_anchored(::RadiusRule) = false

"""
    asymptotic_regime(rule) -> Symbol

Which of the three asymptotic regimes the rule belongs to:

- `:vanishing`      — `Δ_k → 0` with `Σ_k Δ_k²/M_k < ∞` (criticality-anchored);
- `:step_summable`  — `Σ_k Δ_k < ∞` in the local regime (step-anchored);
- `:bounded_below`  — `liminf_k Δ_k > 0` (criticality-blind, radius-anchored).
"""
asymptotic_regime(::RadiusRule) = :bounded_below

"""
    validate_thresholds(rule, η, η₁, η₂) -> nothing

Hook called once per solve, after the parameters are known, for rules that
require more of `(η, η₁, η₂)` than `0 ≤ η ≤ η₁ ≤ η₂ < 1`. Default: no-op.
Throws `ArgumentError` on a violation.
"""
validate_thresholds(::RadiusRule, ::Real, ::Real, ::Real) = nothing

"""
    check_factors(name; γ₁, γ₂, γ₃) -> nothing

Enforce the standing convention `0 < γ₁ ≤ γ₂ < 1 < γ₃` on the scaling factors
of a rule. Pass `nothing` for a factor the rule does not use; the remaining
inequalities are still checked.

An `ArgumentError` rather than `@assert`, because `@assert` is documented as
liable to be disabled and these are argument checks, not internal invariants.
"""
function check_factors(name::Symbol;
                       γ₁ = nothing, γ₂ = nothing, γ₃ = nothing)
    if γ₁ !== nothing && !(0 < γ₁ < 1)
        throw(ArgumentError("$name: need 0 < γ₁ < 1, got γ₁ = $γ₁"))
    end
    if γ₂ !== nothing && !(0 < γ₂ < 1)
        throw(ArgumentError("$name: need 0 < γ₂ < 1, got γ₂ = $γ₂"))
    end
    if γ₃ !== nothing && !(γ₃ > 1)
        throw(ArgumentError("$name: need γ₃ > 1, got γ₃ = $γ₃"))
    end
    if γ₁ !== nothing && γ₂ !== nothing && !(γ₁ <= γ₂)
        throw(ArgumentError("$name: need γ₁ ≤ γ₂, got γ₁ = $γ₁, γ₂ = $γ₂"))
    end
    return nothing
end

# =============================================================================
# RDelta — classical multiplicative update
# =============================================================================

"""
    RDelta(; γ₁ = 0.25, γ₂ = 0.5, γ₃ = 2.0, Δmax = Inf)

Classical three-case multiplicative update (Conn, Gould & Toint §6.1):

    ρ < η₁       →  Δ ← γ₁ Δ
    η₁ ≤ ρ < η₂  →  Δ ← γ₂ Δ
    ρ ≥ η₂       →  Δ ← γ₃ Δ

The branches are read off ρ alone, never off acceptance, so with `η < η₁` this
rule contracts on iterations whose step was nonetheless taken — which is the
whole content of decoupling the two thresholds.

Criticality-blind: the radius depends on its own history and on ρ, never on
‖g‖. Consequently `liminf Δ_k > 0`, the trust-region constraint eventually
stops binding with no side condition, and the radius carries no asymptotic
information about criticality.
"""
mutable struct RDelta <: RadiusRule
    γ₁::Float64
    γ₂::Float64
    γ₃::Float64
    Δmax::Float64
    function RDelta(; γ₁::Float64 = 0.25, γ₂::Float64 = 0.5,
                      γ₃::Float64 = 2.0, Δmax::Float64 = Inf)
        check_factors(:RDelta; γ₁ = γ₁, γ₂ = γ₂, γ₃ = γ₃)
        new(γ₁, γ₂, γ₃, Δmax)
    end
end

RDelta(γ₁::Float64, γ₂::Float64, γ₃::Float64) = RDelta(; γ₁ = γ₁, γ₂ = γ₂, γ₃ = γ₃)

function update_radius!(r::RDelta, Δ::Float64, ρ::Float64, ::Bool,
                        η₁::Float64, η₂::Float64,
                        ::Float64, ::Float64, ::Float64)
    ρ >= η₂ && return min(r.γ₃ * Δ, r.Δmax)
    ρ >= η₁ && return r.γ₂ * Δ
    return r.γ₁ * Δ
end

asymptotic_regime(::RDelta) = :bounded_below

# =============================================================================
# RStep — step-anchored update with constant factors
# =============================================================================

"""
    RStep(; γ₁ = 0.25, γ₂ = 0.5, γ₃ = 2.0, Δmax = Inf, Δmin = 1e-14,
            contract_on_step = true)

Radius proportional to the realised step:

    ρ < η₁       →  Δ ← γ₁ ‖s_k‖     (or γ₁ Δ_k, see `contract_on_step`)
    η₁ ≤ ρ < η₂  →  Δ ← γ₂ ‖s_k‖
    ρ ≥ η₂       →  Δ ← γ₃ ‖s_k‖

`contract_on_step = true` is the rule as stated in Part I, in which all three
branches are anchored to `‖s_k‖`. `false` gives the variant that contracts on
the previous radius instead, which is milder whenever the subsolver returned an
interior step: `γ₁‖s_k‖ ≤ γ₁Δ_k`. Both satisfy the contraction condition; they
differ in how fast the radius falls after a rejection, so which one is in force
should be reported rather than left implicit.

Requires `η₁ > 0`: at `η₁ = 0` the aggressive branch is unreachable for every
ρ ≥ 0 and the lower-bound constant of Part I degenerates.

!!! warning "Δmin exists for a reason"
    On an accepted step the new radius is proportional to `‖s_k‖`, so a short
    step gives a small radius, which gives a shorter step still. Near
    convergence, or when a truncated-CG subsolver stops on its first iteration,
    this can collapse to zero, after which every step is zero, ρ is `NaN`, and
    the solver spins to `max_iterations`. `Δmin` floors the radius and turns
    that silent hang into ordinary slow progress. Set `Δmin = 0.0` to reproduce
    the unguarded rule exactly — which is what the asymptotic claims of Part II
    are stated about.
"""
mutable struct RStep <: RadiusRule
    γ₁::Float64
    γ₂::Float64
    γ₃::Float64
    Δmax::Float64
    Δmin::Float64
    contract_on_step::Bool
    function RStep(; γ₁::Float64 = 0.25, γ₂::Float64 = 0.5, γ₃::Float64 = 2.0,
                     Δmax::Float64 = Inf, Δmin::Float64 = 1e-14,
                     contract_on_step::Bool = true)
        check_factors(:RStep; γ₁ = γ₁, γ₂ = γ₂, γ₃ = γ₃)
        Δmin >= 0 || throw(ArgumentError("RStep: need Δmin ≥ 0, got $Δmin"))
        new(γ₁, γ₂, γ₃, Δmax, Δmin, contract_on_step)
    end
end

RStep(γ₁::Float64, γ₂::Float64, γ₃::Float64) = RStep(; γ₁ = γ₁, γ₂ = γ₂, γ₃ = γ₃)

function update_radius!(r::RStep, Δ::Float64, ρ::Float64, ::Bool,
                        η₁::Float64, η₂::Float64,
                        s_norm::Float64, ::Float64, ::Float64)
    Δnew = if ρ < η₁
               r.γ₁ * (r.contract_on_step ? s_norm : Δ)
           elseif ρ < η₂
               r.γ₂ * s_norm
           else
               r.γ₃ * s_norm
           end
    return clamp(Δnew, r.Δmin, r.Δmax)
end

asymptotic_regime(::RStep) = :step_summable

function validate_thresholds(::RStep, ::Real, η₁::Real, ::Real)
    η₁ > 0 || throw(ArgumentError("RStep: the step-driven update needs η₁ > 0"))
    return nothing
end

# =============================================================================
# RDFO — criticality-anchored, parameter ζ
# =============================================================================

"""
    RDFO(; γ₁ = 0.25, γ₂ = 0.5, γ₃ = 2.0, ζ = 1.0, Δmax = Inf)

DFO-like update comparing the radius to the criticality measure:

    ρ < η₁                 →  Δ ← γ₁ Δ
    ρ ≥ η₁ and Δ > ζ‖g_k‖  →  Δ ← γ₂ Δ     (radius large relative to criticality)
    ρ ≥ η₁ and Δ ≤ ζ‖g_k‖  →  Δ ← γ₃ Δ     (room to expand)

Uses ‖g_k‖ *before* the accept/reject decision.

Drives `Δ_k → 0`, so eventual inactivity of the trust region requires
`ζ > κ̄ = 4/λ*_min(∇²f(x*))`. That constant is a property of the solution and
cannot be checked in advance; below the threshold the constraint can bind at
every iteration, the method converges only linearly, and no first-order
diagnostic shows anything wrong. Choose `ζ` generously: on a heterogeneous test
set an over-large ζ costs nothing, while an over-small one costs solved
problems.
"""
mutable struct RDFO <: RadiusRule
    γ₁::Float64
    γ₂::Float64
    γ₃::Float64
    ζ::Float64
    Δmax::Float64
    function RDFO(; γ₁::Float64 = 0.25, γ₂::Float64 = 0.5, γ₃::Float64 = 2.0,
                    ζ::Float64 = 1.0, Δmax::Float64 = Inf)
        check_factors(:RDFO; γ₁ = γ₁, γ₂ = γ₂, γ₃ = γ₃)
        ζ > 0 || throw(ArgumentError("RDFO: need ζ > 0, got $ζ"))
        new(γ₁, γ₂, γ₃, ζ, Δmax)
    end
end

RDFO(γ₁::Float64, γ₂::Float64, γ₃::Float64, ζ::Float64) =
    RDFO(; γ₁ = γ₁, γ₂ = γ₂, γ₃ = γ₃, ζ = ζ)

function update_radius!(r::RDFO, Δ::Float64, ρ::Float64, ::Bool,
                        η₁::Float64, ::Float64,
                        ::Float64, g_norm_old::Float64, ::Float64)
    ρ < η₁ && return r.γ₁ * Δ
    Δ > r.ζ * g_norm_old && return r.γ₂ * Δ
    return min(r.γ₃ * Δ, r.Δmax)
end

is_criticality_anchored(::RDFO) = true
asymptotic_regime(::RDFO) = :vanishing

# =============================================================================
# RGrad — gradient-scaled, μ uncapped
# =============================================================================

"""
    RGrad(; γ₁ = 0.25, γ₂ = 0.5, γ₃ = 2.0, μ = 1.0, half_test = true)

Gradient-scaled radius `Δ_k = μ_k ‖g_k‖` with an unbounded multiplier:

    ρ < η₁                     →  μ ← γ₁ μ
    η₁ ≤ ρ < η₂                →  μ ← γ₂ μ
    ρ ≥ η₂ and ‖s_k‖ > ½Δ_k    →  μ ← γ₃ μ
    otherwise                  →  μ unchanged

Here `μ_k` *is* the radius-to-criticality ratio `Δ_k/‖g_k‖`, so the rule
performs a search for the inactivity threshold κ̄ without knowing it: μ climbs
on very successful iterations that use the region, contracts otherwise, and
stops climbing once the region stops binding. Because μ is uncapped it crosses
any threshold eventually, which makes eventual inactivity **unconditional** —
the only rule in the survey for which this holds without a hypothesis on an
unknowable constant.

The middle branch is not decoration: it is what keeps μ from ratcheting upward
on mediocre iterations, and the four-branch form is the one the boundedness
argument for μ is stated about.

The guard `‖s_k‖ > ½Δ_k` is not cosmetic either. It is what converts "μ grows"
into "‖s_k‖ is comparable to μ_k‖g_k‖", which is the step that makes the
boundedness argument work. Any constant in (0,1) serves; `half_test = false`
disables the guard, which breaks that argument.

See [`RGradCapped`](@ref) for the bounded variant required by the global
asymptotic theory (`Δ_k → 0` needs `μ_k ≤ μ̄`).
"""
mutable struct RGrad <: RadiusRule
    γ₁::Float64
    γ₂::Float64
    γ₃::Float64
    μ::Float64
    μ₀::Float64
    half_test::Bool
    function RGrad(; γ₁::Float64 = 0.25, γ₂::Float64 = 0.5, γ₃::Float64 = 2.0,
                     μ::Float64 = 1.0, half_test::Bool = true)
        check_factors(:RGrad; γ₁ = γ₁, γ₂ = γ₂, γ₃ = γ₃)
        μ > 0 || throw(ArgumentError("RGrad: need μ > 0, got $μ"))
        new(γ₁, γ₂, γ₃, μ, μ, half_test)
    end
end

RGrad(γ₁::Float64, γ₂::Float64, γ₃::Float64, μ::Float64) =
    RGrad(; γ₁ = γ₁, γ₂ = γ₂, γ₃ = γ₃, μ = μ)

initial_radius(r::RGrad, ::Float64, g_norm::Float64) = r.μ * g_norm
reset_rule!(r::RGrad) = (r.μ = r.μ₀; nothing)
is_criticality_anchored(::RGrad) = true
asymptotic_regime(::RGrad) = :vanishing

function update_radius!(r::RGrad, Δ::Float64, ρ::Float64, ::Bool,
                        η₁::Float64, η₂::Float64,
                        s_norm::Float64, ::Float64, g_norm_new::Float64)
    if ρ < η₁
        r.μ *= r.γ₁
    elseif ρ < η₂
        r.μ *= r.γ₂
    elseif !r.half_test || s_norm > 0.5 * Δ
        r.μ *= r.γ₃
    end
    return r.μ * g_norm_new
end

# =============================================================================
# RGradCapped — gradient-scaled with μ ≤ μ_max
# =============================================================================

"""
    RGradCapped(; γ₁ = 0.25, γ₂ = 0.5, γ₃ = 2.0, μ = 1.0, μ_max = 1.0,
                  half_test = true)

[`RGrad`](@ref) with an explicit cap `μ_k ≤ μ_max`.

The cap is what the asymptotic results requiring `Δ_k → 0` assume, and it is
not free. Eventual inactivity now needs `μ_max > κ̄ = 4/λ*_min`; below that the
trust region binds at every iteration and the method degrades to linear
convergence while ρ stays healthy and ‖g‖ keeps falling.

With a truncated-CG subsolver a small `μ_max` is worse than slow. The first CG
iterate lies along `-g`, so if the region is small enough that CG truncates
there, the returned step is exactly the Cauchy point and the model Hessian
stops influencing the search direction: the method has silently become gradient
descent, and its limit can be a degenerate critical point that is not a
minimiser.
"""
mutable struct RGradCapped <: RadiusRule
    γ₁::Float64
    γ₂::Float64
    γ₃::Float64
    μ::Float64
    μ₀::Float64
    μ_max::Float64
    half_test::Bool
    function RGradCapped(; γ₁::Float64 = 0.25, γ₂::Float64 = 0.5, γ₃::Float64 = 2.0,
                           μ::Float64 = 1.0, μ_max::Float64 = 1.0,
                           half_test::Bool = true)
        check_factors(:RGradCapped; γ₁ = γ₁, γ₂ = γ₂, γ₃ = γ₃)
        μ > 0 || throw(ArgumentError("RGradCapped: need μ > 0, got $μ"))
        μ_max >= μ || throw(ArgumentError("RGradCapped: need μ_max ≥ μ"))
        new(γ₁, γ₂, γ₃, μ, μ, μ_max, half_test)
    end
end

initial_radius(r::RGradCapped, ::Float64, g_norm::Float64) = r.μ * g_norm
reset_rule!(r::RGradCapped) = (r.μ = r.μ₀; nothing)
is_criticality_anchored(::RGradCapped) = true
asymptotic_regime(::RGradCapped) = :vanishing

function update_radius!(r::RGradCapped, Δ::Float64, ρ::Float64, ::Bool,
                        η₁::Float64, η₂::Float64,
                        s_norm::Float64, ::Float64, g_norm_new::Float64)
    if ρ < η₁
        r.μ *= r.γ₁
    elseif ρ < η₂
        r.μ *= r.γ₂
    elseif !r.half_test || s_norm > 0.5 * Δ
        r.μ = min(r.γ₃ * r.μ, r.μ_max)
    end
    return r.μ * g_norm_new
end

# =============================================================================
# Hei family — piecewise-exponential factor
# =============================================================================

"""
    _r_exp(t, η₁, γ₁, γ₂, γ₃, λ₁, λ₂) -> Float64

The R-function of Hei (2003), in the normalisation of Part I: a non-decreasing
`R_{η₁} : ℝ → ℝ₊` with

    lim_{t→-∞} R = γ₁,     R(t) ≤ γ₂ for t < η₁,
    R(η₁) = 1 + γ₂,        lim_{t→+∞} R = γ₃,

realised as

    t < η₁  →  γ₁ + (γ₂ − γ₁) exp(λ₁(t − η₁))
    t ≥ η₁  →  (1 + γ₂) + (γ₃ − 1 − γ₂)(1 − exp(−λ₂(t − η₁)))

Both pieces are increasing and the jump at `η₁` is upward, so `R` is
non-decreasing on ℝ. Since `R ≤ γ₂ < 1` below `η₁` and `η ≤ η₁`, any rule that
multiplies by `R(ρ_k)` contracts on unsuccessful iterations automatically.

Requires `γ₃ > 1 + γ₂`, which is stronger than the standing `γ₃ > 1`: the value
at the threshold must itself be exceeded by the asymptote.
"""
@inline function _r_exp(t::Float64, η₁::Float64,
                        γ₁::Float64, γ₂::Float64, γ₃::Float64,
                        λ₁::Float64, λ₂::Float64)
    if t < η₁
        return γ₁ + (γ₂ - γ₁) * exp(λ₁ * (t - η₁))
    else
        return (1.0 + γ₂) + (γ₃ - 1.0 - γ₂) * (1.0 - exp(-λ₂ * (t - η₁)))
    end
end

"Default Hei constants, shared by the three adaptive rules."
const HEI_DEFAULTS = (γ₁ = 0.0625, γ₂ = 0.5, γ₃ = 4.0, λ₁ = 5.0, λ₂ = 5.0)

"""
    check_hei_factors(name, γ₁, γ₂, γ₃, λ₁, λ₂) -> nothing

`check_factors` plus the two requirements specific to the R-function:
`γ₃ > 1 + γ₂` and positive rates.
"""
function check_hei_factors(name::Symbol, γ₁, γ₂, γ₃, λ₁, λ₂)
    check_factors(name; γ₁ = γ₁, γ₂ = γ₂, γ₃ = γ₃)
    γ₃ > 1 + γ₂ || throw(ArgumentError(
        "$name: the R-function needs γ₃ > 1 + γ₂, got γ₃ = $γ₃, γ₂ = $γ₂"))
    (λ₁ > 0 && λ₂ > 0) || throw(ArgumentError("$name: need λ₁ > 0 and λ₂ > 0"))
    return nothing
end

"""
    RAdaptiveStep(; γ₁, γ₂, γ₃, λ₁, λ₂, Δmin = 1e-14)

Hei (2003) step-anchored rule with a smooth factor:

    Δ_{k+1} = R_{η₁}(ρ_k) · ‖s_k‖

The switch point of `R` is the scaling threshold `η₁` supplied by the solver,
not a constant of the rule, so the family stays comparable with the three-case
rules under the same `(η, η₁, η₂)`.

Step-anchored like [`RStep`](@ref), so the same `Δmin` guard applies and for the
same reason, and `η₁ > 0` is required for the same reason.
"""
mutable struct RAdaptiveStep <: RadiusRule
    γ₁::Float64
    γ₂::Float64
    γ₃::Float64
    λ₁::Float64
    λ₂::Float64
    Δmin::Float64
    function RAdaptiveStep(; γ₁::Float64 = HEI_DEFAULTS.γ₁, γ₂::Float64 = HEI_DEFAULTS.γ₂,
                             γ₃::Float64 = HEI_DEFAULTS.γ₃, λ₁::Float64 = HEI_DEFAULTS.λ₁,
                             λ₂::Float64 = HEI_DEFAULTS.λ₂, Δmin::Float64 = 1e-14)
        check_hei_factors(:RAdaptiveStep, γ₁, γ₂, γ₃, λ₁, λ₂)
        Δmin >= 0 || throw(ArgumentError("RAdaptiveStep: need Δmin ≥ 0"))
        new(γ₁, γ₂, γ₃, λ₁, λ₂, Δmin)
    end
end

function update_radius!(r::RAdaptiveStep, ::Float64, ρ::Float64, ::Bool,
                        η₁::Float64, ::Float64,
                        s_norm::Float64, ::Float64, ::Float64)
    f = _r_exp(ρ, η₁, r.γ₁, r.γ₂, r.γ₃, r.λ₁, r.λ₂)
    return max(f * s_norm, r.Δmin)
end

asymptotic_regime(::RAdaptiveStep) = :step_summable

function validate_thresholds(::RAdaptiveStep, ::Real, η₁::Real, ::Real)
    η₁ > 0 || throw(ArgumentError("RAdaptiveStep: the step-driven update needs η₁ > 0"))
    return nothing
end

"""
    RAdaptiveGrad(; γ₁, γ₂, γ₃, λ₁, λ₂)

Gradient-anchored Hei variant:

    Δ_{k+1} = min{ R_{η₁}(ρ_k) · ‖g_{k+1}‖ ,  γ₂ Δ_k }  if ρ_k < η₁
              R_{η₁}(ρ_k) · ‖g_{k+1}‖                    otherwise

Uses the gradient *after* the accept/reject decision. Criticality-anchored, so
`Δ_k → 0`.

The `min` on the first branch is not in the source rule and is not decorative.
`R` here *replaces* the multiplier rather than scaling it, so the realised ratio
is `Δ_{k+1}/Δ_k = R(ρ_k)/R(ρ_{k-1})` on a rejected step (‖g‖ being unchanged),
which exceeds one whenever the previous iteration was worse than this one. The
radius would then grow on an unsuccessful iteration and the contraction
condition would fail. Compare [`RAdaptiveFanYuan`](@ref), which scales `μ`
multiplicatively and needs no such guard.
"""
mutable struct RAdaptiveGrad <: RadiusRule
    γ₁::Float64
    γ₂::Float64
    γ₃::Float64
    λ₁::Float64
    λ₂::Float64
    function RAdaptiveGrad(; γ₁::Float64 = HEI_DEFAULTS.γ₁, γ₂::Float64 = HEI_DEFAULTS.γ₂,
                             γ₃::Float64 = HEI_DEFAULTS.γ₃, λ₁::Float64 = HEI_DEFAULTS.λ₁,
                             λ₂::Float64 = HEI_DEFAULTS.λ₂)
        check_hei_factors(:RAdaptiveGrad, γ₁, γ₂, γ₃, λ₁, λ₂)
        new(γ₁, γ₂, γ₃, λ₁, λ₂)
    end
end

function update_radius!(r::RAdaptiveGrad, Δ::Float64, ρ::Float64, ::Bool,
                        η₁::Float64, ::Float64,
                        ::Float64, ::Float64, g_norm_new::Float64)
    Δnew = _r_exp(ρ, η₁, r.γ₁, r.γ₂, r.γ₃, r.λ₁, r.λ₂) * g_norm_new
    return ρ < η₁ ? min(Δnew, r.γ₂ * Δ) : Δnew
end

is_criticality_anchored(::RAdaptiveGrad) = true
asymptotic_regime(::RAdaptiveGrad) = :vanishing

"""
    RAdaptiveFanYuan(; μ = 1.0, γ₁, γ₂, γ₃, λ₁, λ₂)

Hei factor driving a Fan-Yuan multiplier:

    μ_{k+1} = μ_k · R_{η₁}(ρ_k),    Δ_{k+1} = μ_{k+1} · ‖g_{k+1}‖

Combines the smooth factor of the Hei family with the ratio-tracking of
[`RGrad`](@ref): μ is again the radius-to-criticality ratio, but it is scaled
continuously in ρ rather than by one of three constants. Because the scaling is
multiplicative and `R ≤ γ₂ < 1` below `η₁`, contraction on unsuccessful
iterations holds without a guard.
"""
mutable struct RAdaptiveFanYuan <: RadiusRule
    μ::Float64
    μ₀::Float64
    γ₁::Float64
    γ₂::Float64
    γ₃::Float64
    λ₁::Float64
    λ₂::Float64
    function RAdaptiveFanYuan(; μ::Float64 = 1.0,
                                γ₁::Float64 = HEI_DEFAULTS.γ₁, γ₂::Float64 = HEI_DEFAULTS.γ₂,
                                γ₃::Float64 = HEI_DEFAULTS.γ₃, λ₁::Float64 = HEI_DEFAULTS.λ₁,
                                λ₂::Float64 = HEI_DEFAULTS.λ₂)
        check_hei_factors(:RAdaptiveFanYuan, γ₁, γ₂, γ₃, λ₁, λ₂)
        μ > 0 || throw(ArgumentError("RAdaptiveFanYuan: need μ > 0, got $μ"))
        new(μ, μ, γ₁, γ₂, γ₃, λ₁, λ₂)
    end
end

initial_radius(r::RAdaptiveFanYuan, ::Float64, g_norm::Float64) = r.μ * g_norm
reset_rule!(r::RAdaptiveFanYuan) = (r.μ = r.μ₀; nothing)
is_criticality_anchored(::RAdaptiveFanYuan) = true
asymptotic_regime(::RAdaptiveFanYuan) = :vanishing

function update_radius!(r::RAdaptiveFanYuan, ::Float64, ρ::Float64, ::Bool,
                        η₁::Float64, ::Float64,
                        ::Float64, ::Float64, g_norm_new::Float64)
    r.μ *= _r_exp(ρ, η₁, r.γ₁, r.γ₂, r.γ₃, r.λ₁, r.λ₂)
    return r.μ * g_norm_new
end

# =============================================================================
# Retrospective family
# =============================================================================

"""
    RRTR(; γ₁ = 0.0625, γ₂ = 0.25, γ₃ = 2.5, η̃₁ = 0.05, η̃₂ = 0.9,
           Δmax = Inf, Δmin = 1e-14)

Retrospective update of Bastin, Malmedy, Mouffe, Toint & Tomanos (2010).

The radius for iteration `k+1` is chosen from how well the *new* model `m_{k+1}`
retrospectively predicts `f` at the *previous* iterate, rather than from how
well `m_k` predicted `f` at the new one. Acceptance is still decided by ρ and η;
only the scaling uses ρ̃.

On a rejected step (`accepted = false`), ρ̃ is undefined — it judges a step that
was not taken — and the solver passes ρ instead, so the rule branches on the
flag first and uses only the *sign* of the ratio, as the source rule does:

    rejected, ρ ≥ 0     →  Δ ← γ₂ ‖s_k‖
    rejected, ρ < 0     →  Δ ← min(γ₂‖s_k‖, γ₁Δ_k)

and on an accepted step

    ρ̃ ≥ η̃₂             →  Δ ← max(γ₃‖s_k‖, Δ_k)
    η̃₁ ≤ ρ̃ < η̃₂        →  Δ ← Δ_k
    0 ≤ ρ̃ < η̃₁         →  Δ ← γ₂‖s_k‖
    ρ̃ < 0              →  Δ ← min(γ₂‖s_k‖, γ₁Δ_k)

The safeguard factor `θ_{k-1}` of Part I is taken as `0`, so `max[γ₁, θ] = γ₁`.

Branching on `accepted` is mandatory, not defensive. Comparing a *classical* ρ
against the *retrospective* thresholds would leave the radius untouched for
`ρ ∈ [η̃₁, η)` — an unsuccessful iteration with no contraction, which forfeits
weak admissibility and, since a rejected step changes neither the model nor the
iterate, reproduces the same subproblem for ever.

The first accepted branch is the interesting one: `max(γ₃‖s_k‖, Δ_k)` is
simultaneously step-driven *and* non-decreasing, which is what lets this rule
secure eventual inactivity from a *consequence* of the standing assumptions
(ρ̃ → 1) rather than from a condition on a user parameter, as `RDFO` needs. The
price is structural: ρ̃ → 1 requires either the secant condition on the model,
or asymptotic second-order coherence plus a quadratic model decrease.
"""
mutable struct RRTR <: RadiusRule
    γ₁::Float64
    γ₂::Float64
    γ₃::Float64
    η̃₁::Float64
    η̃₂::Float64
    Δmax::Float64
    Δmin::Float64
    function RRTR(; γ₁::Float64 = 0.0625, γ₂::Float64 = 0.25, γ₃::Float64 = 2.5,
                    η̃₁::Float64 = 0.05, η̃₂::Float64 = 0.9,
                    Δmax::Float64 = Inf, Δmin::Float64 = 1e-14)
        check_factors(:RRTR; γ₁ = γ₁, γ₂ = γ₂, γ₃ = γ₃)
        0 < η̃₁ <= η̃₂ < 1 || throw(ArgumentError("RRTR: need 0 < η̃₁ ≤ η̃₂ < 1"))
        Δmin >= 0 || throw(ArgumentError("RRTR: need Δmin ≥ 0"))
        new(γ₁, γ₂, γ₃, η̃₁, η̃₂, Δmax, Δmin)
    end
end

needs_retrospective(::RRTR) = true
asymptotic_regime(::RRTR) = :step_summable

function update_radius!(r::RRTR, Δ::Float64, ρ̃::Float64, accepted::Bool,
                        ::Float64, ::Float64,
                        s_norm::Float64, ::Float64, ::Float64)
    Δnew = if !accepted
               ρ̃ < 0 ? min(r.γ₂ * s_norm, r.γ₁ * Δ) : r.γ₂ * s_norm
           elseif ρ̃ >= r.η̃₂
               max(r.γ₃ * s_norm, Δ)
           elseif ρ̃ >= r.η̃₁
               Δ
           elseif ρ̃ >= 0
               r.γ₂ * s_norm
           else
               min(r.γ₂ * s_norm, r.γ₁ * Δ)
           end
    return clamp(Δnew, r.Δmin, r.Δmax)
end

"""
    RRTRGrad(; γ₁ = 0.25, γ₃ = 2.0, μ = 1.0, η̃₁ = 0.05, η̃₂ = 0.9,
               half_test = true)

Retrospective gradient-scaled rule of Fan, Pan & Song (2016):

    μ_{k+1} = γ₁ μ_k     if the step was rejected, or ρ̃ < η̃₁
              γ₃ μ_k     if ρ̃ ≥ η̃₂ and ‖s_k‖ > ½Δ_k
              μ_k        otherwise
    Δ_{k+1} = μ_{k+1} ‖g_{k+1}‖

Structurally [`RGrad`](@ref) with ρ replaced by ρ̃ and no intermediate
contraction, so `γ₂` is absent from this rule. The argument that μ cannot
diverge is *ratio-agnostic* — it uses only the monotone decrease of `f`, the
local quadratic bounds, and the ½Δ test — so it transfers unchanged from `RGrad`
to this rule. All the ratio has to supply is convergence to 1.

The explicit rejected branch matters for the same reason as in [`RRTR`](@ref):
on a rejected step `‖g_{k+1}‖ = ‖g_k‖`, so leaving μ untouched returns exactly
`Δ_k` and the solver cannot make progress.
"""
mutable struct RRTRGrad <: RadiusRule
    γ₁::Float64
    γ₃::Float64
    μ::Float64
    μ₀::Float64
    η̃₁::Float64
    η̃₂::Float64
    half_test::Bool
    function RRTRGrad(; γ₁::Float64 = 0.25, γ₃::Float64 = 2.0, μ::Float64 = 1.0,
                        η̃₁::Float64 = 0.05, η̃₂::Float64 = 0.9,
                        half_test::Bool = true)
        check_factors(:RRTRGrad; γ₁ = γ₁, γ₃ = γ₃)
        μ > 0 || throw(ArgumentError("RRTRGrad: need μ > 0, got $μ"))
        0 < η̃₁ <= η̃₂ < 1 || throw(ArgumentError("RRTRGrad: need 0 < η̃₁ ≤ η̃₂ < 1"))
        new(γ₁, γ₃, μ, μ, η̃₁, η̃₂, half_test)
    end
end

needs_retrospective(::RRTRGrad) = true
initial_radius(r::RRTRGrad, ::Float64, g_norm::Float64) = r.μ * g_norm
reset_rule!(r::RRTRGrad) = (r.μ = r.μ₀; nothing)
is_criticality_anchored(::RRTRGrad) = true
asymptotic_regime(::RRTRGrad) = :vanishing

function update_radius!(r::RRTRGrad, Δ::Float64, ρ̃::Float64, accepted::Bool,
                        ::Float64, ::Float64,
                        s_norm::Float64, ::Float64, g_norm_new::Float64)
    if !accepted || ρ̃ < r.η̃₁
        r.μ *= r.γ₁
    elseif ρ̃ >= r.η̃₂ && (!r.half_test || s_norm > 0.5 * Δ)
        r.μ *= r.γ₃
    end
    return r.μ * g_norm_new
end

# =============================================================================
# Display
# =============================================================================

_rule_fields(r) = join(("$(f) = $(getfield(r, f))" for f in fieldnames(typeof(r))), ", ")

Base.show(io::IO, r::RadiusRule) = print(io, nameof(typeof(r)), "(", _rule_fields(r), ")")

function Base.show(io::IO, ::MIME"text/plain", r::RadiusRule)
    println(io, nameof(typeof(r)), ":")
    for f in fieldnames(typeof(r))
        println(io, "  ", rpad(string(f), 10), " ", getfield(r, f))
    end
    println(io, "  (asymptotic regime: ", asymptotic_regime(r), ")")
    needs_retrospective(r) && println(io, "  (driven by the retrospective ratio ρ̃)")
end
