# =============================================================================
# src/Radius_updates/rules.jl
#
# Every trust-region radius update mechanism, under its primary name.
#
#   RDelta        criticality-blind, fixed factor        (Conn–Gould–Toint)
#   RStep         step-anchored                          (Hei-style, constant factors)
#   RDFO          criticality-anchored, parameter ζ      (Scheinberg et al.)
#   RGrad         gradient-scaled, Δ = μ‖g‖, μ uncapped  (Fan–Yuan)
#   RGradCapped   gradient-scaled with μ ≤ μ_max
#   RAdaptiveStep step-anchored, exponential factor      (Hei 2003)
#   RAdaptiveGrad gradient-anchored, exponential factor
#   RAdaptiveFanYuan  μ *= R_exp(ρ), Δ = μ‖g‖            (Hei + Fan–Yuan)
#   RRTR          retrospective                          (Bastin et al. 2010)
#   RRTRGrad      retrospective gradient-scaled          (Fan–Pan–Song 2016)
#
# All are concrete subtypes of `RadiusRule` and implement
#
#     initial_radius(rule, Δ₀, g_norm)                          -> Float64
#     update_radius!(rule, Δ, ρ, η₁, η₂, s_norm, g_old, g_new)  -> Float64
#     reset_rule!(rule)                                          -> nothing
#
# Retrospective rules additionally implement `needs_retrospective(rule) = true`
# and receive ρ̃ in place of ρ; see `retrospective.jl`.
# =============================================================================

# -----------------------------------------------------------------------------
# Interface
# -----------------------------------------------------------------------------

"""
    RadiusRule

Abstract supertype for trust-region radius update mechanisms.

A concrete `MyRule <: RadiusRule` must implement

    initial_radius(rule, Δ₀, g_norm) -> Float64
    update_radius!(rule, Δ, ρ, η₁, η₂, s_norm, g_norm_old, g_norm_new) -> Float64

and, if it carries mutable state,

    reset_rule!(rule) -> nothing

`g_norm_old` is ‖g_k‖ *before* the accept/reject decision and `g_norm_new` is
‖g_{k+1}‖ *after* it. Rules anchored to criticality at the current iterate
(`RDFO`) use the former; rules that set the next radius from the next gradient
(`RGrad` and relatives) use the latter.

Rules that must be judged by the retrospective ratio ρ̃ instead of ρ declare
`needs_retrospective(rule) = true`.
"""
abstract type RadiusRule end

"""
    initial_radius(rule, Δ₀, g_norm) -> Float64

Radius at iteration 0. Most rules return `Δ₀`; those of the form
`Δ = μ‖g‖` return `μ₀ · g_norm` and so ignore `Δ₀` entirely.
"""
initial_radius(::RadiusRule, Δ₀::Float64, ::Float64) = Δ₀

"""
    update_radius!(rule, Δ, ρ, η₁, η₂, s_norm, g_norm_old, g_norm_new) -> Float64

Radius for the next iteration. May mutate `rule` (e.g. the multiplier μ).
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
costs one extra model evaluation at the previous iterate.
"""
needs_retrospective(::RadiusRule) = false

"""
    is_criticality_anchored(rule) -> Bool

`true` if the radius is tied to a criticality measure, so that `Δ_k → 0` along
a convergent run. For these rules eventual inactivity of the trust region
requires a parameter above the problem-dependent threshold κ̄ = 4/λ*_min; for
the others `liminf Δ_k > 0` and inactivity is automatic.
"""
is_criticality_anchored(::RadiusRule) = false

# =============================================================================
# RDelta — classical multiplicative update
# =============================================================================

"""
    RDelta(; γ₁ = 0.25, γ₂ = 0.5, γ₃ = 2.0, Δmax = Inf)

Classical three-case multiplicative update (Conn, Gould & Toint §6.1):

    ρ ≥ η₂       →  Δ ← γ₃ Δ
    η₁ ≤ ρ < η₂  →  Δ ← γ₂ Δ
    ρ < η₁       →  Δ ← γ₁ Δ

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
        @assert 0 < γ₁ < 1 "RDelta: need 0 < γ₁ < 1"
        @assert γ₂ > 0     "RDelta: need γ₂ > 0"
        @assert γ₃ >= 1    "RDelta: need γ₃ ≥ 1"
        new(γ₁, γ₂, γ₃, Δmax)
    end
end

# positional constructor, for compatibility with existing config files
RDelta(γ₁::Float64, γ₂::Float64, γ₃::Float64) = RDelta(; γ₁ = γ₁, γ₂ = γ₂, γ₃ = γ₃)

function update_radius!(r::RDelta, Δ::Float64, ρ::Float64, η₁::Float64, η₂::Float64,
                        ::Float64, ::Float64, ::Float64)
    ρ >= η₂ && return min(r.γ₃ * Δ, r.Δmax)
    ρ >= η₁ && return r.γ₂ * Δ
    return r.γ₁ * Δ
end

# =============================================================================
# RStep — step-anchored update with constant factors
# =============================================================================

"""
    RStep(; γ₁ = 0.25, γ₂ = 0.5, γ₃ = 2.0, Δmax = Inf, Δmin = 1e-14)

Radius proportional to the accepted step:

    ρ < η₁       →  Δ ← γ₁ Δ          (rejected: contract the radius)
    η₁ ≤ ρ < η₂  →  Δ ← γ₂ ‖s_k‖
    ρ ≥ η₂       →  Δ ← γ₃ ‖s_k‖

!!! warning "Δmin exists for a reason"
    On an accepted step the new radius is proportional to `‖s_k‖`, so a short
    step gives a small radius, which gives a shorter step still. Near
    convergence, or when a truncated-CG subsolver stops on its first iteration,
    this can collapse to zero, after which every step is zero, ρ is `NaN`, and
    the solver spins to `max_iterations`. `Δmin` floors the radius and turns
    that silent hang into ordinary slow progress. Set `Δmin = 0.0` to reproduce
    the unguarded rule exactly.
"""
mutable struct RStep <: RadiusRule
    γ₁::Float64
    γ₂::Float64
    γ₃::Float64
    Δmax::Float64
    Δmin::Float64
    function RStep(; γ₁::Float64 = 0.25, γ₂::Float64 = 0.5, γ₃::Float64 = 2.0,
                     Δmax::Float64 = Inf, Δmin::Float64 = 1e-14)
        @assert 0 < γ₁ < 1  "RStep: need 0 < γ₁ < 1"
        @assert γ₃ > γ₂ > 0 "RStep: need γ₃ > γ₂ > 0"
        @assert Δmin >= 0   "RStep: need Δmin ≥ 0"
        new(γ₁, γ₂, γ₃, Δmax, Δmin)
    end
end

RStep(γ₁::Float64, γ₂::Float64, γ₃::Float64) = RStep(; γ₁ = γ₁, γ₂ = γ₂, γ₃ = γ₃)

function update_radius!(r::RStep, Δ::Float64, ρ::Float64, η₁::Float64, η₂::Float64,
                        s_norm::Float64, ::Float64, ::Float64)
    Δnew = ρ < η₁ ? r.γ₁ * Δ :
           ρ < η₂ ? r.γ₂ * s_norm :
                    r.γ₃ * s_norm
    return clamp(Δnew, r.Δmin, r.Δmax)
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
        @assert 0 < γ₁ < 1 "RDFO: need 0 < γ₁ < 1"
        @assert γ₃ >= 1    "RDFO: need γ₃ ≥ 1"
        @assert ζ > 0      "RDFO: need ζ > 0"
        new(γ₁, γ₂, γ₃, ζ, Δmax)
    end
end

RDFO(γ₁::Float64, γ₂::Float64, γ₃::Float64, ζ::Float64) =
    RDFO(; γ₁ = γ₁, γ₂ = γ₂, γ₃ = γ₃, ζ = ζ)

function update_radius!(r::RDFO, Δ::Float64, ρ::Float64, η₁::Float64, ::Float64,
                        ::Float64, g_norm_old::Float64, ::Float64)
    ρ < η₁ && return r.γ₁ * Δ
    Δ > r.ζ * g_norm_old && return r.γ₂ * Δ
    return min(r.γ₃ * Δ, r.Δmax)
end

is_criticality_anchored(::RDFO) = true

# =============================================================================
# RGrad — gradient-scaled, μ uncapped
# =============================================================================

"""
    RGrad(; γ₁ = 0.25, γ₂ = 2.0, μ = 1.0, half_test = true)

Gradient-scaled radius `Δ_k = μ_k ‖g_k‖` with an unbounded multiplier:

    ρ ≥ η₂ and ‖s_k‖ > ½Δ_k  →  μ ← γ₂ μ
    ρ < η₁                    →  μ ← γ₁ μ
    otherwise                 →  μ unchanged

Here `μ_k` *is* the radius-to-criticality ratio `Δ_k/‖g_k‖`, so the rule
performs a geometric search for the inactivity threshold κ̄ without knowing it:
μ climbs until the trust region stops binding, then stops climbing. Because μ
is uncapped it crosses any threshold eventually, which makes eventual
inactivity **unconditional** — the only rule in the survey for which this holds
without a hypothesis on an unknowable constant.

The guard `‖s_k‖ > ½Δ_k` is not cosmetic. It is what converts "μ grows" into
"‖s_k‖ is comparable to μ_k‖g_k‖", which is the step that makes the boundedness
argument for μ work. Any constant in (0,1) serves; `half_test = false` disables
the guard, which breaks that argument.

See [`RGradCapped`](@ref) for the bounded variant required by the global
asymptotic theory (`Δ_k → 0` needs `μ_k ≤ μ̄`).
"""
mutable struct RGrad <: RadiusRule
    γ₁::Float64
    γ₂::Float64
    μ::Float64
    μ₀::Float64
    half_test::Bool
    function RGrad(; γ₁::Float64 = 0.25, γ₂::Float64 = 2.0, μ::Float64 = 1.0,
                     half_test::Bool = true)
        @assert 0 < γ₁ < 1 "RGrad: need 0 < γ₁ < 1"
        @assert γ₂ > 1     "RGrad: need γ₂ > 1"
        @assert μ > 0      "RGrad: need μ > 0"
        new(γ₁, γ₂, μ, μ, half_test)
    end
end

RGrad(γ₁::Float64, γ₂::Float64, μ::Float64) = RGrad(; γ₁ = γ₁, γ₂ = γ₂, μ = μ)

initial_radius(r::RGrad, ::Float64, g_norm::Float64) = r.μ * g_norm
reset_rule!(r::RGrad) = (r.μ = r.μ₀; nothing)
is_criticality_anchored(::RGrad) = true

function update_radius!(r::RGrad, Δ::Float64, ρ::Float64, η₁::Float64, η₂::Float64,
                        s_norm::Float64, ::Float64, g_norm_new::Float64)
    if ρ >= η₂ && (!r.half_test || s_norm > 0.5 * Δ)
        r.μ *= r.γ₂
    elseif ρ < η₁
        r.μ *= r.γ₁
    end
    return r.μ * g_norm_new
end

# =============================================================================
# RGradCapped — gradient-scaled with μ ≤ μ_max
# =============================================================================

"""
    RGradCapped(; γ₁ = 0.25, γ₂ = 2.0, μ = 1.0, μ_max = 1.0, half_test = true)

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
    μ::Float64
    μ₀::Float64
    μ_max::Float64
    half_test::Bool
    function RGradCapped(; γ₁::Float64 = 0.25, γ₂::Float64 = 2.0, μ::Float64 = 1.0,
                           μ_max::Float64 = 1.0, half_test::Bool = true)
        @assert 0 < γ₁ < 1  "RGradCapped: need 0 < γ₁ < 1"
        @assert γ₂ > 1      "RGradCapped: need γ₂ > 1"
        @assert μ > 0       "RGradCapped: need μ > 0"
        @assert μ_max >= μ  "RGradCapped: need μ_max ≥ μ"
        new(γ₁, γ₂, μ, μ, μ_max, half_test)
    end
end

initial_radius(r::RGradCapped, ::Float64, g_norm::Float64) = r.μ * g_norm
reset_rule!(r::RGradCapped) = (r.μ = r.μ₀; nothing)
is_criticality_anchored(::RGradCapped) = true

function update_radius!(r::RGradCapped, Δ::Float64, ρ::Float64, η₁::Float64, η₂::Float64,
                        s_norm::Float64, ::Float64, g_norm_new::Float64)
    if ρ >= η₂ && (!r.half_test || s_norm > 0.5 * Δ)
        r.μ = min(r.γ₂ * r.μ, r.μ_max)
    elseif ρ < η₁
        r.μ *= r.γ₁
    end
    return r.μ * g_norm_new
end

# =============================================================================
# Hei family — piecewise-exponential factor
# =============================================================================

"""
    _r_exp(t, η, β, γ₁, γ₂, M, λ₁, λ₂) -> Float64

Piecewise-exponential multiplier of Hei (2003):

    t < η  →  β + (1 − γ₁ − β) exp(λ₁(t − η))
    t ≥ η  →  1 + γ₂ + (M − 1 − γ₂)(1 − exp(−λ₂(t − η)))

Continuous in `t`, bounded below by `β` and above by `M`. Unlike the three-case
rules it varies the factor *smoothly* with ρ instead of bucketing it.
"""
@inline function _r_exp(t::Float64, η::Float64, β::Float64,
                        γ₁::Float64, γ₂::Float64, M::Float64,
                        λ₁::Float64, λ₂::Float64)
    if t < η
        return β + (1.0 - γ₁ - β) * exp(λ₁ * (t - η))
    else
        return 1.0 + γ₂ + (M - (1.0 + γ₂)) * (1.0 - exp(-λ₂ * (t - η)))
    end
end

"Default Hei constants, shared by the three adaptive rules."
const HEI_DEFAULTS = (η = 0.25, β = 0.0625, γ₁ = 0.25, γ₂ = 0.5,
                      M = 4.0, λ₁ = 5.0, λ₂ = 5.0)

"""
    RAdaptiveStep(; η, β, γ₁, γ₂, M, λ₁, λ₂, Δmin = 1e-14)

Hei (2003) step-anchored rule with a smooth factor:

    Δ_{k+1} = R_exp(ρ_k) · ‖s_k‖

Step-anchored like [`RStep`](@ref), so the same `Δmin` guard applies and for
the same reason.
"""
mutable struct RAdaptiveStep <: RadiusRule
    η::Float64
    β::Float64
    γ₁::Float64
    γ₂::Float64
    M::Float64
    λ₁::Float64
    λ₂::Float64
    Δmin::Float64
    function RAdaptiveStep(; η::Float64 = HEI_DEFAULTS.η, β::Float64 = HEI_DEFAULTS.β,
                             γ₁::Float64 = HEI_DEFAULTS.γ₁, γ₂::Float64 = HEI_DEFAULTS.γ₂,
                             M::Float64 = HEI_DEFAULTS.M, λ₁::Float64 = HEI_DEFAULTS.λ₁,
                             λ₂::Float64 = HEI_DEFAULTS.λ₂, Δmin::Float64 = 1e-14)
        @assert M > 1 + γ₂ "RAdaptiveStep: need M > 1 + γ₂ so the factor can exceed 1"
        new(η, β, γ₁, γ₂, M, λ₁, λ₂, Δmin)
    end
end

RAdaptiveStep(η::Float64, β::Float64, γ₁::Float64, γ₂::Float64,
              M::Float64, λ₁::Float64, λ₂::Float64) =
    RAdaptiveStep(; η = η, β = β, γ₁ = γ₁, γ₂ = γ₂, M = M, λ₁ = λ₁, λ₂ = λ₂)

function update_radius!(r::RAdaptiveStep, ::Float64, ρ::Float64, ::Float64, ::Float64,
                        s_norm::Float64, ::Float64, ::Float64)
    f = _r_exp(ρ, r.η, r.β, r.γ₁, r.γ₂, r.M, r.λ₁, r.λ₂)
    return max(f * s_norm, r.Δmin)
end

"""
    RAdaptiveGrad(; η, β, γ₁, γ₂, M, λ₁, λ₂)

Gradient-anchored Hei variant:

    Δ_{k+1} = R_exp(ρ_k) · ‖g_{k+1}‖

Uses the gradient *after* the accept/reject decision. Criticality-anchored, so
`Δ_k → 0`.
"""
mutable struct RAdaptiveGrad <: RadiusRule
    η::Float64
    β::Float64
    γ₁::Float64
    γ₂::Float64
    M::Float64
    λ₁::Float64
    λ₂::Float64
    function RAdaptiveGrad(; η::Float64 = HEI_DEFAULTS.η, β::Float64 = HEI_DEFAULTS.β,
                             γ₁::Float64 = HEI_DEFAULTS.γ₁, γ₂::Float64 = HEI_DEFAULTS.γ₂,
                             M::Float64 = HEI_DEFAULTS.M, λ₁::Float64 = HEI_DEFAULTS.λ₁,
                             λ₂::Float64 = HEI_DEFAULTS.λ₂)
        @assert M > 1 + γ₂ "RAdaptiveGrad: need M > 1 + γ₂"
        new(η, β, γ₁, γ₂, M, λ₁, λ₂)
    end
end

RAdaptiveGrad(η::Float64, β::Float64, γ₁::Float64, γ₂::Float64,
              M::Float64, λ₁::Float64, λ₂::Float64) =
    RAdaptiveGrad(; η = η, β = β, γ₁ = γ₁, γ₂ = γ₂, M = M, λ₁ = λ₁, λ₂ = λ₂)

function update_radius!(r::RAdaptiveGrad, ::Float64, ρ::Float64, ::Float64, ::Float64,
                        ::Float64, ::Float64, g_norm_new::Float64)
    return _r_exp(ρ, r.η, r.β, r.γ₁, r.γ₂, r.M, r.λ₁, r.λ₂) * g_norm_new
end

is_criticality_anchored(::RAdaptiveGrad) = true

"""
    RAdaptiveFanYuan(; μ = 1.0, η, β, γ₁, γ₂, M, λ₁, λ₂)

Hei factor driving a Fan–Yuan multiplier:

    μ_{k+1} = μ_k · R_exp(ρ_k),    Δ_{k+1} = μ_{k+1} · ‖g_{k+1}‖

Combines the smooth factor of the Hei family with the ratio-tracking of
[`RGrad`](@ref): μ is again the radius-to-criticality ratio, but it is scaled
continuously in ρ rather than by one of two constants.
"""
mutable struct RAdaptiveFanYuan <: RadiusRule
    μ::Float64
    μ₀::Float64
    η::Float64
    β::Float64
    γ₁::Float64
    γ₂::Float64
    M::Float64
    λ₁::Float64
    λ₂::Float64
    function RAdaptiveFanYuan(; μ::Float64 = 1.0, η::Float64 = HEI_DEFAULTS.η,
                                β::Float64 = HEI_DEFAULTS.β, γ₁::Float64 = HEI_DEFAULTS.γ₁,
                                γ₂::Float64 = HEI_DEFAULTS.γ₂, M::Float64 = HEI_DEFAULTS.M,
                                λ₁::Float64 = HEI_DEFAULTS.λ₁, λ₂::Float64 = HEI_DEFAULTS.λ₂)
        @assert μ > 0      "RAdaptiveFanYuan: need μ > 0"
        @assert M > 1 + γ₂ "RAdaptiveFanYuan: need M > 1 + γ₂"
        new(μ, μ, η, β, γ₁, γ₂, M, λ₁, λ₂)
    end
end

RAdaptiveFanYuan(μ::Float64, η::Float64, β::Float64, γ₁::Float64,
                 γ₂::Float64, M::Float64, λ₁::Float64, λ₂::Float64) =
    RAdaptiveFanYuan(; μ = μ, η = η, β = β, γ₁ = γ₁, γ₂ = γ₂, M = M, λ₁ = λ₁, λ₂ = λ₂)

initial_radius(r::RAdaptiveFanYuan, ::Float64, g_norm::Float64) = r.μ * g_norm
reset_rule!(r::RAdaptiveFanYuan) = (r.μ = r.μ₀; nothing)
is_criticality_anchored(::RAdaptiveFanYuan) = true

function update_radius!(r::RAdaptiveFanYuan, ::Float64, ρ::Float64, ::Float64, ::Float64,
                        ::Float64, ::Float64, g_norm_new::Float64)
    r.μ *= _r_exp(ρ, r.η, r.β, r.γ₁, r.γ₂, r.M, r.λ₁, r.λ₂)
    return r.μ * g_norm_new
end

# =============================================================================
# Retrospective family
# =============================================================================

"""
    RRTR(; γ₀ = 0.0625, γ₁ = 0.25, γ₂ = 2.5, η̃₁ = 0.05, η̃₂ = 0.9, Δmax = Inf)

Retrospective update of Bastin, Malmedy, Mouffe, Toint & Tomanos (2010).

The radius for iteration `k+1` is chosen from how well the *new* model `m_{k+1}`
retrospectively predicts `f` at the *previous* iterate, rather than from how
well `m_k` predicted `f` at the new one. Acceptance is still decided by ρ; only
the radius update uses ρ̃.

    ρ̃ ≥ η̃₂             →  Δ ← max(γ₂‖s_k‖, Δ_k)
    η̃₁ ≤ ρ̃ < η̃₂        →  Δ ← Δ_k
    0 ≤ ρ̃ < η̃₁         →  Δ ← γ₁‖s_k‖
    ρ̃ < 0               →  Δ ← min(γ₁‖s_k‖, γ₀Δ_k)

Rejected steps contract as usual, `Δ ← γ₁‖s_k‖`.

The first branch is the interesting one: `max(γ₂‖s_k‖, Δ_k)` is simultaneously
step-driven *and* non-decreasing, which is what lets this rule secure eventual
inactivity from a *consequence* of the standing assumptions (ρ̃ → 1) rather than
from a condition on a user parameter, as `RDFO` needs. The price is structural:
ρ̃ → 1 requires either the secant condition on the model, or asymptotic
second-order coherence plus a quadratic model decrease.
"""
mutable struct RRTR <: RadiusRule
    γ₀::Float64
    γ₁::Float64
    γ₂::Float64
    η̃₁::Float64
    η̃₂::Float64
    Δmax::Float64
    Δmin::Float64
    function RRTR(; γ₀::Float64 = 0.0625, γ₁::Float64 = 0.25, γ₂::Float64 = 2.5,
                    η̃₁::Float64 = 0.05, η̃₂::Float64 = 0.9,
                    Δmax::Float64 = Inf, Δmin::Float64 = 1e-14)
        @assert 0 < γ₀ < γ₁ <= 1 <= γ₂ "RRTR: need 0 < γ₀ < γ₁ ≤ 1 ≤ γ₂"
        @assert 0 < η̃₁ <= η̃₂ < 1      "RRTR: need 0 < η̃₁ ≤ η̃₂ < 1"
        new(γ₀, γ₁, γ₂, η̃₁, η̃₂, Δmax, Δmin)
    end
end

needs_retrospective(::RRTR) = true

function update_radius!(r::RRTR, Δ::Float64, ρ̃::Float64, ::Float64, ::Float64,
                        s_norm::Float64, ::Float64, ::Float64)
    Δnew = if ρ̃ >= r.η̃₂
        max(r.γ₂ * s_norm, Δ)
    elseif ρ̃ >= r.η̃₁
        Δ
    elseif ρ̃ >= 0
        r.γ₁ * s_norm
    else
        min(r.γ₁ * s_norm, r.γ₀ * Δ)
    end
    return clamp(Δnew, r.Δmin, r.Δmax)
end

"""
    RRTRGrad(; γ₁ = 0.25, γ₂ = 2.0, μ = 1.0, η̃₁ = 0.05, η̃₂ = 0.9, half_test = true)

Retrospective gradient-scaled rule of Fan, Pan & Song (2016):

    μ_{k+1} = γ₁ μ_k                     if ρ̃ < η̃₁
              γ₂ μ_k                     if ρ̃ ≥ η̃₂ and ‖s_k‖ > ½Δ_k
              μ_k                        otherwise
    Δ_{k+1} = μ_{k+1} ‖g_{k+1}‖

Structurally [`RGrad`](@ref) with ρ replaced by ρ̃. The argument that μ cannot
diverge is *ratio-agnostic* — it uses only the monotone decrease of `f`, the
local quadratic bounds, and the ½Δ test — so it transfers unchanged from `RGrad`
to this rule. All the ratio has to supply is convergence to 1.
"""
mutable struct RRTRGrad <: RadiusRule
    γ₁::Float64
    γ₂::Float64
    μ::Float64
    μ₀::Float64
    η̃₁::Float64
    η̃₂::Float64
    half_test::Bool
    function RRTRGrad(; γ₁::Float64 = 0.25, γ₂::Float64 = 2.0, μ::Float64 = 1.0,
                        η̃₁::Float64 = 0.05, η̃₂::Float64 = 0.9, half_test::Bool = true)
        @assert 0 < γ₁ < 1        "RRTRGrad: need 0 < γ₁ < 1"
        @assert γ₂ > 1            "RRTRGrad: need γ₂ > 1"
        @assert μ > 0             "RRTRGrad: need μ > 0"
        @assert 0 < η̃₁ <= η̃₂ < 1 "RRTRGrad: need 0 < η̃₁ ≤ η̃₂ < 1"
        new(γ₁, γ₂, μ, μ, η̃₁, η̃₂, half_test)
    end
end

needs_retrospective(::RRTRGrad) = true
initial_radius(r::RRTRGrad, ::Float64, g_norm::Float64) = r.μ * g_norm
reset_rule!(r::RRTRGrad) = (r.μ = r.μ₀; nothing)
is_criticality_anchored(::RRTRGrad) = true

function update_radius!(r::RRTRGrad, Δ::Float64, ρ̃::Float64, ::Float64, ::Float64,
                        s_norm::Float64, ::Float64, g_norm_new::Float64)
    if ρ̃ < r.η̃₁
        r.μ *= r.γ₁
    elseif ρ̃ >= r.η̃₂ && (!r.half_test || s_norm > 0.5 * Δ)
        r.μ *= r.γ₂
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
    is_criticality_anchored(r) && println(io, "  (criticality-anchored: Δₖ → 0)")
    needs_retrospective(r)     && println(io, "  (driven by the retrospective ratio ρ̃)")
end
