
# ============================================================
# Canonical trust-region radius update rules R1–R4
#
# These four rules correspond exactly to the mechanisms
# analysed in the companion COAP papers (Part I & Part II).
# They are dispatched through update_radius!(rule, ...) rather
# than the legacy trust_region_update!(state, params, ...).
# ============================================================

"""
    AbstractRadiusUpdate

Abstract supertype for the four canonical radius update rules R1–R4.
Concrete subtypes implement `update_radius!(rule, Δ, ρ, η₁, η₂, s_norm, g_norm_old, g_norm_new)`.
"""
abstract type AbstractRadiusUpdate end

# ------------------------------------------------------------
# R1 — Classical multiplicative update (Conn–Gould–Toint)
# ------------------------------------------------------------
"""
    R1ClassicalUpdate <: AbstractRadiusUpdate

Three-case multiplicative radius update (Conn, Gould & Toint §6.1):

    ρ ≥ η₂  →  Δ ← γ₃ · Δ   (very successful step, expand)
    η₁ ≤ ρ < η₂  →  Δ ← γ₂ · Δ   (acceptable step, maintain/adjust)
    ρ < η₁  →  Δ ← γ₁ · Δ   (poor step, contract)

Standard values: γ₁ < 1, γ₂ ≤ 1 ≤ γ₃.

# Fields
- `γ₁`: contraction factor when ρ < η₁
- `γ₂`: adjustment factor when η₁ ≤ ρ < η₂
- `γ₃`: expansion factor when ρ ≥ η₂
"""
struct R1ClassicalUpdate <: AbstractRadiusUpdate
    γ₁::Float64
    γ₂::Float64
    γ₃::Float64
    function R1ClassicalUpdate(γ₁::Float64 = 0.25, γ₂::Float64 = 0.5, γ₃::Float64 = 2.0)
        @assert 0 < γ₁ < 1  "R1: need 0 < γ₁ < 1"
        @assert γ₃ >= 1     "R1: need γ₃ ≥ 1"
        new(γ₁, γ₂, γ₃)
    end
end

# ------------------------------------------------------------
# R2 — Step-size proportional update
# ------------------------------------------------------------
"""
    R2StepSizeUpdate <: AbstractRadiusUpdate

Radius set proportional to the step norm ‖s_k‖:

    ρ < η₁   →  Δ ← γ₁ · Δ          (rejected step, contract current radius)
    η₁ ≤ ρ < η₂  →  Δ ← γ₂ · ‖s_k‖  (acceptable step)
    ρ ≥ η₂   →  Δ ← γ₃ · ‖s_k‖      (very successful step)

The radius tracks the last accepted step magnitude rather than
multiplying the previous radius.

# Fields
- `γ₁`: Δ contraction factor when step is rejected (ρ < η₁)
- `γ₂`: ‖s‖ multiplier for acceptable steps (η₁ ≤ ρ < η₂)
- `γ₃`: ‖s‖ multiplier for very successful steps (ρ ≥ η₂)
"""
struct R2StepSizeUpdate <: AbstractRadiusUpdate
    γ₁::Float64
    γ₂::Float64
    γ₃::Float64
    function R2StepSizeUpdate(γ₁::Float64 = 0.25, γ₂::Float64 = 0.5, γ₃::Float64 = 2.0)
        @assert 0 < γ₁ < 1  "R2: need 0 < γ₁ < 1"
        @assert γ₃ > γ₂ > 0 "R2: need γ₃ > γ₂ > 0"
        new(γ₁, γ₂, γ₃)
    end
end

# ------------------------------------------------------------
# R3 — DFO-like gradient-radius comparison (Scheinberg)
# ------------------------------------------------------------
"""
    R3DFOLikeUpdate <: AbstractRadiusUpdate

Radius update that compares Δ_k to ζ · ‖g_k‖ before expanding:

    ρ ≥ η₁  and  Δ_k ≤ ζ · ‖g_k‖  →  accept, Δ ← γ₃ · Δ  (expand)
    ρ ≥ η₁  and  Δ_k  > ζ · ‖g_k‖  →  accept, Δ ← γ₂ · Δ  (do not expand)
    ρ < η₁                           →  reject, Δ ← γ₁ · Δ  (contract)

The gradient norm used in the comparison is ‖g_k‖ evaluated
*before* the acceptance decision (g_norm_old).

# Fields
- `γ₁`: contraction factor when step is rejected
- `γ₂`: Δ multiplier when ρ ≥ η₁ but Δ > ζ‖g‖ (prevent over-expansion)
- `γ₃`: expansion factor when ρ ≥ η₁ and Δ ≤ ζ‖g‖
- `ζ`:  threshold ratio controlling when the radius may grow
"""
struct R3DFOLikeUpdate <: AbstractRadiusUpdate
    γ₁::Float64
    γ₂::Float64
    γ₃::Float64
    ζ::Float64
    function R3DFOLikeUpdate(γ₁::Float64 = 0.25, γ₂::Float64 = 0.5,
                              γ₃::Float64 = 2.0, ζ::Float64 = 1.0)
        @assert 0 < γ₁ < 1  "R3: need 0 < γ₁ < 1"
        @assert γ₃ >= 1     "R3: need γ₃ ≥ 1"
        @assert ζ > 0       "R3: need ζ > 0"
        new(γ₁, γ₂, γ₃, ζ)
    end
end

# ------------------------------------------------------------
# R4 — Relative-gradient update (Yuan–Fan)
# ------------------------------------------------------------
"""
    R4RelativeGradUpdate <: AbstractRadiusUpdate

Radius defined as Δ_{k+1} = μ_{k+1} · ‖g_{k+1}‖, where the multiplier μ
is updated multiplicatively:

    ρ ≥ η₂  and  ‖s_k‖ > 0.5 · Δ_k  →  μ ← γ₂ · μ  (expand)
    ρ < η₁                            →  μ ← γ₁ · μ  (contract)
    otherwise                         →  μ unchanged

The guard ‖s_k‖ > 0.5 · Δ_k prevents expansion when the CG step was
already well inside the trust region, signalling a near-quadratic
objective where a larger radius would not help.

The gradient norm used is ‖g_{k+1}‖ evaluated *after* the
acceptance decision (g_norm_new).

# Fields
- `γ₁`: μ contraction factor when ρ < η₁
- `γ₂`: μ expansion factor when ρ ≥ η₂ and ‖s‖ > 0.5·Δ
- `μ`:  current radius multiplier (mutable)
"""
mutable struct R4RelativeGradUpdate <: AbstractRadiusUpdate
    γ₁::Float64
    γ₂::Float64
    μ::Float64
    function R4RelativeGradUpdate(γ₁::Float64 = 0.25, γ₂::Float64 = 2.0, μ::Float64 = 1.0)
        @assert 0 < γ₁ < 1  "R4: need 0 < γ₁ < 1"
        @assert γ₂ > 1      "R4: need γ₂ > 1"
        @assert μ > 0        "R4: need μ > 0"
        new(γ₁, γ₂, μ)
    end
end

# ------------------------------------------------------------
# Initial radius helper
# ------------------------------------------------------------

"""
    initial_radius(rule, Δ₀, g_norm) -> Float64

Returns the initial trust-region radius.
For R1–R3: returns `Δ₀` directly.
For R4: returns `rule.μ · g_norm` (radius scales with initial gradient).
"""
initial_radius(::AbstractRadiusUpdate, Δ₀::Float64, ::Float64) = Δ₀
initial_radius(rule::R4RelativeGradUpdate, ::Float64, g_norm::Float64) = rule.μ * g_norm

# ------------------------------------------------------------
# update_radius! dispatch
# ------------------------------------------------------------

"""
    update_radius!(rule, Δ, ρ, η₁, η₂, s_norm, g_norm_old, g_norm_new) -> Float64

Returns the updated trust-region radius.

Arguments:
- `rule`:       radius update rule (R1–R4)
- `Δ`:          current radius
- `ρ`:          ratio actual / predicted reduction
- `η₁, η₂`:    lower and upper acceptance thresholds
- `s_norm`:     ‖s_k‖, norm of the computed step
- `g_norm_old`: ‖g_k‖ before the accept/reject decision  (used by R3)
- `g_norm_new`: ‖g_{k+1}‖ after the accept/reject decision (used by R4)

For R4, the rule struct is mutated (μ is updated in-place).
"""
function update_radius!(rule::R1ClassicalUpdate,
                        Δ::Float64, ρ::Float64, η₁::Float64, η₂::Float64,
                        s_norm::Float64, g_norm_old::Float64, g_norm_new::Float64)
    if ρ >= η₂
        return Δ * rule.γ₃
    elseif ρ >= η₁
        return Δ * rule.γ₂
    else
        return Δ * rule.γ₁
    end
end

function update_radius!(rule::R2StepSizeUpdate,
                        Δ::Float64, ρ::Float64, η₁::Float64, η₂::Float64,
                        s_norm::Float64, g_norm_old::Float64, g_norm_new::Float64)
    if ρ < η₁
        return Δ * rule.γ₁
    elseif ρ < η₂
        return rule.γ₂ * s_norm
    else
        return rule.γ₃ * s_norm
    end
end

function update_radius!(rule::R3DFOLikeUpdate,
                        Δ::Float64, ρ::Float64, η₁::Float64, η₂::Float64,
                        s_norm::Float64, g_norm_old::Float64, g_norm_new::Float64)
    if ρ >= η₁
        if Δ <= rule.ζ * g_norm_old
            return Δ * rule.γ₃
        else
            return Δ * rule.γ₂
        end
    else
        return Δ * rule.γ₁
    end
end

function update_radius!(rule::R4RelativeGradUpdate,
                        Δ::Float64, ρ::Float64, η₁::Float64, η₂::Float64,
                        s_norm::Float64, g_norm_old::Float64, g_norm_new::Float64)
    if ρ >= η₂ && s_norm > 0.5 * Δ
        rule.μ *= rule.γ₂
    elseif ρ < η₁
        rule.μ *= rule.γ₁
    end
    return rule.μ * g_norm_new
end

# ------------------------------------------------------------
# Base.show for all canonical rules
# ------------------------------------------------------------

function Base.show(io::IO, r::R1ClassicalUpdate)
    println(io, "R1 Classical Update:")
    println(io, "  γ₁ (contraction):   ", r.γ₁)
    println(io, "  γ₂ (maintenance):   ", r.γ₂)
    println(io, "  γ₃ (expansion):     ", r.γ₃)
end

function Base.show(io::IO, r::R2StepSizeUpdate)
    println(io, "R2 Step-Size Update:")
    println(io, "  γ₁ (Δ contraction): ", r.γ₁)
    println(io, "  γ₂ (‖s‖ factor, acceptable):    ", r.γ₂)
    println(io, "  γ₃ (‖s‖ factor, very good):     ", r.γ₃)
end

function Base.show(io::IO, r::R3DFOLikeUpdate)
    println(io, "R3 DFO-Like Update:")
    println(io, "  γ₁ (contraction):   ", r.γ₁)
    println(io, "  γ₂ (no-expand):     ", r.γ₂)
    println(io, "  γ₃ (expansion):     ", r.γ₃)
    println(io, "  ζ  (threshold):     ", r.ζ)
end

function Base.show(io::IO, r::R4RelativeGradUpdate)
    println(io, "R4 Relative-Gradient Update:")
    println(io, "  γ₁ (μ contraction): ", r.γ₁)
    println(io, "  γ₂ (μ expansion):   ", r.γ₂)
    println(io, "  μ  (current):       ", r.μ)
end
