# =============================================================================
# src/Radius_updates/hei_canonical.jl
#
# AbstractRadiusUpdate implementations for the three Hei-family rules:
#
#   HeiUpdate        — Δ_{k+1} = R_exp(ρ) · ‖s_k‖
#   HeiGradUpdate    — Δ_{k+1} = R_exp(ρ) · ‖g_{k+1}‖
#   HeiFanYuanUpdate — μ_{k+1} = μ_k · R_exp(ρ),  Δ_{k+1} = μ_{k+1} · ‖g_{k+1}‖
#
# where R_exp is the piecewise-exponential multiplier from Hei et al.:
#
#   t < η  →  β + (1 − γ₁ − β) · exp(λ₁(t − η))
#   t ≥ η  →  1 + γ₂ + (M − (1 + γ₂)) · (1 − exp(−λ₂(t − η)))
#
# These complement the legacy HeiParameters / HeiGradParameters /
# HeiFanYuanParameters structs (which use the AbstractTrustRegionParameters
# interface) and are compatible with trust_region_solver.
# =============================================================================

# -----------------------------------------------------------------------------
# Shared helper
# -----------------------------------------------------------------------------

"""
    _r_exp(t, η, β, γ₁, γ₂, M, λ₁, λ₂) -> Float64

Piecewise-exponential multiplier from Hei et al.

- `t < η`: radius contracts — `β + (1 − γ₁ − β) · exp(λ₁(t − η))`
- `t ≥ η`: radius expands  — `1 + γ₂ + (M − (1 + γ₂)) · (1 − exp(−λ₂(t − η)))`
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

# -----------------------------------------------------------------------------
# HeiUpdate  —  Δ = R_exp(ρ) · ‖s‖
# -----------------------------------------------------------------------------

"""
    HeiUpdate(η, β, γ₁, γ₂, M, λ₁, λ₂)

Trust-region radius update from Hei (2003):

```
Δ_{k+1} = R_exp(ρ_k) · ‖s_k‖
```

# Arguments
- `η`:  transition threshold for ρ
- `β`:  lower bound of R_exp when ρ < η
- `γ₁`: dampening factor when ρ < η
- `γ₂`: expansion factor when ρ ≥ η
- `M`:  upper bound of R_exp when ρ ≥ η
- `λ₁`: exponential growth rate when ρ < η
- `λ₂`: exponential decay  rate when ρ ≥ η
"""
struct HeiUpdate <: AbstractRadiusUpdate
    η ::Float64
    β ::Float64
    γ₁::Float64
    γ₂::Float64
    M ::Float64
    λ₁::Float64
    λ₂::Float64
end

function update_radius!(rule::HeiUpdate,
        Δ::Float64, ρ::Float64, ::Float64, ::Float64,
        s_norm::Float64, ::Float64, ::Float64)
    return _r_exp(ρ, rule.η, rule.β, rule.γ₁, rule.γ₂, rule.M, rule.λ₁, rule.λ₂) * s_norm
end

initial_radius(::HeiUpdate, Δ₀::Float64, ::Float64) = Δ₀

# -----------------------------------------------------------------------------
# HeiGradUpdate  —  Δ = R_exp(ρ) · ‖g_{k+1}‖
# -----------------------------------------------------------------------------

"""
    HeiGradUpdate(η, β, γ₁, γ₂, M, λ₁, λ₂)

Gradient-based variant of the Hei update:

```
Δ_{k+1} = R_exp(ρ_k) · ‖g_{k+1}‖
```

Uses the post-update gradient norm (after accept/reject) rather than the step.
Arguments are the same as [`HeiUpdate`](@ref).
"""
struct HeiGradUpdate <: AbstractRadiusUpdate
    η ::Float64
    β ::Float64
    γ₁::Float64
    γ₂::Float64
    M ::Float64
    λ₁::Float64
    λ₂::Float64
end

function update_radius!(rule::HeiGradUpdate,
        Δ::Float64, ρ::Float64, ::Float64, ::Float64,
        ::Float64, ::Float64, g_norm_new::Float64)
    return _r_exp(ρ, rule.η, rule.β, rule.γ₁, rule.γ₂, rule.M, rule.λ₁, rule.λ₂) * g_norm_new
end

initial_radius(::HeiGradUpdate, Δ₀::Float64, ::Float64) = Δ₀

# -----------------------------------------------------------------------------
# HeiFanYuanUpdate  —  μ *= R_exp(ρ),  Δ = μ · ‖g_{k+1}‖
# -----------------------------------------------------------------------------

"""
    HeiFanYuanUpdate(μ, η, β, γ₁, γ₂, M, λ₁, λ₂)

Yuan–Fan–Hei update with adaptive multiplier μ:

```
μ_{k+1} = μ_k · R_exp(ρ_k)
Δ_{k+1} = μ_{k+1} · ‖g_{k+1}‖
```

`μ` is updated in place each iteration, so a fresh instance must be created
for every solver run (use a factory function in benchmark configurations).

# Arguments
- `μ`:  initial radius-to-gradient multiplier
- remaining arguments: same as [`HeiUpdate`](@ref)
"""
mutable struct HeiFanYuanUpdate <: AbstractRadiusUpdate
    μ ::Float64   # current multiplier (updated each iteration)
    μ₀::Float64   # initial value for reset
    η ::Float64
    β ::Float64
    γ₁::Float64
    γ₂::Float64
    M ::Float64
    λ₁::Float64
    λ₂::Float64
    function HeiFanYuanUpdate(μ::Float64, η::Float64, β::Float64,
                               γ₁::Float64, γ₂::Float64, M::Float64,
                               λ₁::Float64, λ₂::Float64)
        new(μ, μ, η, β, γ₁, γ₂, M, λ₁, λ₂)  # μ₀ = μ at construction
    end
end

function update_radius!(rule::HeiFanYuanUpdate,
        Δ::Float64, ρ::Float64, ::Float64, ::Float64,
        ::Float64, ::Float64, g_norm_new::Float64)
    rule.μ *= _r_exp(ρ, rule.η, rule.β, rule.γ₁, rule.γ₂, rule.M, rule.λ₁, rule.λ₂)
    return rule.μ * g_norm_new
end

initial_radius(rule::HeiFanYuanUpdate, ::Float64, g_norm::Float64) = rule.μ * g_norm
