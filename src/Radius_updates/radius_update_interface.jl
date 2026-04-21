
# ============================================================
# radius_update_interface.jl
#
# Interface contract for AbstractRadiusUpdate and the reset_rule!
# mechanism for mutable rules.
# ============================================================

"""
    AbstractRadiusUpdate

Abstract supertype for trust-region radius update rules.

## Interface contract

A concrete subtype `MyRule <: AbstractRadiusUpdate` must implement:

### 1. `initial_radius`

    initial_radius(rule::MyRule, Δ₀::T, g_norm::T) -> T

Returns the trust-region radius at iteration 0.  The default for most
rules is to return `Δ₀` unchanged; rules that couple Δ to the gradient
norm (R4, HeiFanYuan) override this.

### 2. `update_radius!`

    update_radius!(rule::MyRule,
                   Δ::T, ρ::T, η₁::T, η₂::T,
                   s_norm::T, g_norm_old::T, g_norm_new::T) -> T

Returns the radius for the next iteration.  Allowed to mutate `rule`
in place (e.g. updating a multiplier `μ`).

Arguments:
- `Δ`:          current radius
- `ρ`:          ratio of actual to predicted reduction
- `η₁, η₂`:    thresholds `0 ≤ η₁ < η₂ < 1`
- `s_norm`:     ‖step‖
- `g_norm_old`: ‖grad‖ *before* the accept/reject decision
- `g_norm_new`: ‖grad‖ *after*  the accept/reject decision

Rules that only need a subset of these may ignore the rest.

### 3. `reset_rule!` (mutable rules only)

    reset_rule!(rule::MyRule) -> nothing

Restore any mutable state (e.g. `rule.μ ← rule.μ₀`).  The default
implementation is a no-op, so immutable rules need do nothing.

## Example

```julia
struct MyExperimentalUpdate <: AbstractRadiusUpdate
    factor_up::Float64
    factor_dn::Float64
end

function update_radius!(rule::MyExperimentalUpdate,
                        Δ, ρ, η₁, η₂, s_norm, g_old, g_new)
    ρ >= η₁ ? Δ * rule.factor_up : Δ * rule.factor_dn
end

initial_radius(::MyExperimentalUpdate, Δ₀, g_norm) = Δ₀
# reset_rule! not needed -- the struct is immutable
```

After that, the rule is plug-and-play:

```julia
trust_region_radius(nlp; rule = MyExperimentalUpdate(2.0, 0.5))
```
"""
AbstractRadiusUpdate   # just to attach the docstring to the type

# ------------------------------------------------------------
# reset_rule! -- default and overrides
# ------------------------------------------------------------

"""
    reset_rule!(rule::AbstractRadiusUpdate)

Reset any mutable state in `rule` to its construction value.

The default implementation is a no-op.  Mutable rules
(`R4RelativeGradUpdate`, `HeiFanYuanUpdate`) override this to
restore `μ ← μ₀`.  Called automatically at the start of every
`solve!` and via `SolverCore.reset!`.
"""
reset_rule!(::AbstractRadiusUpdate) = nothing

function reset_rule!(rule::R4RelativeGradUpdate)
    rule.μ = rule.μ₀
    return nothing
end

function reset_rule!(rule::HeiFanYuanUpdate)
    rule.μ = rule.μ₀
    return nothing
end
