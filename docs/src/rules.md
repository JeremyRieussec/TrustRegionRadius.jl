# Radius mechanisms

Ten rules, all concrete subtypes of [`RadiusRule`](@ref).

## Choosing one

| rule | anchored to | `Δₖ → 0`? | needs a parameter above `κ̄`? |
|---|---|---|---|
| [`RDelta`](@ref) | nothing | no | no |
| [`RStep`](@ref) | `‖sₖ‖` | in the local regime | no |
| [`RDFO`](@ref) | `‖gₖ‖` via `ζ` | yes | **yes** (`ζ > κ̄`) |
| [`RGrad`](@ref) | `‖gₖ‖` via `μₖ`, uncapped | yes | **no** |
| [`RGradCapped`](@ref) | `‖gₖ‖` via `μₖ ≤ μ_max` | yes | **yes** (`μ_max > κ̄`) |
| [`RAdaptiveStep`](@ref) | `‖sₖ‖`, smooth factor | in the local regime | no |
| [`RAdaptiveGrad`](@ref) | `‖gₖ₊₁‖`, smooth factor | yes | — |
| [`RAdaptiveFanYuan`](@ref) | `‖gₖ₊₁‖` via `μₖ` | yes | — |
| [`RRTR`](@ref) | `‖sₖ‖`, driven by ρ̃ | no | no |
| [`RRTRGrad`](@ref) | `‖gₖ₊₁‖` via `μₖ`, driven by ρ̃ | yes | no |

`κ̄ = 4/λ*_min(∇²f(x*))` is a property of the *solution*, so a rule whose guarantee depends
on exceeding it cannot be configured reliably in advance. That is the single most useful fact
in the table.

## The families

**Criticality-blind** (`RDelta`). The radius depends on its own history and on ρ, never on
`‖gₖ‖`. Consequently `liminf Δₖ > 0`: the constraint eventually stops binding with no side
condition, but the radius carries no asymptotic information about criticality.

**Step-driven** (`RStep`, `RAdaptiveStep`, `RRTR`). The radius tracks the accepted step. Both
carry a `Δmin` floor, and it is not cosmetic — see the warning below.

**Criticality-anchored** (`RDFO`, `RGrad`, `RGradCapped`, `RAdaptiveGrad`,
`RAdaptiveFanYuan`, `RRTRGrad`). The radius is tied to `‖gₖ‖`, so `Δₖ → 0`. These are the
rules for which the `κ̄` threshold matters. Query with
[`is_criticality_anchored`](@ref).

**Retrospective** (`RRTR`, `RRTRGrad`). The radius update is judged by ρ̃, computed from the
*new* model at the *previous* iterate, rather than by ρ. Query with
[`needs_retrospective`](@ref).

!!! warning "`Δmin` on the step-driven rules"
    On an accepted step the new radius is proportional to `‖sₖ‖`, so a short step gives a
    small radius, which gives a shorter step still. Near convergence — or when a truncated-CG
    solver stops on its first iteration — this can collapse to zero, after which every step
    is zero, ρ is `NaN`, and the solver spins to `max_iterations`. `Δmin` (default `1e-14`)
    floors the radius and turns that silent hang into ordinary slow progress. Set
    `Δmin = 0.0` to reproduce the unguarded rule exactly.

## `RGrad` versus `RGradCapped`

`μₖ` *is* the radius-to-criticality ratio `Δₖ/‖gₖ‖`, so `RGrad` performs a geometric search
for the inactivity threshold `κ̄` without knowing it: `μ` climbs until the constraint stops
binding, then stops climbing. Because it is uncapped it crosses any threshold eventually,
which makes eventual inactivity **unconditional**.

`RGradCapped` supplies the bound `μₖ ≤ μ_max` that the global asymptotic theory assumes
(`Δₖ → 0` needs it). The cost is real: with a truncated-CG subsolver a small `μ_max` makes CG
truncate on its first iteration, so the returned step is the Cauchy point and the model
Hessian stops influencing the search direction at all. The method has silently become
gradient descent, and with a quasi-Newton model that can change which critical point is
reached. Diagnose with [`cg_step_info`](@ref).

The guard `‖sₖ‖ > ½Δₖ` in both rules is not cosmetic either: it is what converts "μ grows"
into "‖sₖ‖ is comparable to μₖ‖gₖ‖", which is the step that makes the boundedness argument
for μ work. `half_test = false` disables it and breaks that argument.

## API

```@docs
RadiusRule
RDelta
RStep
RDFO
RGrad
RGradCapped
RAdaptiveStep
RAdaptiveGrad
RAdaptiveFanYuan
RRTR
RRTRGrad
initial_radius
update_radius!
reset_rule!
needs_retrospective
is_criticality_anchored
retrospective_ratio
```
