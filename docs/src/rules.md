# Radius mechanisms

Ten rules, all concrete subtypes of [`RadiusRule`](@ref). Every one reads the scaling
thresholds `η₁` and `η₂` and none reads the acceptance threshold `η`; see
[Thresholds and factors](thresholds.md).

## Choosing one

| rule | anchored to | regime | needs a parameter above `κ̄`? |
|---|---|---|---|
| [`RDelta`](@ref) | nothing | `:bounded_below` | no |
| [`RStep`](@ref) | `‖sₖ‖` | `:step_summable` | no |
| [`RDFO`](@ref) | `‖gₖ‖` via `ζ` | `:vanishing` | **yes** (`ζ > κ̄`) |
| [`RGrad`](@ref) | `‖gₖ‖` via `μₖ`, uncapped | `:vanishing` | **no** |
| [`RGradCapped`](@ref) | `‖gₖ‖` via `μₖ ≤ μ_max` | `:vanishing` | **yes** (`μ_max > κ̄`) |
| [`RAdaptiveStep`](@ref) | `‖sₖ‖`, smooth factor | `:step_summable` | no |
| [`RAdaptiveGrad`](@ref) | `‖gₖ₊₁‖` via `μₖ`, accumulated | `:vanishing` | no |
| [`RRTR`](@ref) | `‖sₖ‖`, driven by ρ̃ | `:step_summable` | no |
| [`RRTRGrad`](@ref) | `‖gₖ₊₁‖` via `μₖ`, driven by ρ̃ | `:vanishing` | no |

`κ̄ = 4/λ*_min(∇²f(x*))` is a property of the *solution*, so a rule whose guarantee depends
on exceeding it cannot be configured reliably in advance. That is the single most useful fact
in the table.

`RAdaptiveGrad` accumulates its multiplier, and so climbs past any threshold as `RGrad`
does. 

## The three regimes

[`asymptotic_regime`](@ref) reports which of the three regimes of Part II §3 a rule belongs
to. The older two-valued [`is_criticality_anchored`](@ref) cannot express the middle case,
which is the one that separates `RStep` from `RDelta`:

| regime | meaning | rules |
|---|---|---|
| `:vanishing` | `Δₖ → 0` with `Σ Δₖ²/Mₖ < ∞`, from the rule alone | `RDFO`, `RGrad`, `RGradCapped`, `RAdaptiveGrad`, `RRTRGrad` |
| `:step_summable` | `Δₖ` inherits the summability of `Σ‖sₖ‖` in the local regime, so `Δₖ → 0` there and nothing follows globally | `RStep`, `RAdaptiveStep`, `RRTR` |
| `:bounded_below` | `liminf Δₖ > 0` | `RDelta` |

## The families

**Criticality-blind** (`RDelta`). The radius depends on its own history and on ρ, never on
`‖gₖ‖` or `‖sₖ‖`. Consequently `liminf Δₖ > 0`: the constraint eventually stops binding with
no side condition, but the radius carries no asymptotic information about criticality.

**Step-driven** (`RStep`, `RAdaptiveStep`, `RRTR`). The radius tracks the trial step. All
carry a `Δmin` floor, and it is not cosmetic — see the warning below. All require `η₁ > 0`,
which [`validate_thresholds`](@ref) enforces when the solver is built.

**Criticality-anchored** (`RDFO`, `RGrad`, `RGradCapped`, `RAdaptiveGrad`,  `RRTRGrad`). The radius is tied to `‖gₖ‖`, so `Δₖ → 0`. These are the
rules for which the `κ̄` threshold matters. Query with [`is_criticality_anchored`](@ref).

**Retrospective** (`RRTR`, `RRTRGrad`). The radius update is judged by ρ̃, computed from the
*new* model at the *previous* iterate, rather than by ρ. Query with
[`needs_retrospective`](@ref).

!!! warning "`Δmin` on the step-driven rules"
    On an accepted step the new radius is proportional to `‖sₖ‖`, so a short step gives a
    small radius, which gives a shorter step still. Near convergence — or when a truncated-CG
    solver stops on its first iteration — this can collapse to zero, after which every step
    is zero, ρ is `NaN`, and the solver spins to `max_iterations`. `Δmin` (default `1e-14`)
    floors the radius and turns that silent hang into ordinary slow progress. Set
    `Δmin = 0.0` to reproduce the unguarded rule exactly, which is the form the asymptotic
    claims of Part II are stated for.

## Retrospective rules and rejected steps

ρ̃ compares the *new* model against `f` at the previous iterate, so it is defined only once the
model has moved. On a rejected step there is no new model, and the solver passes `ρₖ` in the
`ρ̃` slot. Both retrospective rules therefore branch on `accepted` **before** comparing
anything with their own thresholds `η̃₁`, `η̃₂`.

`RRTR` falls back to the step-anchored contraction of Part I:

```math
\Delta_{k+1} =
\begin{cases}
\min\{\gamma_2\|s_k\|,\ \gamma_1\Delta_k\} & \rho_k < 0 ,\\
\gamma_2\|s_k\| & 0 \le \rho_k < \eta ,
\end{cases}
```

and `RRTRGrad` contracts its multiplier, `μ ← γ₁μ`. The `γ₁` slot of `RRTR` plays the role of
the safeguard factor `max[γ₁, θₖ]` of Part I, held at a constant here.

!!! danger "Why the branch order matters"
    Comparing a classical `ρ` against the retrospective thresholds can return
    `Δₖ₊₁ = Δₖ` after a rejection. Since a rejected step changes neither the model nor the
    iterate, the next iteration then re-solves an identical subproblem, gets an identical
    rejection, and the run continues to `max_iterations` making no progress. It also violates
    the contraction condition, so the convergence theorem of Part I does not cover it. A rule
    added outside the package must contract on every unsuccessful iteration;
    `test/test_thresholds.jl` checks this for all nine rules at five values of ρ below `η`,
including `-Inf` — which is what the solver passes whenever the predicted reduction is
non-positive.

On an accepted step the first branch is the interesting one, `max(γ₃‖sₖ‖, Δₖ)`:
simultaneously step-driven *and* non-decreasing, which is what lets `RRTR` secure eventual
inactivity from a *consequence* of the standing assumptions (`ρ̃ → 1`) rather than from a
condition on a user parameter, as `RDFO` needs. The price is structural: `ρ̃ → 1` requires
either the secant condition on the model, or asymptotic second-order coherence together with
a quadratic model decrease.

## `RGrad` versus `RGradCapped`

`μₖ` *is* the radius-to-criticality ratio `Δₖ/‖gₖ‖`, so `RGrad` performs a geometric search
for the inactivity threshold `κ̄` without knowing it: `μ` climbs until the constraint stops
binding, then stops climbing. Because it is uncapped it crosses any threshold eventually,
which makes eventual inactivity **unconditional** — the only rule in the survey for which this
holds without a hypothesis on an unknowable constant.

Both take the four-branch form of Part I:

```math
\mu_{k+1} =
\begin{cases}
\gamma_1\mu_k & \rho_k < \eta_1 ,\\
\gamma_2\mu_k & \eta_1 \le \rho_k < \eta_2 ,\\
\gamma_3\mu_k & \rho_k \ge \eta_2 \ \text{and}\ \|s_k\| > \tfrac12\Delta_k ,\\
\mu_k & \text{otherwise} .
\end{cases}
```

The second branch contracts `μ` on mildly successful iterations, so the climb is not monotone.

The fourth branch is the one with a keyword. `half_test = true`, the default, grows `μ`
only when `‖s_k‖ > ½Δ_k`; `half_test = false` drops that guard and grows it on every
very successful iteration. The guard is what turns "`μ` grows" into "`‖s_k‖` is
comparable to `μ_k‖g_k‖`", which is the step the boundedness argument for `μ` runs on,
so the default keeps it. Any constant in `(0,1)` would serve in place of `½`.

`RGradCapped` supplies the bound `μₖ ≤ μ_max` that the global asymptotic theory assumes
(`Δₖ → 0` needs it). The cost is real: with a truncated-CG subsolver a small `μ_max` makes CG
truncate on its first iteration, so the returned step is the Cauchy point and the model
Hessian stops influencing the search direction at all. The method has silently become
gradient descent, and with a quasi-Newton model that can change which critical point is
reached. Diagnose with [`cg_step_info`](@ref).

The guard `‖sₖ‖ > ½Δₖ` is not cosmetic either: it is what converts "μ grows" into "‖sₖ‖ is
comparable to μₖ‖gₖ‖", which is the step that makes the boundedness argument for μ work. Any
constant in `(0,1)` serves; `half_test = false` disables the guard and breaks the argument.

## The Hei family

`RAdaptiveStep` and `RAdaptiveGrad` scale by a smooth factor
`R_{η₁}(ρₖ)` instead of one of three constants. The factor is parameterised by the same `γ₁, γ₂, γ₃` as
every other rule, plus two shape rates `λ₁, λ₂ > 0`:

```math
R_{\eta_1}(t) =
\begin{cases}
\gamma_1 + (\gamma_2 - \gamma_1)\,e^{\lambda_1(t-\eta_1)} & t < \eta_1 ,\\
(1+\gamma_2) + \bigl(\gamma_3 - 1 - \gamma_2\bigr)\bigl(1 - e^{-\lambda_2(t-\eta_1)}\bigr) & t \ge \eta_1 ,
\end{cases}
```

so that `R(−∞) = γ₁`, `R(t) < γ₂` below the threshold, `R(η₁) = 1 + γ₂`, and `R(+∞) = γ₃`.
Hence the extra requirement `γ₃ > 1 + γ₂`. Only the limits and the value at `η₁` matter for
admissibility, so `λ₁` and `λ₂` are free.

The jump at `η₁`, from `γ₂ < 1` to `1 + γ₂ > 1`, is required by those conditions rather than
an artefact: no continuous function satisfies both `R ≤ γ₂` below the threshold and
`R(η₁) = 1 + γ₂` at it.

The kink sits at the solver's `η₁` rather than at a threshold stored on the rule, so the
family is directly comparable with the three-case rules under one `(η, η₁, η₂)`.

## The catalogue, executed

The nine first-order rules, with the two traits that decide where each one belongs in
the theory. `is_criticality_anchored` says whether the radius is tied to a criticality
measure; `asymptotic_regime` says what the radius does in the limit; and
`needs_retrospective` marks the two rules that score an iteration with the *new* model
rather than the old one.

```@example rules
using TrustRegionRadius, ADNLPModels, Printf

catalogue = ["RDelta"        => RDelta(),
             "RStep"         => RStep(),
             "RDFO"          => RDFO(),
             "RGrad"         => RGrad(),
             "RGradCapped"   => RGradCapped(),
             "RAdaptiveStep" => RAdaptiveStep(),
             "RAdaptiveGrad" => RAdaptiveGrad(),
             "RRTR"          => RRTR(),
             "RRTRGrad"      => RRTRGrad()]

for (name, r) in catalogue
    @printf("%-15s anchored=%-6s regime=%-15s retrospective=%s
",
            name, is_criticality_anchored(r), asymptotic_regime(r), needs_retrospective(r))
end
```

The three regimes partition the nine: `:bounded_below` keeps the radius away from zero,
`:step_summable` makes `Σ Δ_k` finite in the local phase, and `:vanishing` drives
`Δ_k → 0`. Every anchored rule is `:vanishing`, and that is not a coincidence: anchoring
the radius to a criticality measure that tends to zero is what makes it tend to zero.

## One problem, four mechanisms

The same Rosenbrock solve under four rules, with the radius it started and finished on.

```@example rules
nlp = ADNLPModel(x -> (1 - x[1])^2 + 100(x[2] - x[1]^2)^2, [-1.2, 1.0])

for (name, mk) in ["RDelta" => () -> RDelta(), "RStep" => () -> RStep(),
                   "RGrad"  => () -> RGrad(),  "RDFO"  => () -> RDFO()]
    st = tr_solve(nlp; rule = mk(), params = TRParams(tol = 1e-8), trace = true)
    Δ  = st.solver_specific[:delta_trajectory]
    @printf("%-8s status=%-12s iter=%3d  Δ_0=%9.3g  Δ_end=%9.3g
",
            name, st.status, st.iter, Δ[1], Δ[end])
end
```

Read the last column against the regimes above. `RDelta` ends on a radius larger than it
started with, `RStep` on one nine orders of magnitude smaller, and `RGrad` on zero, which
is `Δ_k = μ_k‖g_k‖` doing exactly what it says once `‖g_k‖` reaches machine precision.

`RGrad` also ignores `Δ0` entirely: its `Δ_0` above is `233`, not the `1` the other three
started from, because [`initial_radius`](@ref) for a `μ`-scaled rule returns `μ₀‖g_0‖`.

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
RAdaptiveGradCapped
RDeltaStep
RRTR
RRTRGrad
initial_radius
update_radius!
reset_rule!
needs_retrospective
is_criticality_anchored
asymptotic_regime
retrospective_ratio
last_branch
```
