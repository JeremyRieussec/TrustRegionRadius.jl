# Thresholds and factors

Two conventions govern every mechanism in the package. Both are enforced at construction, so
a configuration that violates either fails immediately rather than producing a run that looks
plausible and is outside the theory.

## Acceptance is not scaling

`TRParams` carries three thresholds satisfying

```math
0 \le \eta \le \eta_1 \le \eta_2 < 1 ,
```

and they answer different questions:

| threshold | question | read by |
|---|---|---|
| `η` | is this step worth taking? | the solver alone |
| `η₁` | was the model poor enough that the region should shrink? | the rule alone |
| `η₂` | was it good enough that the region should grow? | the rule alone |

`η` defines the successful and unsuccessful index sets

```math
\mathcal{S} = \{k : \rho_k \ge \eta\}, \qquad \mathcal{U} = \{k : \rho_k < \eta\} ,
```

and nothing else in the package compares `ρ` with it. `η₁` and `η₂` never decide whether a
step is taken.

```julia
TRParams(η = 0.0, η₁ = 0.1, η₂ = 0.9)     # decoupled
TRParams(η₁ = 0.1, η₂ = 0.9)              # η defaults to η₁: the classical algorithm
```

Because `η` defaults to `η₁`, existing code is unaffected until it asks for something else.

### What decoupling buys

Setting `η < η₁` opens a band in which `ρ_k ∈ [η, η₁)`: the step is **accepted and the radius
still contracts**. The model was good enough to make progress worth keeping and bad enough
that the region should not be trusted at its current size. Nothing in the framework forbids
this, and no rule needed changing to support it — the rules already branch on `η₁`.

Setting `η = 0` accepts every step with a positive predicted reduction. This is admissible in
Part I and excluded by the framework of Curtis & Scheinberg (2020), which requires a strictly
positive acceptance threshold; it is worth a column of its own in any comparison.

!!! note "Count the band, not the shrinking radius"
    To measure how often the new regime is entered, test `ρ_k < η₁` on accepted iterations —
    not whether `Δ` decreased. `RDelta` contracts by `γ₂` on *every* mildly successful
    iteration, so a shrinking radius does not isolate the band:

    ```julia
    ss  = stats.solver_specific
    ρ, acc = ss[:ratio_trajectory], ss[:accepted_trajectory]
    count(i -> acc[i] && ρ[i] < params.η₁, eachindex(acc))
    ```

    The count is zero by construction whenever `η = η₁`.

### Why the rules are told the outcome

```julia
update_radius!(rule, Δ, ρ, accepted, η₁, η₂, s_norm, g_norm_old, g_norm_new)
```

`accepted` is passed rather than recomputed. A rule cannot derive it: acceptance is decided by
`η`, which the rule never receives, and the retrospective rules are handed `ρ̃` in the `ρ` slot,
so they do not see `ρ_k` at all. For [`RRTR`](@ref) and [`RRTRGrad`](@ref) this argument is
load-bearing — see [Retrospective rules and rejected steps](rules.md#Retrospective-rules-and-rejected-steps).

### Rules that constrain the thresholds

Some rules cannot accept every admissible triple. The step-driven rules need `η₁ > 0`,
because at `η₁ = 0` their aggressive branch is unreachable and the lower-bound constant of
Part I degenerates. [`validate_thresholds`](@ref) is called once when the solver is built, so
the failure is a constructor error rather than a silent change of behaviour:

```julia
TRSolver(nlp; rule = RStep(), params = TRParams(η = 0.0, η₁ = 0.0, η₂ = 0.9))
# ArgumentError: RStep: needs η₁ > 0 …
```

Rules with no such requirement accept `η₁ = 0` unchanged. To add a requirement for a new
rule, define a method of `validate_thresholds`; the fallback accepts everything.

## One convention for the scaling factors

Every rule carries `γ₁`, `γ₂`, `γ₃` with

```math
0 < \gamma_1 \le \gamma_2 < 1 < \gamma_3 ,
```

`γ₁` the aggressive contraction, `γ₂` the mild one, `γ₃` the expansion. This is the
convention of Parts I–II, checked in every constructor by [`check_factors`](@ref), which
throws an `ArgumentError` naming the rule and the offending inequality. A rule that does not
use one of the three passes `nothing` for it; the remaining inequalities are still checked.

`ArgumentError` rather than `@assert` deliberately: `@assert` is documented as liable to be
disabled, and these are argument checks rather than internal invariants.

```julia
RGrad(γ₂ = 2.0)                      # ArgumentError: γ₂ is the mild contraction
RDelta(γ₁ = 0.8, γ₂ = 0.3)           # ArgumentError: need γ₁ ≤ γ₂
RDelta(γ₃ = 0.9)                     # ArgumentError: need γ₃ > 1
RGrad(γ₁ = 0.25, γ₂ = 0.5, γ₃ = 2.0) # fine
```

The Hei family carries one extra requirement, `γ₃ > 1 + γ₂`, so that the smooth factor can
exceed the value it takes at the threshold; see
[The Hei family](rules.md#The-Hei-family).

### Rules whose factors were renumbered

Four rules previously used `γ₂` for expansion, which is now the mild contraction slot:

| rule | before | now |
|---|---|---|
| [`RGrad`](@ref) | `γ₁ = 0.25`, `γ₂ = 2.0` | `γ₁ = 0.25`, `γ₂ = 0.5`, `γ₃ = 2.0` |
| [`RGradCapped`](@ref) | `γ₁ = 0.25`, `γ₂ = 2.0` | `γ₁ = 0.25`, `γ₂ = 0.5`, `γ₃ = 2.0` |
| [`RRTR`](@ref) | `γ₀ = 0.0625`, `γ₁ = 0.25`, `γ₂ = 2.5` | `γ₁ = 0.0625`, `γ₂ = 0.25`, `γ₃ = 2.5` |
| [`RRTRGrad`](@ref) | `γ₁ = 0.25`, `γ₂ = 2.0` | `γ₁ = 0.25`, `γ₃ = 2.0` |

`RDelta`, `RStep` and `RDFO` keep their values; only their checks are tighter, since `γ₂ < 1`
and `γ₁ ≤ γ₂` were previously unverified. Old keyword calls now throw, so none of these can
pass silently. `MIGRATION.md` in the repository root lists the one case that can — a
four-argument positional `RGrad(...)` written against the old signature.

## API

```@docs
validate_thresholds
check_factors
```
