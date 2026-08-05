# Second-order variants

Every mechanism in this package controls the radius by comparing it against a
*criticality measure*. The first-order measure is `‖g_k‖`, which vanishes at every
critical point — minimisers, saddles and maxima alike. The second-order measure is

```math
\tau_k = \max\bigl\{\,\|g_k\|,\; -\lambda_{\min}(B_k)\,\bigr\},
```

which vanishes only where the model is genuinely second-order critical.

Replacing one by the other is the whole of the upgrade: **the update logic does not
change**. That is why [`SecondOrder`](@ref) is a wrapper rather than a new family of
rules — it forwards `update_radius!` to the rule it wraps and overrides only
[`criticality`](@ref).

```julia
tr_solve(nlp;
    rule      = RGradTau(μ = 1.0),        # Δ_k = μ_k τ_k, μ updated exactly as in RGrad
    model     = ExactHessian(),           # must be able to report negative curvature
    subsolver = EigenPoint(SteihaugCG()), # must be able to exploit it
    params    = TRParams(tol = 1e-8, tol_H = 1e-6))
```

## The three pieces, and why all three are needed

A second-order run needs a measure that stays positive at a saddle, a model that can
see the negative curvature, and a solver that can move along it. Any one missing and
the run silently degrades to a first-order one.

| piece | what it supplies | what happens without it |
|---|---|---|
| `SecondOrder` / `τ` | a positive radius near a saddle | the radius collapses; see below |
| `ExactHessian`, `SR1Model` | `λ_min < 0` when it exists | `τ ≡ ‖g‖`, an expensive no-op — the solver now warns |
| `EigenPoint`, `ExactMS` | a step along the negative direction | positive radius, nowhere to go |
| `tol_H > 0` | refusal to stop at a saddle | `:first_order` reported at a saddle |

## What τ repairs

At an exact saddle `‖g‖ = 0` while `λ_min < 0`, so a `‖g‖`-anchored rule reports the
radius it would use at a solution. The two anchored families then fail differently:

- **`RGrad(‖g‖)` halts outright.** `Δ_k = μ_k‖g_k‖ = 0`, so every subsequent step is
  zero, `ρ` is `NaN`, and no amount of negative curvature in the model can be used.
- **`RDFO(‖g‖)` contracts geometrically.** The test `Δ_k > ζ‖g_k‖ = 0` succeeds at
  every iteration, so the radius is multiplied by `γ₂ < 1` for ever and reaches the
  stagnation floor in `O(log(1/eps))` iterations.

Both are repaired by `τ`, which equals `-λ_min > 0` there and holds the radius at the
scale of the curvature actually available.

```julia
julia> λ, gn = -1.0, 0.0;                       # an exact saddle

julia> r = RGrad(μ = 1.0);
julia> initial_radius(r, 1.0, criticality(r, gn, λ))
0.0                                              # dead

julia> rτ = RGradTau(μ = 1.0);
julia> initial_radius(rτ, 1.0, criticality(rτ, gn, λ))
1.0                                              # alive
```

## Stopping

`tol_H` adds the second-order test to the stopping rule: the run stops when
`‖g‖ ≤ tol` **and** `λ_min(B) ≥ -tol_H`, and reports `:second_order`. It is disabled
by default (`tol_H = -1`), so nothing changes for a first-order run and no curvature
estimate is paid for.

Note what the status does and does not claim. `:second_order` is a statement about the
*model* Hessian, so it is a statement about `f` only insofar as the model reports the
curvature of `f` — exactly for `ExactHessian`, asymptotically for `SR1Model` under
coherence, and never for a model constrained positive definite.

!!! warning "τ ≡ ‖g‖ over a positive definite model"
    `LBFGSModel` enforces `B ≻ 0`, so `λ_min > 0` always, so `τ = ‖g‖` identically and
    the second-order variant is an expensive no-op that will nonetheless report
    `:second_order` at a saddle. The same holds for `ScaledIdentity` and `SPDTarget`.
    Pair `SecondOrder` with `ExactHessian` or `SR1Model`, and read
    `:lambda_min_trajectory` to confirm the model ever reported negative curvature at
    all. This is declared through [`reports_negative_curvature`](@ref) and the solver
    emits a warning once per run when the two are paired, so the no-op is not silent.

## Why the subsolver matters

`τ`-anchoring keeps the radius positive; it does not make the step go anywhere.

[`SteihaugCG`](@ref) stops at the first direction of negative curvature and runs to the
boundary along it. That is a real decrease, but the direction is whichever the CG
recurrence happened to reach, and the guaranteed decrease carries no `|λ_min|Δ²` term.
[`EigenPoint`](@ref) wraps it, and when `λ_min < 0` compares the inner step against the
eigenpoint `±Δ v_min`, which decreases the model by at least `½|λ_min|Δ²`, keeping
whichever is better. [`ExactMS`](@ref) solves the subproblem exactly, hard case
included, so it satisfies the condition unwrapped.

This is what makes the `O(max{ε_g^{-2}, ε_H^{-3}})` complexity bound attainable in
practice rather than only in principle.

## Curvature estimation

[`lambda_min_estimate`](@ref) uses a dense symmetric eigendecomposition for `n ≤ nmax`
(free when the subsolver is already `ExactMS`) and Lanczos with full reorthogonalisation
above it. [`curvature_estimate`](@ref) is the type-stable form the solver calls: it always
returns a `(λ, v)` pair and hands it to the subsolver through `curv`, so `EigenPoint` no
longer recomputes an eigendecomposition the solver has just done.

!!! note "The Lanczos value is an upper bound"
    A Ritz value satisfies `λ_Ritz ≥ λ_min`, so an under-resolved estimate *understates*
    the negative curvature. In `τ` that biases the measure down and makes the
    second-order test optimistic — it can report `:second_order` at a saddle whose
    negative direction Lanczos has not yet found. Raise `lanczos_k` when the status
    matters; the dense branch is exact.

## Trace

With `trace = true` and a curvature estimate in play, two further trajectories appear:

```julia
ss[:lambda_min_trajectory]   # λ_min(B_k)
ss[:tau_trajectory]          # τ_k, the measure the rule actually saw
```

Comparing `:tau_trajectory` against `:grad_trajectory` is the cheapest second-order
diagnostic available: wherever `τ > ‖g‖`, the run is somewhere the gradient alone calls
critical and the curvature does not.

## API

```@docs
SecondOrder
RGradTau
RGradCappedTau
RDFOTau
RAdaptiveGradTau
RRTRGradTau
criticality
needs_curvature
tau_criticality
lambda_min_estimate
curvature_estimate
EigenPoint
second_order_status
```
