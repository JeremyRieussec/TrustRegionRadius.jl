# Getting started

## Installing

```julia
using Pkg
Pkg.add(url = "https://github.com/USER/TrustRegionRadius.jl")
```

For development, clone and `Pkg.develop(path = ".")`.

## A first solve

```julia
using TrustRegionRadius, ADNLPModels

rosen(x) = (1 - x[1])^2 + 100(x[2] - x[1]^2)^2
nlp = ADNLPModel(rosen, [-1.2, 1.0])

stats = tr_solve(nlp; rule = RDelta())
stats.status      # :first_order
stats.solution    # ≈ [1.0, 1.0]
stats.iter
```

`tr_solve` returns a [`TRResult`](@ref), which is JSO's `GenericExecutionStats`, so
`SolverBenchmark` and the rest of the ecosystem accept it unchanged. Evaluation counts live
on the model — `neval_obj(nlp)`, `neval_grad(nlp)`, `neval_hprod(nlp)` — not on the result.

## Choosing the three axes

```julia
stats = tr_solve(nlp;
    rule      = RGrad(μ = 1.0),      # how Δ moves
    model     = SR1Model(mem = 5),   # what curvature the model reports
    subsolver = SteihaugCG(),        # how the subproblem is solved
    params    = TRParams(η₁ = 0.1, η₂ = 0.9, Δ₀ = 1.0, tol = 1e-8))
```

Defaults are `RDelta()`, `ExactHessian()`, `SteihaugCG()`.

!!! note "Hold `TRParams` fixed across mechanisms"
    In any comparison the thresholds and factors must be identical for every rule. Tuning
    them per rule measures tuning effort rather than algorithmic merit and makes a
    performance profile uninterpretable.

## Tracing

`trace = true` attaches per-iteration trajectories to `stats.solver_specific`:

```julia
stats = tr_solve(nlp; rule = RGrad(), trace = true)

ss = stats.solver_specific
ss[:delta_trajectory]      # Δₖ
ss[:grad_trajectory]       # ‖gₖ‖
ss[:obj_trajectory]        # f(xₖ)
ss[:ratio_trajectory]      # ρₖ
ss[:step_trajectory]       # ‖sₖ‖
ss[:active_trajectory]     # ‖sₖ‖ == Δₖ  (Bool)
```

The last one is the interesting one. Two mechanisms can have identical iteration counts and
identical first-order behaviour while one keeps the constraint permanently active and the
other does not; nothing else in a standard diagnostic distinguishes them.

```julia
tail = ss[:active_trajectory][end - min(end, 20) + 1 : end]
count(tail) / length(tail)     # fraction of late iterations still binding
```

## Statuses

| status | meaning |
|---|---|
| `:first_order` | `‖g‖ ≤ tol` — the only status counted as solved in a profile |
| `:max_iter` | iteration budget exhausted |
| `:max_time` | wall-clock budget exhausted |
| `:stalled` | the step fell below the level at which `f(x) − f(x+s)` carries information, so ρ became noise |
| `:exception` | the model could not be built at the current iterate |
| `:user` | stopped by a callback |

`:stalled` deserves a word. Once the radius collapses far enough, the subtraction
`f(xₖ) − f(xₖ + sₖ)` is pure rounding, ρ becomes noise, every step is rejected, and the
radius contracts forever. This is a genuine numerical limitation of the criticality-anchored
rules in floating point rather than a bug, and it is reported separately so that it is not
silently folded into "failure".

## Adding your own rule

One struct and two methods:

```julia
mutable struct MyRule <: RadiusRule
    up::Float64
    down::Float64
end

TrustRegionRadius.initial_radius(::MyRule, Δ₀, g_norm) = Δ₀

function TrustRegionRadius.update_radius!(r::MyRule, Δ, ρ, η₁, η₂,
                                          s_norm, g_norm_old, g_norm_new)
    return ρ ≥ η₁ ? Δ * r.up : Δ * r.down
end
```

Then `tr_solve(nlp; rule = MyRule(2.0, 0.5))`.

Add `reset_rule!` if the struct carries mutable state, and
`needs_retrospective(::MyRule) = true` if it should be driven by ρ̃ rather than ρ — the solver
then computes the retrospective ratio, at the cost of one extra Hessian-vector product per
accepted iteration.

Note the two gradient-norm arguments: `g_norm_old` is ‖gₖ‖ *before* the accept/reject
decision and `g_norm_new` is ‖gₖ₊₁‖ *after* it. Rules anchored to criticality at the current
iterate (`RDFO`) use the former; rules that set the next radius from the next gradient
(`RGrad` and relatives) use the latter. Getting this backwards is the easiest mistake to make
here, and it shifts every threshold silently.
