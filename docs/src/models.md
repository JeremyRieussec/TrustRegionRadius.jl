# Model Hessians

The solver never forms a matrix: it needs only the operator action `H·v`, which is what
[`hessian_op`](@ref) returns. This keeps large problems feasible and lets quasi-Newton models
be supplied as `LinearOperators` without changing the solver loop.

| model | curvature | can be indefinite? |
|---|---|---|
| [`ExactHessian`](@ref) | `∇²f(xₖ)` | yes |
| [`LBFGSModel`](@ref) | limited-memory BFGS | no — enforced positive definite |
| [`SR1Model`](@ref) | symmetric rank one | **yes**, and that is the point |
| [`ScaledIdentity`](@ref) | `c·I`, i.e. none | no |
| [`SPDTarget`](@ref) | constructed, minimiser pinned | no — positive definite by design |
| [`BHHHModel`](@ref), [`BHHH2Model`](@ref) | outer product of scores | no — see [Outer-product Hessians](likelihood.md) |
| [`GaussNewtonModel`](@ref) | `JᵀJ/N` | no — same |

The last three are restricted to a [problem class](problem_classes.md):
`required_problem` declares the narrowest class they are defined for, and the solver
constructor rejects the rest.

## Which to use

`ExactHessian` for small problems and for any experiment where the model should not be a
confounder. Note it does **not** satisfy the secant equation, so the Fan–Pan–Song route to
`ρ̃ → 1` does not apply to it; the Bastin route, which needs asymptotic second-order coherence
instead, does.

`LBFGSModel` never reports negative curvature. On a problem whose true Hessian is indefinite
along the trajectory this is invisible to every first-order diagnostic: ρ stays healthy,
`‖g‖` decreases, and the limit can still fail second-order optimality.

`SR1Model` may be indefinite, which is what lets a subsolver that exploits negative curvature
escape a saddle a positive-definite model would converge to. Pair it with `SteihaugCG` (which
detects `dᵀHd ≤ 0`) or `ExactMS` (which handles the hard case).

`ScaledIdentity` is a diagnostic instrument, not a practical model: with it the step is
`-g/c` truncated to the region, so the method is exactly gradient descent and any difference
between radius rules is attributable to the rule alone.

## `SPDTarget`

A two-dimensional construction whose unconstrained model minimiser is pinned to `target`
while `H ≻ 0`. With `d = target − x`, `u = d/‖d‖`, `v = u^⊥`:

```math
H = a\,uu^\top + b\,(uv^\top + vu^\top) + c\,vv^\top,
\qquad a = -\frac{g^\top u}{\|d\|},\quad b = -\frac{g^\top v}{\|d\|},\quad c = \frac{b^2}{a} + \lambda_\perp .
```

This has minimiser `target` for *every* `c`, and is positive definite exactly when `a > 0`,
equivalently when

```math
\varphi(x) = g^\top(\text{target} - x) < 0 ,
```

that is, the target lies downhill. That condition is necessary for **any** positive definite
model with that minimiser, so a `DomainError` is not an artefact of the construction: no such
model exists at `x`. Query it directly with [`phi_target`](@ref).

Point a `SPDTarget` at a saddle to obtain a run in which every hypothesis of the first-order
theory holds, every ρ is successful, `‖g‖ → 0` — and the limit is a saddle. The failure is
the model, and no radius rule can repair it.

```@example models
using TrustRegionRadius, ADNLPModels

nlp = ADNLPModel(p -> p[1]^4 - p[1]^3 + (0.25 - p[1]/2)*p[2]^2 + p[2]^4/4, [-0.5, 0.6])
stats = tr_solve(nlp; rule = RDelta(), model = SPDTarget(target = [0.75, 0.0]))
stats.solution     # the saddle
```

## Adding a model

```julia
struct MyModel <: ModelHessian end

TrustRegionRadius.hessian_op(::MyModel, nlp, x) = # anything supporting B * v
TrustRegionRadius.dense_hessian(::MyModel, nlp, x) = # Matrix, for ExactMS and diagnostics
```

Add `reset_model!` and `update_model!` if the model carries state; both default to no-ops.
`update_model!` receives the secant pair `(sₖ, yₖ = gₖ₊₁ − gₖ)` on accepted steps only.

Two further traits are worth declaring:

```julia
TrustRegionRadius.reports_negative_curvature(::MyModel) = false  # if B ⪰ 0 by construction
TrustRegionRadius.required_problem(::MyModel) = LikelihoodProblem  # if it needs one
```

!!! note "Not executed"
    The two blocks above are extension templates with elided right-hand sides. They are
    shown for reference and are not run when the documentation is built.

The first makes the solver warn when the model meets `SecondOrder` or `tol_H`, where it
would be an expensive no-op reporting `:second_order` at a saddle. The second is checked
at solver construction. Both default to the permissive answer.

## API

```@docs
ModelHessian
ExactHessian
LBFGSModel
SR1Model
ScaledIdentity
SPDTarget
hessian_op
dense_hessian
model_hprod!
update_model!
reset_model!
phi_target
reports_negative_curvature
model_eltype
```
