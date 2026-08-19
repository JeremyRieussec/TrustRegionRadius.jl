# Subproblem solvers

All solve

```math
\min_{\|s\| \le \Delta} \; g^\top s + \tfrac12 s^\top H s
```

and return `active::Bool`, `true` when `‖s‖ = Δ`. That flag is worth propagating: the
fraction of iterations on which the constraint binds is the observable that separates radius
mechanisms whose first-order behaviour is identical.

| solver | needs | handles indefinite `H`? | size |
|---|---|---|---|
| [`SteihaugCG`](@ref) | `B * v` only | yes, stops on negative curvature | any |
| [`ExactMS`](@ref) | dense eigendecomposition | yes, including the hard case | `n ≤ nmax` |
| [`KrylovCG`](@ref) | `B * v` | no — assumes `H ≻ 0` | any |
| [`KrylovCGLanczos`](@ref) | `B * v` | yes | any |

## `SteihaugCG` and the Cauchy point

The first CG direction is `-g`. A step that leaves the trust region on CG iteration 1 is
therefore **exactly the Cauchy point**, and the model Hessian has had no influence on its
direction whatsoever.

This is not a corner case. When the radius is small — a low `μ_max` in `RGradCapped`, a low
`ζ` in `RDFO` — it happens at *every* iteration, and the method is gradient descent in
disguise. Nothing in ρ, `‖g‖` or the radius trace reveals it. [`cg_step_info`](@ref) does:

```@example subsolvers
using TrustRegionRadius, ADNLPModels, NLPModels

nlp = ADNLPModel(x -> (1 - x[1])^2 + 100(x[2] - x[1]^2)^2, [-1.2, 1.0])
x = [-1.2, 1.0]
g = grad(nlp, x)
info = cg_step_info(SteihaugCG(), ExactHessian(), nlp, x, g, 0.01)
(info.cg_iters, info.active, info.cos_cauchy)
```

A radius small enough to truncate CG on its first iteration returns the Cauchy point:
`cg_iters == 1`, the constraint active, and `cos(s, -g) == 1`, so the model Hessian
played no part in choosing the direction.

`cg_iters == 1 && active` together with `cos_cauchy ≈ 1` certify the degeneration.

## `ExactMS` and the hard case

Uses the Moré–Sorensen characterisation: `s*` is optimal iff there is `λ ≥ 0` with

```math
(B + \lambda I)s^* = -g, \qquad B + \lambda I \succeq 0, \qquad \lambda(\Delta - \|s^*\|) = 0 .
```

`‖s(λ)‖` decreases strictly in `λ`, so the root is bracketed and bisected.

The **hard case** — `g` orthogonal to the eigenspace of `λ_min(B)`, where no `λ` attains
`‖s(λ)‖ = Δ` — is handled explicitly by adding the right multiple of that eigenvector. This
matters with `ExactHessian` or `SR1Model`, where indefiniteness is exactly the information
one wants to exploit; it cannot arise with a positive definite model, which is why solvers
written for that case often omit it.

Requires an eigendecomposition, hence `n ≤ nmax` (default 200). Larger problems raise rather
than silently allocating an `n × n` array.

## Adding a solver

```julia
struct MySolver <: SubproblemSolver end

function TrustRegionRadius.solve_subproblem!(::MySolver, model::ModelHessian, nlp,
                                             x, g, Δ, s, Hs, ws::SubWorkspace;
                                             curv = nothing)
    B = hessian_op(model, nlp, x)
    # ... write the step into s, using ws.r, ws.d, ws.cand, ws.Hd as scratch ...
    return norm(s) ≥ (1 - 1e-8) * Δ     # active?
end
```

!!! note "Not executed"
    An extension template with an elided body; shown for reference, not run.

Take the curvature from `model`, not from `nlp`: that is what lets one solver serve every
model Hessian.

Three optional pieces:

- `ws` is a caller-owned [`SubWorkspace`](@ref), so nothing is allocated per iteration. An
  eight-argument convenience method allocates one for you, which is right at the REPL and
  wrong in the loop.
- `Hs` is an **output**. Declare `returns_hprod(::MySolver) = true` if you leave `B·s` in
  it — CG forms it incrementally anyway, and the solver then skips its own `model_hprod!`,
  saving one Hessian-vector product per iteration.
- `curv` carries `(λ_min, v_min)` when the solver has already computed it. Declare
  `needs_eigenvector(::MySolver) = true` to be given the eigenvector too, rather than
  recomputing an eigendecomposition the solver has just done.

## API

```@docs
SubproblemSolver
SubWorkspace
SteihaugCG
ExactMS
KrylovCG
KrylovCGLanczos
solve_subproblem!
cg_step_info
returns_hprod
needs_eigenvector
```
