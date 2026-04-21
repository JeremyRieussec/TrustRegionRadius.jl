# TrustRegionRadius.jl — JSO-Compliance Patch

This patch fixes the ten issues flagged in the previous review and
turns `TrustRegionRadius.jl` into a fully JSO-compliant package.

## What changed

### 1. `src/TrustRegionRadius.jl`
- Added `using Krylov` and `using SolverCore` at top level.
- Added `abstract type AbstractTRSubproblemSolver end` to the module's
  abstract hierarchy (it was previously only defined inside an
  included file, which made the symbol invisible to downstream code
  in the same module).
- Removed the unused `using Plots, LaTeXStrings, CUTEst, Test`
  dependencies from the module preamble. (Tests should pull these
  in themselves; the module shouldn't force Plots on every user.)
- Cleaned up the export list, grouped by category.

### 2. `src/Trust-region/TRRSolver.jl`
Major rewrite. Concrete fixes:

- **TRSolverParams is now parametric on `T`**. `η₁`, `η₂`, `Δ₀`,
  `tol` are stored in the NLPModel's native precision so the hot
  loop stays in `Float32` / `Float64` / `BigFloat` without implicit
  promotion.
- **TRRSolver is parametric on `{T, V, R, S}`** — element type,
  vector type, rule type, and subsolver type. Every dispatch in the
  hot loop is resolved statically at compile time.
- **Keyword-only constructor**: `TRRSolver(nlp; rule, subsolver,
  params)`. Matches JSOSolvers convention (`TrunkSolver(nlp; mem=5)`).
- **`subsolver` is actually used**. The hot loop now calls
  `solve_subproblem!(solver.subsolver, nlp, x, g, Δ, solver.p,
  solver.Hp)` instead of hard-coding `truncated_cg_steihaug`.
- **Zero-allocation hot loop**: `solver.p` and `solver.x_cand` are
  pre-allocated buffers filled in place via `@. x_cand = x + p` and
  by the subsolver. No more `x + p` creating a fresh vector each
  iteration.
- **`SolverCore.reset!(solver)` and `reset!(solver, nlp)` methods**
  so SolverBenchmark can reuse a solver across problems.
- **Proper status handling**: `:first_order`, `:max_iter`, `:user`
  (the last one lets callbacks stop the run).
- **stats.solution is updated every iteration** so callbacks can
  read the current iterate.

### 3. `src/Subproblem/subproblem_interface.jl`
- **New in-place API**:
  `solve_subproblem!(sub, nlp, x, g, Δ, p, Hbuf) -> on_boundary::Bool`
  fills the output buffer `p` instead of allocating and returning it.
- **`SteihaugTointCG` rewritten** to write into `p` in place and reuse
  `Hbuf` as scratch for Hessian-vector products.
- **`KrylovCG` and `KrylovCGLanczos` now use `Krylov.cg` /
  `Krylov.cg_lanczos`** correctly — they build `rhs = -g` and pass
  `radius = Δ`, and copy the Krylov result into `p` via `copyto!`
  (no more `V(p)` which was invalid for arbitrary vector types).
- **`_get_on_boundary` helper** that handles Krylov-version drift:
  checks for `stats.on_boundary` first, falls back to a norm
  comparison `‖p‖ ≈ Δ` otherwise.

### 4. `src/Radius_updates/radius_update_interface.jl`
- **Full interface contract documented** as the `AbstractRadiusUpdate`
  docstring: which methods a subtype must implement (`update_radius!`,
  `initial_radius`, optional `reset_rule!`), with argument semantics
  and a working skeleton example (`MyExperimentalUpdate`).
- `reset_rule!` methods for `R4RelativeGradUpdate` and
  `HeiFanYuanUpdate` retained as before.

### 5. `Project.toml`
- Added `Krylov`, `SolverCore` as direct dependencies.
- Added `ADNLPModels` + `Test` as `[extras]` for the test target.
- Added `CUTEst`, `JSOSolvers`, `SolverBenchmark` as extras for the
  benchmark target (not automatically installed).

### 6. `test/runtests.jl`
Eight test sets covering:

1. Constructor — correct field sizes, default and custom rule/subsolver
2. Return type is `GenericExecutionStats`; convergence on quadratic
3. Convergence on Rosenbrock for **all six radius update rules**
4. NLPModels counters (`neval_obj`, `neval_grad`, `neval_hprod`) are populated
5. `reset_rule!` and `SolverCore.reset!` restore mutable state
6. Callback is called every iteration and can stop via `stats.status = :user`
7. All three subsolvers (`SteihaugTointCG`, `KrylovCG`,
   `KrylovCGLanczos`) reach first-order
8. Float32 parametric support works end-to-end
9. In-place `solve!` and functional `trust_region_radius` wrapper agree

### 7. `benchmark/compare_jso.jl`
Runnable benchmark comparing TRRSolver (with R1, R2, R3, R4, Hei,
HeiFanYuan) against JSOSolvers.jl's `trunk`, `lbfgs`, `R2`, `fomo`
on 25 CUTEst problems. Produces a Dolan–Moré performance profile on
gradient-evaluation count and saves it to
`benchmark_gradient_profile.pdf`.

## How to apply the patch

1. Copy `src/TrustRegionRadius.jl` over the top-level module file.
2. Copy `src/Trust-region/TRRSolver.jl` over your existing version.
3. Copy `src/Subproblem/subproblem_interface.jl` over your existing version.
4. Copy `src/Radius_updates/radius_update_interface.jl` over your existing version.
5. Copy `Project.toml` over (edit the `uuid` field with your real UUID).
6. Copy `test/runtests.jl` into your `test/` directory.
7. Copy `benchmark/compare_jso.jl` into a new `benchmark/` directory.

Then:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

All tests should pass.  If the Krylov tests fail, check which version
of Krylov.jl is installed — `stats.on_boundary` was introduced in
recent versions; the fallback should cover older versions.

## Include-order reminder

`src/TrustRegionRadius.jl` must include the subproblem interface
*before* `TRRSolver.jl`, because the latter references
`AbstractTRSubproblemSolver` in its struct definition and
`SteihaugTointCG()` in its default constructor. The include order
inside `src/Subproblem/main.jl` should look like:

```julia
include("subproblem_interface.jl")
include("Truncated_CG.jl")   # optional: legacy standalone function
```

and inside `src/Trust-region/main.jl`:

```julia
include("trust_region_solver.jl")   # legacy TROutput-based solver
include("TRRSolver.jl")             # new JSO-compliant solver
```
