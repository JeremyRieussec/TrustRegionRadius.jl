# Integration

Status of the previous checklist, then what changed in this pass.

**Nothing here has been executed.** No Julia toolchain was available, so the
code is structurally checked — balanced blocks, every export resolving, no
duplicate names — but not compiled or run. The first `using TrustRegionRadius`
is the real test.

---

## 1. Previous checklist — status

| Item | Status |
|---|---|
| `AbstractTRSubproblemSolver` declared twice | **fixed** — one `SubproblemSolver`, declared once in `Subproblem/subproblem.jl` |
| `radius_update_interface.jl` load order | **dissolved** — `reset_rule!` methods now sit beside the types they reset, so no ordering constraint remains |
| `R2StepSizeUpdate` can drive Δ → 0 | **fixed** — `RStep` and `RAdaptiveStep` carry `Δmin` (default `1e-14`); set `Δmin = 0.0` to reproduce the unguarded rule |
| `neval_hess` counted instead of `neval_hprod` | **fixed** — the harness records `neval_hprod` |
| `LinearOperators`, `Printf` in `Project.toml` | **still to do by you** (see §5) |
| `π` shadowing `Base.π` | **fixed** — `performance_profile` returns `(τ, prof)` |

---

## 2. Rules renamed, aliases removed

Every rule is now a real struct under its primary name. No `const X = Y`
aliases anywhere.

| Old | New | Note |
|---|---|---|
| `R1ClassicalUpdate` | `RDelta` | |
| `R2StepSizeUpdate` | `RStep` | gains `Δmin` |
| `R3DFOLikeUpdate` | `RDFO` | |
| `R4RelativeGradUpdate` | `RGrad` | now the **uncapped** rule |
| — | `RGradCapped` | new: `μ ≤ μ_max` |
| `HeiUpdate` | `RAdaptiveStep` | gains `Δmin` |
| `HeiGradUpdate` | `RAdaptiveGrad` | |
| `HeiFanYuanUpdate` | `RAdaptiveFanYuan` | |
| — | `RRTR` | new: retrospective (Bastin et al. 2010) |
| — | `RRTRGrad` | new: retrospective gradient-scaled (Fan–Pan–Song 2016) |
| `AbstractRadiusUpdate` | `RadiusRule` | |

Keyword constructors throughout, with positional forms retained where your
`config.jl` used them, so `RDelta(0.25, 0.5, 2.0)` still works.

### `RGrad` is now uncapped by default

This is a behaviour change worth knowing about. The old `R4RelativeGradUpdate`
had no cap but was documented as if bounded; the two variants are now
separate types, because the distinction is not cosmetic:

- `RGrad` — μ grows geometrically past any threshold, so eventual inactivity of
  the trust region is **unconditional**. The only rule in the survey for which
  this holds without a hypothesis on `κ̄ = 4/λ*_min`, a constant that depends on
  the solution and cannot be checked in advance.
- `RGradCapped` — supplies the bound `μ_k ≤ μ_max` that the asymptotic results
  requiring `Δ_k → 0` assume. The cost is that inactivity now needs
  `μ_max > κ̄`, and below that the constraint binds for ever.

To reproduce the old default exactly: `RGrad(γ₁ = 0.25, γ₂ = 2.0, μ = 1.0)`.

### Two new predicates

```julia
needs_retrospective(rule)       # rule wants ρ̃, not ρ
is_criticality_anchored(rule)   # Δ_k → 0, so κ̄ matters for this rule
```

The solver computes ρ̃ only when the first returns `true`, so rules that do not
use it pay nothing. The second is informational, and is what the experiment
scripts use to group results by family.

---

## 3. Export redundancies resolved

You flagged four overlapping groups. All are gone; each name is now exported
exactly once (47 exports, no duplicates).

**Subproblem solvers.** Two abstract types collapsed into one:

```julia
# before: two supertypes, two spellings of each solver
export SubSolver, SteihaugCG, ExactMS
export AbstractTRSubproblemSolver, solve_subproblem!
export SteihaugTointCG, KrylovCG, KrylovCGLanczos

# after
export SubproblemSolver
export SteihaugCG, ExactMS, KrylovCG, KrylovCGLanczos
export solve_subproblem!, cg_step_info
```

`solve_subproblem!` also gained the model as an argument, so the same solvers
serve the exact Hessian and every quasi-Newton model:

```julia
solve_subproblem!(sub, model, nlp, x, g, Δ, s, Hbuf) -> active::Bool
```

**Solver entry points.** `TRSolverParams`/`TROutput` and `TRParams`/`TRResult`
were the same things twice. One of each survives:

```julia
export TRParams, TRResult, TRSolver, tr_solve
```

`TRResult` is `GenericExecutionStats`, so the JSO ecosystem accepts it
directly. `TRParams` gains `Δmax` and `max_time`.

**`AlgorithmInfoTR` / `AlgorithmInfoGD`** are gone. Their role — recording
per-iteration history — is now `trace = true`, which attaches trajectories to
`stats.solver_specific`:

```julia
stats = tr_solve(nlp; rule = RGrad(), trace = true)
stats.solver_specific[:delta_trajectory]
stats.solver_specific[:active_trajectory]   # ‖s_k‖ == Δ_k per iteration
```

That last one is worth having by default. Whether the constraint eventually
stops binding is what separates mechanisms whose first-order behaviour is
identical, and no standard diagnostic records it.

---

## 4. Files

```
TrustRegionRadius/
├── Project.toml                      package deps  (see §5 -- set the UUID)
├── README.md
├── INTEGRATION.md                    this file
├── .gitignore
│
├── src/
│   ├── TrustRegionRadius.jl          includes + 47 exports
│   ├── Radius_updates/
│   │   ├── main.jl
│   │   ├── rules.jl                  NEW  all ten rules, real names
│   │   └── retrospective.jl          NEW  the ratio rho-tilde
│   ├── Model_Hessians/
│   │   ├── main.jl
│   │   └── model_hessian.jl          ModelHessian axis + model_hprod!
│   ├── Subproblem/
│   │   ├── main.jl
│   │   └── subproblem.jl             NEW  one supertype, all four solvers
│   ├── Trust-region/
│   │   ├── main.jl
│   │   └── solver.jl                 NEW  TRParams, TRSolver, tr_solve
│   └── Benchmark/
│       ├── main.jl
│       ├── profiles.jl               performance/data profiles, pgfplots
│       └── run_matrix.jl             TRConfig, run_matrix, summarise
│
├── test/
│   ├── runtests.jl
│   ├── test_rules.jl                 the update contract and its invariants
│   ├── test_models.jl                operator contract, SPDTarget existence
│   ├── test_subproblem.jl            optimality, boundary, hard case
│   ├── test_solver.jl                convergence across all three axes
│   └── test_profiles.jl              Dolan-More and More-Wild definitions
│
├── benchmark/
│   ├── Project.toml                  separate env: Plots, JLD2, CUTEst
│   ├── config.jl                     rules, params, problem selection
│   ├── archive.jl                    NEW  ExperimentArchive
│   ├── harness.jl                    NEW  problems, runner, metric matrices
│   ├── experiments/
│   │   ├── exp1_comparison.jl
│   │   ├── exp2_trajectories.jl
│   │   ├── exp3_zeta_sweep.jl
│   │   ├── exp4_mu_sweep.jl
│   │   ├── exp5_inactivity.jl
│   │   ├── exp6_interaction.jl
│   │   ├── exp7_convergence_rate.jl
│   │   └── run_all.jl
│   └── results/                      archives land here (git-ignored)
│       └── exp_YYYY-MM-DD_HH-MM-SS_tag/
│           ├── experiment_config.toml
│           ├── experiment_summary.md
│           ├── figures/
│           ├── tables/
│           └── data/
│
└── docs/
    ├── Project.toml
    ├── make.jl
    └── src/index.md
```

**Deleted:** `canonical_R1R2R3R4.jl`, `hei_canonical.jl`,
`radius_update_interface.jl`, `rule_aliases.jl`, `Truncated_CG.jl`,
`subproblem_interface.jl`, `exact_ms.jl`, `TRRSolver.jl`,
`trust_region_solver.jl`, `model_hessian_solver.jl`.

Your `load_results.jl` and `config_utils.jl` are superseded by `harness.jl` and
`archive.jl` respectively. Nothing imports them, so they can stay until you are
satisfied the replacements work.

## 5. `Project.toml`

Both `Project.toml` files are written. Two things to do before first use:

1. **Set the package UUID.** `Project.toml` carries a placeholder. Generate a
   real one and paste it into both `Project.toml` and
   `benchmark/Project.toml` (the `TrustRegionRadius` entry there must match):

   ```julia
   using UUIDs; uuid4()
   ```

2. **Instantiate the benchmark environment**, which is deliberately separate so
   that Plots, JLD2 and CUTEst are not dependencies of the package itself:

   ```bash
   julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
   ```

`Printf`, `TOML`, `Dates` and `LinearAlgebra` are standard libraries and need no
`[compat]` bound.

---

## 6. Experiment layout

Each run creates one self-documenting directory:

```
benchmark/results/
  exp_2026-04-16_02-08-34_zeta_sweep/
    experiment_config.toml      what was run
    experiment_summary.md       what came out
    figures/                    every PDF
    tables/                     every text table, and .tex profile coordinates
    data/                       raw JLD2, one file per (problem, rule)
```

The timestamp is in the directory name, so runs never collide and sort by date.
The optional tag distinguishes concurrent runs.

```julia
arch = ExperimentArchive(tag = "zeta_sweep")
save_config(arch; rules = RULES, params = SOLVER_PARAMS,
            problem_selection = PROBLEM_SELECTION)
savefig_archived(arch, "exp3_perf_profile_zeta.pdf", plt)
save_table(arch, "exp3_zeta_summary.txt", txt)
finalize_archive(arch; notes = "…")     # writes experiment_summary.md
```

`load_config(dir)` reads a past run back, and `latest_archive()` returns the
most recent one.

### The seven experiments

| File | Tests |
|---|---|
| `exp1_comparison.jl` | like-for-like comparison; performance + data profiles |
| `exp2_trajectories.jl` | Δ, ‖g‖, ρ per iteration; ΣΔ and ΣΔ² by family |
| `exp3_zeta_sweep.jl` | ζ threshold; reliability vs efficiency |
| `exp4_mu_sweep.jl` | μ_max threshold; Cauchy-point diagnostic; uncapped comparison |
| `exp5_inactivity.jl` | fraction of late iterations with ‖s‖ = Δ |
| `exp6_interaction.jl` | rule × model grid; additivity residuals |
| `exp7_convergence_rate.jl` | local order, conditioned on inactivity |

Run one, or all:

```bash
julia --project=benchmark benchmark/experiments/exp3_zeta_sweep.jl
julia --project=benchmark benchmark/experiments/run_all.jl
julia --project=benchmark benchmark/experiments/run_all.jl 1 3 5
```

Two design points worth stating, since both are easy to get wrong.

**Experiment 7 conditions on inactivity.** The convergence order is estimated
only over the final run of *inactive* iterations. Including active ones mixes a
linear phase with the asymptotic one and returns an order between the two,
which is an artefact rather than a finding. Configurations whose constraint
never stops binding report `never` and no order — that is the result, not a gap
in it.

**Experiments 3 and 4 test different things despite looking alike.** A small ζ
costs *reliability* while leaving efficiency intact. A small μ_max does that
too, but with truncated CG it additionally makes CG truncate on its first
iteration, so the step is the Cauchy point and the model Hessian stops
influencing the direction — with a quasi-Newton model that can change which
critical point is reached. `exp4_cauchy_diagnostic.txt` measures it directly
via `cos(s, −g) = 1`.

---

## 7. Quick start

```julia
using TrustRegionRadius, ADNLPModels

nlp = ADNLPModel(x -> (1 - x[1])^2 + 100(x[2] - x[1]^2)^2, [-1.2, 1.0])

stats = tr_solve(nlp;
    rule      = RGrad(),
    model     = SR1Model(mem = 5),
    subsolver = SteihaugCG(),
    params    = TRParams(tol = 1e-8),
    trace     = true)

stats.status                                       # :first_order
count(stats.solver_specific[:active_trajectory])   # iterations with ‖s‖ = Δ
```

A ζ sweep with a profile:

```julia
configs = sweep_configs("ζ", [0.01, 0.1, 1.0, 10.0, 100.0], ζ -> RDFO(ζ = ζ))
T, _    = run_matrix(problems, configs)
summarise(T, [c.label for c in configs])
τ, prof = performance_profile(T)
open(io -> profile_to_pgfplots(io, τ, prof, [c.label for c in configs]),
     "prof_zeta.tex", "w")
```

---

## 8. What to check first

In rough order of how likely they are to bite:

1. `LBFGSOperator(Float64, n, mem = m)` and `SR1Operator(Float64, n, mem = m)`
   against your installed LinearOperators version.
2. `set_solver_specific!` — the name and signature vary across SolverCore
   releases; it is used in `Trust-region/solver.jl` under `trace = true`.
3. `CUTEst.select_sif_problems` keyword names; `harness.jl` falls back to
   `CUTEst.select`, but the `max_con` filter is worth checking on a small
   `limit` first.
4. The `_apply!` fallback in `subproblem.jl` wraps `mul!` in a `try`. That is
   deliberate — `UniformScaling` supports `*` but not a three-argument `mul!`
   for every element type — but a `try` in an inner loop is worth replacing
   with a proper trait dispatch if it shows up in a profile.
