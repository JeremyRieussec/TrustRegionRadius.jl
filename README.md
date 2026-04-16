# TrustRegionRadius.jl

[![CI](https://github.com/JeremyRieussec/TrustRegionRadius.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JeremyRieussec/TrustRegionRadius.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/JeremyRieussec/TrustRegionRadius.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JeremyRieussec/TrustRegionRadius.jl)

A Julia package for implementing and benchmarking trust-region radius update mechanisms.

The package provides canonical implementations of seven radius update rules (R1–R4 and the Hei family), a trust-region solver with truncated CG subproblem, and a full CUTEst benchmark harness with seven experiment scripts that reproduce the numerical results of the companion papers.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Radius Update Rules](#radius-update-rules)
  - [R1 — Classical (Conn–Gould–Toint)](#r1--classical-conngould-toint)
  - [R2 — Step-size proportional](#r2--step-size-proportional)
  - [R3 — DFO-like (Scheinberg)](#r3--dfo-like-scheinberg)
  - [R4 — Relative-gradient (Yuan–Fan)](#r4--relative-gradient-yuanfan)
  - [Hei family](#hei-family)
- [Solver Interface](#solver-interface)
- [Implementing a Custom Rule](#implementing-a-custom-rule)
- [Benchmark Workflow](#benchmark-workflow)
- [Experiment Scripts](#experiment-scripts)
- [Repository Layout](#repository-layout)

---

## Installation

The package is not yet registered. Install directly from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/JeremyRieussec/TrustRegionRadius.jl")
```

Or clone and develop locally:

```bash
git clone https://github.com/JeremyRieussec/TrustRegionRadius.jl
cd TrustRegionRadius.jl
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

---

## Quick Start

```julia
using TrustRegionRadius
using ADNLPModels   # or CUTEst

# Define a problem
nlp = ADNLPModel(
    x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2,
    [-1.2, 1.0],
)

# Choose a radius update rule
rule = R1ClassicalUpdate(0.25, 0.50, 2.0)   # γ₁, γ₂, γ₃

# Set solver parameters
params = TRSolverParams(
    η₁             = 0.1,
    η₂             = 0.9,
    Δ₀             = 1.0,
    max_iterations = 10_000,
    tol            = 1e-5,
)

# Solve
out = trust_region_solver(nlp, rule, params)

println(out.status)          # :solved / :max_iter / :failure
println(out.iterations)      # number of iterations
println(out.final_grad_norm) # ‖∇f(x*)‖ at termination
```

---

## Radius Update Rules

All rules implement the `AbstractRadiusUpdate` interface:

```julia
update_radius!(rule, Δ, ρ, η₁, η₂, s_norm, g_norm_old, g_norm_new) -> Float64
initial_radius(rule, Δ₀, g_norm) -> Float64
```

| Argument | Description |
|----------|-------------|
| `Δ` | current radius |
| `ρ` | ratio actual/predicted reduction |
| `η₁`, `η₂` | acceptance thresholds (from `TRSolverParams`) |
| `s_norm` | ‖s_k‖, norm of the computed step |
| `g_norm_old` | ‖g_k‖ before the accept/reject decision |
| `g_norm_new` | ‖g_{k+1}‖ after the accept/reject decision |

### R1 — Classical (Conn–Gould–Toint)

Three-case multiplicative update:

```
ρ ≥ η₂          →  Δ ← γ₃ · Δ    (very successful: expand)
η₁ ≤ ρ < η₂    →  Δ ← γ₂ · Δ    (acceptable: maintain)
ρ < η₁          →  Δ ← γ₁ · Δ    (poor: contract)
```

```julia
rule = R1ClassicalUpdate(γ₁, γ₂, γ₃)
# Recommended: γ₁ ∈ (0,1), γ₂ ≤ 1, γ₃ ≥ 1
# Default:     R1ClassicalUpdate(0.25, 0.50, 2.0)
```

### R2 — Step-size proportional

Sets the radius proportional to the last accepted step norm ‖s_k‖:

```
ρ < η₁          →  Δ ← γ₁ · Δ       (rejected: contract current radius)
η₁ ≤ ρ < η₂    →  Δ ← γ₂ · ‖s_k‖   (acceptable)
ρ ≥ η₂          →  Δ ← γ₃ · ‖s_k‖   (very successful)
```

```julia
rule = R2StepSizeUpdate(γ₁, γ₂, γ₃)
# Default: R2StepSizeUpdate(0.25, 0.80, 2.0)
```

### R3 — DFO-like (Scheinberg)

Prevents expansion when the radius is already large relative to the gradient:

```
ρ ≥ η₁  and  Δ_k ≤ ζ · ‖g_k‖   →  Δ ← γ₃ · Δ   (expand)
ρ ≥ η₁  and  Δ_k  > ζ · ‖g_k‖   →  Δ ← γ₂ · Δ   (no expand)
ρ < η₁                            →  Δ ← γ₁ · Δ   (contract)
```

```julia
rule = R3DFOLikeUpdate(γ₁, γ₂, γ₃, ζ)
# Default: R3DFOLikeUpdate(0.25, 0.50, 2.0, 1.0)
```

### R4 — Relative-gradient (Yuan–Fan)

Maintains `Δ_{k+1} = μ_{k+1} · ‖g_{k+1}‖` with an adaptive multiplier μ:

```
ρ ≥ η₂  and  ‖s_k‖ > 0.5 · Δ_k   →  μ ← γ₂ · μ   (expand)
ρ < η₁                             →  μ ← γ₁ · μ   (contract)
otherwise                          →  μ unchanged
Δ_{k+1} = μ_{k+1} · ‖g_{k+1}‖
```

The guard `‖s_k‖ > 0.5·Δ_k` prevents expansion when the CG step is well inside
the trust region (near-quadratic regime where a larger radius would not help).
Initial radius: `Δ₀ = μ · ‖g₀‖` (ignores `params.Δ₀`).

```julia
rule = R4RelativeGradUpdate(γ₁, γ₂, μ)   # μ = initial multiplier
# Default: R4RelativeGradUpdate(0.25, 2.0, 1.0)
```

> **Note:** R4 is a `mutable struct` (μ is updated in place). Always create a
> fresh instance for each solver run — use a factory function in benchmarks.

### Hei family

All three variants use a piecewise-exponential multiplier:

```
R_exp(ρ) = β + (1 − γ₁ − β) · exp(λ₁(ρ − η))    if ρ < η   (contract)
           1 + γ₂ + (M − 1 − γ₂) · (1 − exp(−λ₂(ρ − η)))  if ρ ≥ η   (expand)
```

**`HeiUpdate`** — radius proportional to step norm:
```julia
Δ_{k+1} = R_exp(ρ_k) · ‖s_k‖
rule = HeiUpdate(η, β, γ₁, γ₂, M, λ₁, λ₂)
```

**`HeiGradUpdate`** — radius proportional to gradient norm:
```julia
Δ_{k+1} = R_exp(ρ_k) · ‖g_{k+1}‖
rule = HeiGradUpdate(η, β, γ₁, γ₂, M, λ₁, λ₂)
```

**`HeiFanYuanUpdate`** — adaptive multiplier μ with gradient-based radius:
```julia
μ_{k+1} = μ_k · R_exp(ρ_k)
Δ_{k+1} = μ_{k+1} · ‖g_{k+1}‖
rule = HeiFanYuanUpdate(μ, η, β, γ₁, γ₂, M, λ₁, λ₂)
```

> **Note:** `HeiFanYuanUpdate` is a `mutable struct`. Use a factory function.

Default parameters used in benchmarks:
```julia
HeiUpdate(       0.1, 0.1, 0.1, 0.5, 10.0, 0.5, 0.1)
HeiGradUpdate(   0.1, 0.1, 0.1, 0.5, 10.0, 0.5, 0.1)
HeiFanYuanUpdate(0.1, 0.1, 0.1, 0.1, 0.5, 10.0, 0.5, 0.1)
```

---

## Solver Interface

### `trust_region_solver`

```julia
out = trust_region_solver(nlp, rule, params)
```

Runs a trust-region method with truncated CG Steihaug subproblem solver.
The step is accepted when `ρ ≥ η₁`.

### `TRSolverParams`

```julia
params = TRSolverParams(
    η₁             = 0.1,     # lower acceptance threshold
    η₂             = 0.9,     # upper threshold ("very successful")
    Δ₀             = 1.0,     # initial radius (ignored by R4 and HeiFanYuan)
    max_iterations = 10_000,  # iteration budget
    tol            = 1e-5,    # ‖∇f‖ convergence tolerance
)
```

### `TROutput`

| Field | Type | Description |
|-------|------|-------------|
| `status` | `Symbol` | `:solved`, `:max_iter`, or `:failure` |
| `iterations` | `Int` | number of iterations performed |
| `f_evals` | `Int` | objective function evaluations |
| `g_evals` | `Int` | gradient evaluations |
| `h_evals` | `Int` | Hessian evaluations |
| `h_prod_evals` | `Int` | Hessian–vector product evaluations |
| `final_grad_norm` | `Float64` | ‖∇f(x*)‖ at termination |
| `final_delta` | `Float64` | trust-region radius Δ at termination |
| `delta_trajectory` | `Vector{Float64}` | Δ_k for k = 0, 1, … |
| `grad_norm_trajectory` | `Vector{Float64}` | ‖g_k‖ for k = 0, 1, … |
| `obj_trajectory` | `Vector{Float64}` | f(x_k) for k = 0, 1, … |
| `solve_time` | `Float64` | wall-clock time in seconds |

---

## Implementing a Custom Rule

Subtype `AbstractRadiusUpdate` and implement two methods:

```julia
struct MyUpdate <: AbstractRadiusUpdate
    γ_contract::Float64
    γ_expand::Float64
end

function TrustRegionRadius.update_radius!(
        rule::MyUpdate,
        Δ::Float64, ρ::Float64, η₁::Float64, η₂::Float64,
        s_norm::Float64, g_norm_old::Float64, g_norm_new::Float64)
    return ρ >= η₂ ? rule.γ_expand * Δ : rule.γ_contract * Δ
end

# Optional: override initial radius (default returns Δ₀)
TrustRegionRadius.initial_radius(::MyUpdate, Δ₀::Float64, ::Float64) = Δ₀
```

Then pass it directly to `trust_region_solver`:

```julia
rule = MyUpdate(0.25, 2.0)
out  = trust_region_solver(nlp, rule, params)
```

---

## Benchmark Workflow

The benchmark harness runs every configured rule on the full CUTEst unconstrained
problem suite and archives the results in a self-documented, timestamped directory.

### Prerequisites

- [CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl) installed and
  the `MASTSIF` environment variable pointing to the SIF decoder database.
- Benchmark dependencies installed:
  ```bash
  julia --project=benchmark -e "using Pkg; Pkg.instantiate()"
  ```

### Step 1 — Run the benchmark

```bash
julia --project=benchmark benchmark/run_benchmark.jl
```

- Iterates over all unconstrained CUTEst problems with 2 ≤ n ≤ 500.
- Runs every rule listed in `benchmark/config.jl`.
- Saves one JLD2 file per (problem, rule) pair to `benchmark/results/exp_TIMESTAMP/jld2/`.
- Writes `experiment_config.toml` into the same directory recording all solver
  and rule parameters (ASCII-keyed TOML, fully reproducible).
- Already-completed (problem, rule) pairs are skipped on re-run.

### Step 2 — Generate figures

```bash
julia --project=benchmark benchmark/generate_all_figures.jl exp_2026-04-16_00-40-55
```

Pass the name of the archive directory under `benchmark/results/`.

- Runs exp1–exp7, writing PDFs and LaTeX tables into the archive.
- Writes `experiment_summary.md` with parameter tables and output inventory.

To skip specific experiments while iterating:

```bash
SKIP_EXP3=1 SKIP_EXP4=1 julia --project=benchmark benchmark/generate_all_figures.jl exp_2026-04-16_00-40-55
```

### Configuration

Edit `benchmark/config.jl` — no other file needs to change:

```julia
const SOLVER_PARAMS = TRSolverParams(η₁=0.1, η₂=0.9, Δ₀=1.0, max_iterations=10_000, tol=1e-5)

const RULES = [
    ("R1",     () -> R1ClassicalUpdate(0.25, 0.50, 2.0)),
    ("R2",     () -> R2StepSizeUpdate(0.25, 0.80, 2.0)),
    ("R3",     () -> R3DFOLikeUpdate(0.25, 0.50, 2.0, 1.0)),
    ("R4",     () -> R4RelativeGradUpdate(0.25, 2.0, 1.0)),
    ("R4-Alt", () -> R4RelativeGradUpdate(0.25, 2.0, 0.5)),
    ("Hei",    () -> HeiUpdate(       0.1, 0.1, 0.1, 0.5, 10.0, 0.5, 0.1)),
    ("HeiG",   () -> HeiGradUpdate(   0.1, 0.1, 0.1, 0.5, 10.0, 0.5, 0.1)),
    ("HFY",    () -> HeiFanYuanUpdate(0.1, 0.1, 0.1, 0.1, 0.5, 10.0, 0.5, 0.1)),
]

const MIN_VAR = 2     # minimum number of variables
const MAX_VAR = 500   # maximum number of variables
const MAX_CON = 0     # unconstrained only
```

### Archive layout

```
benchmark/results/exp_2026-04-16_00-40-55/
  experiment_config.toml   # exact parameters used (TOML, ASCII keys)
  experiment_summary.md    # human-readable summary and output inventory
  jld2/                    # one <PROBLEM>_<RULE>.jld2 per (problem, rule) pair
  figures/                 # PDFs from exp1–exp7
  tables/                  # LaTeX .txt files from exp1–exp7
```

---

## Experiment Scripts

| Script | Outputs | Description |
|--------|---------|-------------|
| `exp1_global.jl` | perf profiles, data profile, success-rate table | Global comparison of all rules on the full CUTEst suite (Dolan–Moré profiles) |
| `exp2_radius_regime.jl` | trajectory PDFs per problem | Δ_k, ‖g_k‖, ratio Δ_k/‖g_k‖, and cumulative Σ Δ_k for representative problems |
| `exp3_zeta.jl` | perf profile, solve-rate plot, table | R3 sensitivity sweep over ζ ∈ {0.1, 0.25, 0.5, 1, 2, 5, 10} |
| `exp4_mu.jl` | perf profile, solve-rate plot, μ-drift scatter | R4 sensitivity sweep over μ₀ |
| `exp5_illcond.jl` | perf profile, bar chart, table | Comparison on ill-conditioned problems (proxy: ≥ 100 iterations for at least one rule) |
| `exp6_delta0.jl` | per-rule perf profiles, solve-rate/median-iter plots | Δ₀ sensitivity sweep for R1, R2, R3 (R4 excluded — it ignores Δ₀) |
| `exp7_superlinear.jl` | convergence-order boxplot, scatter vs n | Asymptotic convergence order q estimated by tail linear regression of log ‖g_k‖ |

---

## Repository Layout

```
TrustRegionRadius.jl/
  src/
    TrustRegionRadius.jl          # module entry point, exports
    Radius_updates/
      canonical_R1R2R3R4.jl       # R1, R2, R3, R4 structs + update_radius!
      hei_canonical.jl             # HeiUpdate, HeiGradUpdate, HeiFanYuanUpdate
    Trust-region/
      trust_region_solver.jl       # main solver (AbstractRadiusUpdate dispatch)
    Subproblem/
      Truncated_CG.jl              # truncated CG Steihaug
    Saving_info/
      TROutput.jl                  # TROutput + TRSolverParams structs
    Stopping_tests/                # gradient-norm stopping criterion
    State/                         # internal solver state types
  benchmark/
    config.jl                      # SOLVER_PARAMS, RULES, problem selection bounds
    config_utils.jl                # TOML serialisation utilities
    run_benchmark.jl               # CUTEst benchmark loop → results/exp_TIMESTAMP/jld2/
    generate_all_figures.jl        # figure generation + summary for a given archive
    load_results.jl                # JLD2 loading utilities (load_all_results, etc.)
    exp1_global.jl – exp7_superlinear.jl   # individual experiment scripts
  test/
    ...
```
