# TrustRegionRadius

[![CI](https://github.com/JeremyRieussec/TrustRegionRadius.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JeremyRieussec/TrustRegionRadius.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/JeremyRieussec/TrustRegionRadius.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JeremyRieussec/TrustRegionRadius.jl)

TrustRegionRadius is a Julia package for implementing and testing trust-region radius update mechanisms.
It provides canonical implementations of R1–R4 and Hei-family update rules, a CUTEst benchmark
harness, and experiment scripts that reproduce the figures from the companion COAP papers.

## Features

- Canonical radius update rules: R1 (classical), R2 (step-size), R3 (DFO-like), R4 (relative-grad),
  and the Hei/HeiGrad/HeiFanYuan exponential-multiplier family.
- Trust-region solver (`trust_region_solver`) dispatching through the `AbstractRadiusUpdate` interface.
- CUTEst benchmark harness with per-(problem, rule) JLD2 result files.
- Seven experiment scripts (exp1–exp7) covering global comparison, trajectory analysis,
  parameter sensitivity, ill-conditioned problems, and convergence-order estimation.
- Timestamped archive workflow: every benchmark run is self-documented and reproducible.

## Installation

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Benchmark Workflow

Results are produced in two stages: first run the solver on all CUTEst problems, then
generate all figures.  Intermediate results land in `benchmark/temp_results/`; the
final archive is written to `benchmark/results/exp_YYYY-MM-DD_HH-MM-SS/`.

### Step 1 — Run the benchmark

```bash
julia --project=benchmark benchmark/run_benchmark.jl
```

- Iterates over all unconstrained CUTEst problems with 2 ≤ n ≤ 500.
- Runs every rule listed in `benchmark/config.jl` (currently R1–R4, R4-Alt, Hei, HeiG, HFY).
- Saves one JLD2 file per (problem, rule) pair to `benchmark/temp_results/`.
- Writes `benchmark/temp_results/experiment_config.toml` recording all solver and rule
  parameters so the run is exactly reproducible.
- Already-completed (problem, rule) pairs are skipped automatically; pass `--force` to
  discard existing temp results and restart from scratch.

To edit solver parameters or the list of rules, modify `benchmark/config.jl` — no other
file needs to change.

### Step 2 — Generate figures and archive

```bash
julia --project=benchmark benchmark/generate_all_figures.jl
```

- Reads JLD2 results from `benchmark/temp_results/`.
- Runs exp1–exp7 in sequence, writing PDFs to `benchmark/figures/` and LaTeX tables to
  `benchmark/tables/`.
- Creates a timestamped archive directory `benchmark/results/exp_YYYY-MM-DD_HH-MM-SS/`
  containing: all JLD2 files, all figures, all tables, the config TOML, and a
  human-readable `experiment_summary.md`.
- Clears `benchmark/temp_results/` after a successful archive.

To skip specific experiments (e.g. while iterating on one script):

```bash
SKIP_EXP3=1 SKIP_EXP4=1 julia --project=benchmark benchmark/generate_all_figures.jl
```

### Configuration

Edit `benchmark/config.jl` to change:

| Setting | Description |
|---------|-------------|
| `SOLVER_PARAMS` | η₁, η₂, Δ₀, max\_iterations, tol |
| `RULES` | list of `(name, factory)` pairs for radius update rules |
| `MIN_VAR`, `MAX_VAR`, `MAX_CON` | CUTEst problem selection bounds |

### Archive layout

```
benchmark/results/exp_2026-04-15_10-30-00/
  experiment_config.toml   # solver params + rule params (TOML, ASCII keys)
  experiment_summary.md    # human-readable summary with parameter tables
  jld2/                    # raw per-(problem, rule) result files
  figures/                 # all PDFs from exp1–exp7
  tables/                  # all LaTeX table files from exp1–exp7
```

## Experiment Scripts

| Script | Description |
|--------|-------------|
| `exp1_global.jl` | Performance and data profiles across all rules on the full CUTEst suite |
| `exp2_radius_regime.jl` | Trajectory plots (Δ_k, ‖g_k‖, ratio, cumulative sum) for representative problems |
| `exp3_zeta.jl` | R3 sensitivity sweep over ζ ∈ {0.1, 0.25, 0.5, 1, 2, 5, 10} |
| `exp4_mu.jl` | R4 sensitivity sweep over μ₀ |
| `exp5_illcond.jl` | Comparison on ill-conditioned problems |
| `exp6_delta0.jl` | Δ₀ sensitivity sweep for R1, R2, R3 |
| `exp7_superlinear.jl` | Convergence-order estimation (tail fit of log ‖g_k‖) |
