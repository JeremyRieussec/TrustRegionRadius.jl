# Task 7: slow saddles and the radius axis

A criticality-anchored radius cannot escape a slow saddle in bounded time, with
perfect curvature information. The proposition, its proof, and the experiments
that measure its constants.

## Reproducing

The experiment runs through `initialisation.jl`, not as a script. Experiment
files carry no `using` and no `include`, so handing the path to `julia` directly
fails with `UndefVarError`.

```bash
julia --project=benchmark -e 'include("benchmark/initialisation.jl"); slow_saddle()'
```

That writes one timestamped archive under `benchmark/results/`. Copy its
figures beside the report and build:

```bash
pdflatex -interaction=nonstopmode task7-slow-saddle.tex
```

Two auxiliary scripts under `_audit/` support the report and are not part of the
deliverable:

```bash
julia --project=benchmark _audit/task7_probe.jl
```

checks the three facts the proposition rests on, and

```bash
julia --project=benchmark _audit/task7_findings.jl
```

recomputes the numbers behind Section 5 of the report.

## Environment

| | |
|---|---|
| Julia | 1.11.0 |
| CUTEst.jl | 1.4.0 (not used here) |
| NLPModels | 0.21.12 |
| ADNLPModels | 0.8.13 |
| SolverCore | 0.3.10 |
| Krylov | 0.10.8 |
| LinearOperators | 2.14.2 |
| Plots | 1.41.6 |
| Machine | Intel Core i7-8565U @ 1.80 GHz, 8 logical cores, 15.8 GB |
| OS | Windows 11 Pro 10.0.26200, `x86_64-w64-mingw32` |
| Julia threads | 1 |
| Wall time | 3.0 minutes for 1009 runs |

The wall time was measured with an unrelated CUTEst campaign running on the same
machine, so it is an upper bound rather than a clean figure.

## The grid

| axis | values |
|---|---|
| `ε` | `1e-1, 1e-2, 1e-3, 1e-4, 1e-5` |
| `μ̄` | `0.01, 0.1, 1.0, 10.0` |
| `ζ` | `0.01, 0.1, 1.0, 10.0` |
| `y₀` | `1e-3, 1e-6` |
| `x₀` | `0.0, 0.8` |
| rules | `RDelta` (comparator), `RGradCapped`, `RDFO`, `RGradCappedTau`, `RDFOTau` |
| model | `ExactHessian` throughout |
| subsolver | `SteihaugCG` throughout |

The true Hessian is the point: the stall has to appear with perfect curvature
information, or it is a statement about the model rather than about the radius.

Runs that do not reach `|y_k| > 1/2` within 50000 iterations are reported
`never` and are never treated as a number.

## Two tolerances, never pooled

| regime | `tol` | purpose |
|---|---|---|
| `rate` | `1e-14` | lets the dynamics run so `k_esc` and the fitted rate are measurable |
| `outcome` | `1e-5` | the value `SOLVER_PARAMS` carries, where false convergence appears |
| `second_order` | `1e-5` with `tol_H = 1e-6` | reruns the false-convergence cells |

Neither tolerance was chosen to make a prediction hold. The radius measure `ω_k`
and the stopping test `tol_H` are two different uses of second-order information
and are reported separately everywhere.

## The `RDFO` entry arm

`RDFO` from `Δ₀ = 1` starts at `Δ/‖g‖ ≈ 10⁶` to `10⁸`, far outside the
criticality-controlled band, and escapes at `k = 1` for every `ζ` and every `ε`.
The proposition says nothing about such a run. A separate arm starts at
`Δ₀ = ζ‖g₀‖`, which satisfies the entry condition at `k₀ = 0` by construction,
and is the only arm on which the `RDFO` branch can be tested at all.

## Archive contents

| file | contents |
|---|---|
| `tables/exp15_grid_grad.txt` | the `(μ̄, ε)` grid, fitted rate against `ln(1+μ̄ε)`, the collapse |
| `tables/exp15_grid_dfo.txt` | the `(ζ, ε)` grid from `Δ₀ = 1`, where entry never holds |
| `tables/exp15_grid_dfo_entry.txt` | the same grid started inside the band |
| `tables/exp15_comparator.txt` | `RDelta` from the same starting points |
| `tables/exp15_measures.txt` | `‖g‖` against `τ`, the escape-time ratio, the curvature cost |
| `tables/exp15_outcomes.txt` | the three outcomes, and what `tol_H` changes |
| `figures/exp15_fig1..6.pdf` | the six figures of the report |
| `data/slow_saddle_rows.jld2` | every run, reduced |

## Files added

Nothing under `src/`, no existing experiment file, and neither `config.jl` nor
`harness.jl` was modified. One line was added to `benchmark/initialisation.jl`
to register the new experiment, which is how every experiment is loaded.

- `benchmark/experiments/exp15_slow_saddle.jl`
- `report/task7-slow-saddle/task7-slow-saddle.tex`
- `report/task7-slow-saddle/README.md`
- `_audit/task7_probe.jl`, `_audit/task7_findings.jl`
