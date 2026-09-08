# Reading the step diagnostics

Companion to `notebooks/Saddle/step_diagnostics_v1.ipynb`. It explains the eight
columns of the step-diagnostic tables, the six functions that produce them, and
what the tables say when read together.

Every number in the report is printed by a cell of that notebook that ran.
`_audit/` carries the script that inserted the tables, and it asserts that each
row it writes appears verbatim in the notebook's own output, so a table here
cannot drift from the cell that produced it.

## Reproducing

From the root of `TrustRegionRadius.jl`. The notebook is built from its driver
and then executed, so the two cannot drift:

```bash
python _audit/build_saddle_notebooks.py nb4_step_diagnostics.jl
```

```bash
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=julia-1.11 --ExecutePreprocessor.timeout=5400 notebooks/Saddle/step_diagnostics_v1.ipynb
```

Then build the report:

```bash
pdflatex -interaction=nonstopmode task11-step-diagnostics.tex
```

The driver also runs on its own, which is the quicker check while editing:

```bash
julia --project=notebooks _audit/nb4_step_diagnostics.jl
```

A clean script run is not evidence that the notebook runs. Cell scope differs
from file scope in Julia, and a top-level function in one cell can collide with a
soft-scope assignment in another. Execute the notebook.

## What the report answers

| section | question |
|---|---|
| 1.1 | the three ways `SteihaugCG` leaves its loop, which everything else follows from |
| 2 | what `trunc`, `inside`, `krylov`, `cauchy`, `cg1` and `cos_min` compute, and how to read each |
| 3 | why `cauchy` and `cg1` ask the same question, why they print the same number on 28 of 30 rows, and what happens on the other two |
| 4 | the six diagnostics, each with what it measures and when it does not apply |
| 5 | the four tables, integrated |

## The three results

**`cg1` is contained in `cauchy`, always.** All three conjugate-gradient exits
return a positive multiple of `-g_k`, so one inner iteration forces `cos = 1`
exactly. The two columns are nested rather than independent, which is why they
agree wherever the containment is tight.

**The near-eigenvector explanation of their disagreement is refuted.**
`eig_deviation` has median `7.0695e-01` at the disagreeing iterations against
`7.0617e-01` at the agreeing ones. What separates them is the gap between the
first conjugate-gradient iterate and the boundary, median `6.536e-09` against
`1.6266e-03`.

**The truncation condition holds on 17755 of 17763 exact-Hessian iterations.**
All eight exceptions are the forcing-condition exit, which `z1_ratio` does not
model. There is no exception in the other direction, so `‖z₁‖/Δ ≥ 1` is
sufficient for truncation and not necessary.

## Environment

| | |
|---|---|
| Julia | 1.11.0 |
| NLPModels | 0.21.12 |
| ADNLPModels | 0.8.13 |
| SolverCore | 0.3.10 |
| Krylov | 0.10.8 |
| LinearOperators | 2.14.2 |
| Plots | 1.41.6 |
| Machine | Intel Core i7-8565U @ 1.80 GHz, 8 logical cores, 15.8 GB |
| OS | Windows 11 Pro 10.0.26200 |

## Files

| file | contents |
|---|---|
| `task11-step-diagnostics.tex`, `.pdf` | the report, 9 pages |
| `figs/` | empty; the notebook writes its figure to `notebooks/Saddle/` |

The notebook's own figure is
`notebooks/Saddle/step_diagnostics_fig1_cauchy_reconcile.pdf`, six panels
comparing `eig_deviation` and the boundary gap at the agreeing and disagreeing
iterations.
