# Flat regions on the radius axis

Task 12. The theory behind the slow-saddle stall, cited rather than reproved, and
the seven figures and three tables of `notebooks/Saddle/flat_regions_v1.ipynb`,
with interpretation.

The model is the true Hessian throughout, so the stall appears with perfect
curvature information. That is the point of the family and every caption says so.

## What is new here

The experiment `exp15_slow_saddle.jl` produces the scalar tables and the
collapse. The report `report/task7-slow-saddle` carries the proofs. This report
adds three things.

**The envelope of `prop: escape` drawn on the same axes as the run.** It is never
crossed, on any run, at any epsilon.

**`as: RCA` holds with equality.** Over 640 iterations, `Δ_k/(μ̄‖g_k‖)` has
minimum and maximum both `1.00000000`, the constraint is active on every
iteration, and `‖s_k‖ = Δ_k` to `1e-8` on every iteration. The assumption is an
inequality and the run satisfies it as an equality, so the escape bound is not
resting on slack.

**The escape-time ratio between the two criticality measures.** The earlier
prediction missed it by three per cent. Accounting for the branch switch in `τ`
recovers it to within 0.3 per cent on the best-resolved cells.

## Reproducing

From the root of `TrustRegionRadius.jl`. The notebook is built from its driver
and then executed, so the two cannot drift:

```bash
python _audit/build_saddle_notebooks.py nb5_flat_regions.jl
```

```bash
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=julia-1.11 --ExecutePreprocessor.timeout=7200 notebooks/Saddle/flat_regions_v1.ipynb
```

Then copy the figures and build:

```bash
cp notebooks/Saddle/flat_regions_fig*.pdf report/task12-flat-regions/figs/
```

```bash
pdflatex -interaction=nonstopmode task12-flat-regions.tex
```

The driver also runs on its own, which is the quicker check while editing:

```bash
julia --project=benchmark _audit/nb5_flat_regions.jl
```

A clean script run is not evidence that the notebook runs. Cell scope differs
from file scope in Julia. Execute the notebook.

## Every number is checked

`_audit/` carries the verification script for this report. It parses every
`tabular`, extracts each row's numbers, and requires that a single line of the
notebook's printed output carries all of them. The comparison is on values and
not on strings, so `1.00000` matches `1.0000e+00` and `10^{-6}` matches `1e-06`.

At the last build: **11 tabulars, 107 numeric rows, 0 not found.**

## Where the numbers come from

| source | what it supplies |
|---|---|
| archive `exp_2026-08-29_21-51-19_slow_saddle` | the scalar grids, 1009 rows |
| the notebook, rerun | the seven figures, 26 runs |
| the notebook, computed | the two integrals, the branch switch, the derived columns |

`exp15` strips `ys` and `deltas` before archiving, through
`Base.structdiff(r, (ys = 0, deltas = 0))`, so any figure that draws a trajectory
must rerun its cells. The three grid tables read the archive and recompute only
the columns marked derived.

## Results in one line each

- **Envelope**: never crossed. Worst `|y|/envelope` is `1.000000`, at `k = 0`
  where the two are equal by construction.
- **Binding**: `Δ_k = μ̄‖g_k‖` to eight decimals on 100% of iterations.
- **`R-DFO` entry**: 9 of 12 cells enter the band at `k₀` between 0 and 22 and
  stall; 3 never enter and **none of the three converges to the saddle**, so the
  second branch of `prop: rdfo` is unoccupied on this scan.
- **Two measures**: predicted ratio `8.6341` and `17.9896` from the corrected
  integral, against fitted `8.6440` and `18.0095` on the best-resolved cells.
- **Outcome maps**: 0 of 40 cells report false convergence at `tol = 1e-14`,
  36 of 40 at `tol = 1e-5`.
- **The gap**: 37 cells, **0 above the bound**; ratio `0.9869` to `1.0000` for
  `μ̄ε ≤ 0.1`, falling to `0.8476` at `μ̄ε = 1`.

## One finding for `Survey-part3-v1.tex`

Line 1150 carries `\ln(1/\abs{y_0})` in `eqn: escape time capped`. Restricted to
the cells where the continuous limit applies, the collapse lands on
`ln(1/(sqrt(3) y0))`, with residuals smaller by a factor of 133 and 72. This is a
different location from the finding `task7-slow-saddle` makes at line 1107, which
concerns the model axis under `R-delta`.

Described, not applied. Both readings stand for the author to choose.

## Environment

| | |
|---|---|
| Julia | 1.11.0 |
| NLPModels | 0.21.12 |
| ADNLPModels | 0.8.13 |
| SolverCore | 0.3.10 |
| Krylov | 0.10.8 |
| LinearOperators | 2.14.2 |
| JLD2 | 0.6.5 |
| Plots | 1.41.6 |
| Machine | Intel Core i7-8565U @ 1.80 GHz, 8 logical cores, 15.8 GB |
| OS | Windows 11 Pro 10.0.26200 |

## Files

| file | contents |
|---|---|
| `task12-flat-regions.tex`, `.pdf` | the report, 17 pages |
| `figs/flat_regions_fig1_paths.pdf` | iterate paths, six panels |
| `figs/flat_regions_fig2_envelope.pdf` | the growth envelope over the runs |
| `figs/flat_regions_fig3_binding.pdf` | `Δ_k`, `C‖g_k‖` and `‖s_k‖` on one run |
| `figs/flat_regions_fig4_dfo_entry.pdf` | the `R-DFO` entry condition, two cases |
| `figs/flat_regions_fig5_measures.pdf` | the two measures, with the predicted curves |
| `figs/flat_regions_fig6_outcomes.pdf` | the outcome map, both tolerance regimes |
| `figs/flat_regions_fig7_gap.pdf` | the fitted rate against the bound |

## Deviations, both deliberate

**`exp15` is not included a second time.** `benchmark/initialisation.jl` already
includes it, and including it again would re-evaluate its `const`s. The notebook
loads the suite once and asserts that all eighteen names it uses came from it.

**Figure 5 is not `exp15_fig6_measures.pdf` repeated.** That figure already plots
both measures at one cell. This one draws the continuous-limit curves over the
observed ones, which ties it to the corrected integral and makes it a test rather
than a picture.
