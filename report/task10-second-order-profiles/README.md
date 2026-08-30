# Performance profiles for the second-order rules on CUTEst

Task 10. Three profiles: P1 among the second-order rules, P2 and P3 the two
families against each other on the first-order and the second-order task in turn.

Every run is scored afterwards, from the point it returned, by one criterion
applied by the same code to every run. No profile here is drawn on the solver's
own status. The two families stop on different tests, so profiling them on their
own statuses would compare the cost of an easy task with the cost of a hard one.

## Reproducing

From the root of `TrustRegionRadius.jl`. The experiment runs through
`initialisation.jl`, not as a script: experiment files carry no `using` and no
`include`, so handing the path to `julia` fails with `UndefVarError`.

```bash
TRR_SO_MAXVAR=1000 TRR_SO_MAXTIME=60 julia --project=benchmark -e 'include("benchmark/initialisation.jl"); second_order_profiles()'
```

That writes a timestamped archive under `benchmark/results/`. Copy its figures
into `figs/` and its tables beside this file, then build the report:

```bash
pdflatex -interaction=nonstopmode task10-second-order-profiles.tex
```

The campaign is resumable per `(problem, rule, arm, pass)`. To continue an
interrupted one into the same archive:

```bash
TRR_SO_RESUME=benchmark/results/<archive> julia --project=benchmark -e 'include("benchmark/initialisation.jl"); second_order_profiles()'
```

| variable | meaning | default |
|---|---|---|
| `TRR_SO_MAXVAR` | upper bound on `n` | `MAX_VAR` of `config.jl`, 1000 |
| `TRR_SO_LIMIT` | first N problems only, for a pilot | all |
| `TRR_SO_MAXTIME` | seconds per run | `SOLVER_PARAMS.max_time`, 120 |
| `TRR_SO_RESUME` | an existing archive to continue into | a fresh archive |

The campaign reported here ran at `TRR_SO_MAXTIME=60`, half the shared default.
The budget is wall clock, it is identical across every column, and runs that
exhaust it appear as `max_time` in the status tables rather than being dropped.

## Environment

| | |
|---|---|
| Julia | 1.11.0 |
| CUTEst | 1.4.0 |
| NLPModels | 0.21.12 |
| ADNLPModels | 0.8.13 |
| SolverCore | 0.3.10 |
| Krylov | 0.10.8 |
| LinearOperators | 2.14.2 |
| JLD2 | 0.6.5 |
| Plots | 1.41.6 |
| Machine | Intel Core i7-8565U @ 1.80 GHz, 8 logical cores, 15.8 GB |
| OS | Windows 11 Pro 10.0.26200 |
| Julia threads | 1 |
| Wall time | PLACEHOLDER-WALL |

An unrelated Julia campaign was running on the same machine throughout, so the
wall time is an upper bound rather than a clean measurement of this experiment
alone. The evaluation counts, which every profile is drawn on, are unaffected.

## The problem set

CUTEst, unconstrained, all variables free, `2 <= n <= 1000`, no constraints.
The query returns **196** problems, **185** with `n <= 200` and **11** above.
There is no analytic fallback: the experiment checks `HAS_CUTEST`, checks that
the query returned something, and raises rather than falling back, because
`default_problems()` would otherwise produce a table that reads as a CUTEst
benchmark and is not one.

P3 runs on the 185 problems with `n <= 200`. Above that dimension the package
estimates the smallest eigenvalue by Lanczos; a Ritz value over-states it, so the
second-order test becomes optimistic and scoring P3 there would rank runs by the
machinery the profile exists to compare.

## The two passes

The trace carries no per-iteration evaluation counts, so the cost a run had spent
when it first met a criterion is not recoverable from an archived run. Each
configuration is therefore run twice, with the tightened criterion as its actual
stopping rule.

| pass | halts at the first iterate with | scores |
|---|---|---|
| `g` | `‖g‖ <= 1e-5` | P2 |
| `h` | `‖g‖ <= 1e-5` and `λ_min >= -1e-6` | P1, P3 |

Both passes keep their arm's own `tol_H`, which controls the curvature estimate as
well as the stopping test, so the second-order arm pays for its curvature in both.
Pass `g` halts through a callback, which is why its runs report `user` rather than
`first_order`.

## What is here

| file | contents |
|---|---|
| `task10-second-order-profiles.tex`, `.pdf` | the standalone report |
| `figs/exp17_P1_*.pdf` | P1, one profile per cost axis |
| `figs/exp17_P2_*.pdf` | P2, likewise |
| `figs/exp17_P3_*.pdf` | P3, likewise |
| `figs/exp17_P?_data.pdf` | the three data profiles |
| `exp17_P1_status.txt`, `exp17_P2_status.txt`, `exp17_P3_status.txt` | status composition per configuration, no filtering |
| `exp17_P1_cost.txt`, `exp17_P2_cost.txt`, `exp17_P3_cost.txt` | totals over every problem, met or not |
| `exp17_checks.txt` | the four checks of section 2 of the report |
| `exp17_problems.txt` | the realised list of names and dimensions |
| `exp17_hazards.txt` | four hazard counts |

## Cost axes

Profiles are given on Hessian-vector products, Hessian evaluations, gradient
evaluations, objective evaluations and iterations. Two axes are needed rather than
one. With `ExactHessian` at `n <= 200` the curvature estimate goes through
`dense_hessian`, hence `NLPModels.hess`, so on the small problems the entire cost
of the second-order machinery is in the Hessian-evaluation count and none of it in
the Hessian-vector count. Above 200 it goes through Lanczos on `hessian_op` and
the cost moves to Hessian-vector products. A profile on either axis alone hides
the overhead on half the problem set.

The iteration-count profiles are a secondary figure, labelled as such.

## Files added to the package

Nothing under `src/`, no existing experiment file, and neither `config.jl` nor
`harness.jl` was modified. No thesis source was edited. One line was added to
`benchmark/initialisation.jl` to register the new experiment, which is how every
experiment is loaded.

- `benchmark/experiments/exp17_second_order_profiles.jl`

The experiment carries its own run loop rather than calling `run_experiment`, for
two reasons. `RunRecord` keeps `neval_hprod` and not `neval_hess`, so the shared
record cannot see the cost of the curvature estimate on the small problems at all.
And the subproblem solver of the second-order arm depends on the dimension of the
problem, which a config factory never sees. The archive layer, the profile
functions and the scoring criterion are the package's own.

## Deviations from the brief, all deliberate

**The experiment is `exp17`, not `exp14`.** `exp14`, `exp15` and `exp16` were
already taken and renumbering an existing file is forbidden.

**The per-run budget is 60 seconds, not the shared 120.** The campaign is
4704 runs on a laptop. The budget is identical across every column, so it cannot
bias the comparison between them; it caps the hardest runs, which then appear as
`max_time` in the status tables.

**The roster is not `config.jl`'s `RULES`.** Section 4 of the brief names six
rules, `R-delta`, `R-step`, `R-RTR`, `R-DFO`, `R-grad` and `R-gRTR`, three that
read a criticality measure and three that do not. `config.jl` comments `RRTR` out
of `RULES` and keeps `RRTRGrad` in a separate `RRTRGRAD_CONFIG`, so that the list
names the eight mechanisms of the paper's table. This experiment builds the six of
the brief directly, at `config.jl`'s shared constants. Relative to `RULES` it adds
`RRTR` and `RRTRGrad` and leaves out `RGradCapped` and `RAdaptiveGrad`, neither of
which the brief names.
