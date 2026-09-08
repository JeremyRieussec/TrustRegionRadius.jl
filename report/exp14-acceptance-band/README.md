# Experiment 14: decoupling acceptance from scaling

What it costs to accept a step the radius does not trust. `η` is swept from `0`
to `η₁` at fixed `η₁`, for the eight radius rules Part III marks tested, on the
185 unconstrained CUTEst problems with `2 ≤ n ≤ 200`.

## Reproducing

From the repository root. The experiment runs through `initialisation.jl`, not
as a script: the file carries no `using` and no `include`, so handing its path
to `julia` directly fails with `UndefVarError: RULES`.

```bash
julia --project=benchmark -e 'include("benchmark/initialisation.jl"); acceptance_band()'
```

That writes one timestamped archive under `benchmark/results/`. The run is
resumable at problem granularity, one JLD2 per problem, so an interrupted
campaign continues where it stopped with

```bash
TRR_RESUME=benchmark/results/exp_YYYY-MM-DD_HH-MM-SS_acceptance_band julia --project=benchmark -e 'include("benchmark/initialisation.jl"); acceptance_band()'
```

Then turn the archive into the LaTeX fragments the report inputs:

```bash
julia --project=benchmark report/exp14-acceptance-band/make_tables.jl benchmark/results/exp_YYYY-MM-DD_HH-MM-SS_acceptance_band
```

and build the report:

```bash
pdflatex -interaction=nonstopmode exp14-acceptance-band.tex
```

`TRR_AB_LIMIT=n` restricts the run to the first `n` problems. That is a pilot,
not the experiment. The realised problem count is written into
`experiment_config.toml` and the pilot note into `experiment_summary.md`, so a
pilot archive cannot be mistaken for a full one.

The experiment refuses to run without CUTEst. `default_problems()` falls back to
eight analytic problems when the CUTEst query returns empty, which would produce
a table that reads as a CUTEst benchmark without being one.

## Environment

| | |
|---|---|
| Julia | 1.11.0 |
| CUTEst.jl | 1.4.0 |
| NLPModels | 0.21.12 |
| ADNLPModels | 0.8.13 |
| SolverCore | 0.3.10 |
| Krylov | 0.10.8 |
| LinearOperators | 2.14.2 |
| Plots | 1.41.6 |
| Machine | Intel Core i7-8565U @ 1.80 GHz, 8 logical cores, 15.8 GB |
| OS | Windows 11 Pro 10.0.26200, `x86_64-w64-mingw32` |
| Julia threads | 1 |
| Wall time | WALLTIME-PLACEHOLDER |

The exact package set is pinned by `benchmark/Manifest.toml`, whose content hash
is recorded in every archive's `experiment_config.toml` under `[provenance]`,
alongside the git commit and a `git_dirty` flag.

## What the archive contains

| file | contents |
|---|---|
| `tables/exp14_problem_set.txt` | the realised problem names and dimensions |
| `tables/exp14_occupancy.txt` | band occupancy per rule and η, spread across problems |
| `tables/exp14_paired_fevals.txt` | paired counts on objective evaluations |
| `tables/exp14_paired_gevals.txt` | paired counts on gradient evaluations |
| `tables/exp14_paired_hprods.txt` | paired counts on Hessian-vector products |
| `tables/exp14_paired_iterations.txt` | paired counts on iterations, secondary |
| `tables/exp14_exclusive.txt` | problems solved by exactly one setting, named |
| `tables/exp14_status.txt` | full status breakdown, no filtering |
| `tables/exp14_eta0_rounding.txt` | the η = 0 column with and without rounding-level iterations |
| `tables/exp14_limit_points.txt` | problems where the limit point moved |
| `tables/exp14_retrospective.txt` | which rules branch on a ratio other than ρ_k |
| `tables/exp14_refused.txt` | configurations refused at construction |
| `figures/exp14_occupancy_*.pdf` | occupancy against η, one series per rule |
| `figures/exp14_logratio_*.pdf` | paired log ratio against η, one panel per rule |
| `figures/exp14_workmix_*.pdf` | f, ∇f and ∇²f·v evaluations stacked against η |
| `figures/exp14_profile_*.pdf` | performance profiles on evaluations, η = 0 against η = η₁ |
| `data/<PROBLEM>__band.jld2` | every run on that problem, all 80 configurations |

Read `exp14_occupancy.txt` first. The band is what the experiment is about, and
no effect reported anywhere else can exceed the exposure recorded there. The
column at `η/η₁ = 1` is zero by construction and is the check that the
instrument works.

## Design notes

`η` is swept over `{0, ¼η₁, ½η₁, ¾η₁, η₁}`. The last is the classical coupling
and the baseline of every comparison. Two strata, `η₁ = 0.1` (the value the
Part III campaign uses) and `η₁ = 0.4`. Changing `η₁` changes the rule's own
behaviour, so the strata are a separate axis and are never pooled.

Model Hessian and subproblem solver are held fixed at `ExactHessian` and
`SteihaugCG(max_iters = 1000)`, so the swept threshold is the only difference
within a rule. `ExactMS` is admissible on all 185 problems at `n ≤ 200` and is
not used, so no subsolver switch is confounded with the sweep.

The file does not call `run_experiment`. That function takes one `TRParams` for
every configuration, and this sweep varies `η` per configuration; it also builds
a `RunRecord`, which does not carry `:accepted_trajectory`, without which the
occupancy count cannot be formed. The loop opens each CUTEst model once and
calls `tr_solve`, the package entry point, against all 80 configurations.

## Files added

Nothing under `src/`, no existing experiment file, and neither `config.jl` nor
`harness.jl` was modified. One line was added to `benchmark/initialisation.jl`
to register the new experiment, which is how every experiment is loaded.

- `benchmark/experiments/exp14_acceptance_band.jl`
- `report/exp14-acceptance-band/exp14-acceptance-band.tex`
- `report/exp14-acceptance-band/make_tables.jl`
- `report/exp14-acceptance-band/README.md`
