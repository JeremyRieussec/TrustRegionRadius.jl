# Benchmarking

Two layers: profile functions in the package (pure functions of a cost matrix), and an
experiment harness in `benchmark/` that produces self-documenting archives.

## Profiles

```julia
T = [ 12.0  18.0  Inf ;      # rows: problems, columns: solvers
      30.0  25.0  41.0 ]     # Inf marks a failure

τ, prof = performance_profile(T)
```

`prof[1, s]` is the fraction of problems on which solver `s` is fastest (efficiency);
`prof[end, s]` is the fraction it solves at all (reliability). These move independently, and
the distinction is the substance of most comparisons here — a parameter choice can leave
efficiency untouched while collapsing reliability.

!!! note "Ranking depends on the solver set"
    Ratios are normalised by the per-problem best, so adding or removing a solver can reorder
    the curves (Gould & Scott, 2016). Report pairwise profiles against a fixed baseline
    alongside the full comparison.

!!! warning "Problems that cost nothing"
    A problem already critical at `x₀` returns `:first_order` with zero iterations. A cost of
    `0` is treated as a failure here, while `success_table` counts the same run as solved, so
    one such problem lowers every reliability asymptote *and* every median — and neither
    number says anything about a radius rule. Screen these out of the problem list before
    building the cost matrix, and log the names dropped. Several CUTEst `*NE` variants have a
    null objective, so `∇f ≡ 0` everywhere and every mechanism reports zero iterations.

[`data_profile`](@ref) is the complementary instrument: it reports the fraction solved within
a budget measured in units of `n+1` evaluations, is not normalised by the best solver, and so
does not suffer that set-dependence.

Export to `pgfplots` with [`profile_to_pgfplots`](@ref):

```julia
open(io -> profile_to_pgfplots(io, τ, prof, labels), "prof.tex", "w")
```

```latex
\begin{axis}[xmode=log, xlabel={$\tau$}, ylabel={$\pi_s(\tau)$}]
  \input{prof.tex}
\end{axis}
```

## Run matrix

```julia
problems = [() -> ADNLPModel(f, x0, name = nm) for (nm, f, x0) in my_list]
configs  = sweep_configs("ζ", [0.01, 0.1, 1.0, 10.0], ζ -> RDFO(ζ = ζ))

T, S = run_matrix(problems, configs; cost = :iter)
summarise(T, [c.label for c in configs])
```

Problems are **thunks**, not models. This matters for CUTEst, where each `CUTEstModel` holds
an open handle to a compiled SIF problem and only one may be live at a time; `run_matrix`
finalises each before opening the next.

Only runs ending `:first_order` count as solved. A solver that stops early with a large
gradient has not solved the problem, whatever its iteration count.

Rules and quasi-Newton models carry mutable state, so a sweep passes **factories** rather
than instances: `ζ -> RDFO(ζ = ζ)` above, not a vector of built rules. Reusing one instance
across problems makes the results depend on the order they were visited.

## The experiment suite

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark benchmark/experiments/run_all.jl
julia --project=benchmark benchmark/experiments/run_all.jl 1 3 5   # a subset
```

| experiment | tests |
|---|---|
| `exp1_comparison` | like-for-like comparison; performance + data profiles |
| `exp2_trajectories` | Δ, ‖g‖, ρ per iteration; ΣΔ and ΣΔ² by family |
| `exp3_zeta_sweep` | the ζ threshold: reliability against efficiency |
| `exp4_mu_sweep` | the μ_max threshold; Cauchy-point diagnostic |
| `exp5_inactivity` | fraction of late iterations with ‖s‖ = Δ |
| `exp6_interaction` | rule × model grid; additivity residuals |
| `exp7_convergence_rate` | local order, conditioned on inactivity |
| `exp8_single_problem` | one problem in detail: Δ, ‖g‖, ρ and the inactivity countdown |
| `exp9_second_order` | ‖g‖- vs τ-anchoring at a saddle; measure × subsolver grid |

Experiment 8 takes a problem name, so it can be pointed anywhere without editing the script:

```bash
julia --project=benchmark benchmark/experiments/exp8_single_problem.jl WOOD
TRR_PROBLEM=ROSENBR julia --project=benchmark benchmark/experiments/exp8_single_problem.jl
```

It looks in the analytic set first and falls through to `CUTEstModel`, and writes one figure
per quantity plus a stacked panel sharing an iteration axis. The countdown panel plots the
number of active iterations still ahead of each index, so the onset of inactivity can be read
against what Δ, `‖g‖` and ρ were doing at the time; a run whose staircase never reaches zero
is labelled `[never]`.

Each run writes a self-documenting archive:

```
benchmark/results/exp_2026-04-16_02-08-34_zeta_sweep/
├── experiment_config.toml     what was run
├── experiment_summary.md      what came out
├── figures/                   PDFs
├── tables/                    text tables and .tex profile coordinates
└── data/                      raw JLD2, one file per (problem, rule)
```

Recover a past configuration with `load_config(dir)`, or the most recent with
`latest_archive()`. An interrupted campaign resumes in place:

```bash
TRR_RESUME=benchmark/results/exp_2026-07-29_02-15-23_comparison \
  julia --project=benchmark benchmark/experiments/exp1_comparison.jl
```

`run_experiment` then skips any `(problem, configuration)` whose `.jld2` already exists, and
does not open a model at all when every configuration for that problem is cached — on CUTEst
the SIF decode dominates the cost of a resume. See `RESUME.md` for the details.

!!! warning "Experiment 7 conditions on inactivity"
    The convergence order is estimated only over the final run of *inactive* iterations.
    Including active ones mixes a linear phase with the asymptotic one and returns an order
    between the two — an artefact, not a finding. Configurations whose constraint never stops
    binding report `never` and no order; that is the result, not a gap in it.

## API

```@docs
performance_profile
data_profile
profile_to_pgfplots
run_matrix
summarise
TRConfig
sweep_configs
```
