# Benchmarking

Two layers: profile functions in the package (pure functions of a cost matrix), and an
experiment harness in `benchmark/` that produces self-documenting archives.

## Profiles

```@example benchmarking
using TrustRegionRadius

T = [ 12.0  18.0  Inf ;      # rows: problems, columns: solvers
      30.0  25.0  41.0 ]     # Inf marks a failure

τ, prof = performance_profile(T)
(size(prof), prof[1, :], prof[end, :])
```

`prof[1, s]` is the fraction of problems on which solver `s` is fastest, and
`prof[end, s]` the fraction it solves at all. Solver 3 fails one of the two problems,
so its reliability asymptote is `0.5`.

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

!!! note "Not executed"
    The next three blocks are templates: they refer to a problem list and a sampled
    problem you supply. They are shown for reference and are not run when the
    documentation is built.

```julia
problems = [() -> ADNLPModel(f, x0, name = nm) for (nm, f, x0) in my_list]
configs  = sweep_configs("ζ", [0.01, 0.1, 1.0, 10.0], ζ -> RDFO(ζ = ζ))

T, S = run_matrix(problems, configs; cost = :iter)
summarise(T, [c.label for c in configs])
```

Problems are **thunks**, not models. This matters for CUTEst, where each `CUTEstModel` holds
an open handle to a compiled SIF problem and only one may be live at a time; `run_matrix`
finalises each before opening the next.

Runs ending `:first_order` or `:second_order` count as solved. A solver that stops early
with a large gradient has not solved the problem, whatever its iteration count — but a run
that reached a *certified second-order* point has solved it more strongly than one that
stopped at `‖g‖ ≤ tol`, so counting `:second_order` as a failure would zero the reliability
of exactly the columns doing best.

`cost = :samples` fills the matrix with cumulative term evaluations, and is the measure to
use whenever the columns differ in sampling rule — `:iter` is proportional to work only
under `FixedSample`. It raises on a deterministic problem rather than returning zero.

The sampling rule lives on the **oracle**, not on `TRConfig`, so a sweep over sampling rules
varies the problem thunks:

```julia
problems = [() -> FiniteSumNLP(prob, r) for r in (FixedSample(64), RadiusProportional())]
T, _ = run_matrix(problems, [TRConfig("R-delta"; rule = RDelta())]; cost = :samples)
```

Rules and quasi-Newton models carry mutable state, so a sweep passes **factories** rather
than instances: `ζ -> RDFO(ζ = ζ)` above, not a vector of built rules. Reusing one instance
across problems makes the results depend on the order they were visited.

## The experiment suite

!!! warning "`benchmark/` is not part of the package"
    `success_table`, `run_experiment`, `load_config`, `latest_archive` and
    `ExperimentArchive` live in `benchmark/`, not in `src/`. They are **not** available
    from `using TrustRegionRadius`; load them with `include("benchmark/initialisation.jl")`
    from the `benchmark` project. Only [`performance_profile`](@ref),
    [`data_profile`](@ref), [`profile_to_pgfplots`](@ref), [`run_matrix`](@ref),
    [`summarise`](@ref), [`TRConfig`](@ref) and [`sweep_configs`](@ref) are exported by
    the package itself.


```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark -e 'include("benchmark/initialisation.jl"); comparison()'
```

`benchmark/initialisation.jl` loads the package, the harness, the archive layer and
`config.jl` once, in that order, and then includes every experiment file. Each
experiment defines one entry point, so call the one you want by name. The experiment
files themselves contain no `using` and no `include`: re-including `config.jl` would
re-evaluate its `const`s, which Julia rejects.

!!! note "Not executed"
    The shell commands on this page are shown for reference. They are not run when the
    documentation is built, and the benchmark suite is not a dependency of the docs.

| entry point | experiment |
|---|---|
| `comparison()` | 1 |
| `trajectories()` | 2 |
| `zeta_sweep()` | 3 |
| `mu_sweep()` | 4 |
| `inactivity()` | 5 |
| `interaction()` | 6 |
| `CVRate()` | 7 |
| `second_order()` | 9 |
| `bhhh_study()` | 11 |
| `sampling_examples()` | 12 |
| `flat_well_experiment()` | 13 |

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
| `exp11_bhhh` | BHHH and BHHH2 against the exact Hessian on a likelihood |
| `exp12_sampling_examples` | the sampling rules on the three worked examples |
| `exp13_flat_well` | the radius thresholds measured by bisection on ε |

There is no `exp10`: the suite jumps from `exp9_second_order` to `exp11_bhhh`.

Experiment 8 is absent from the entry-point table because `initialisation.jl` does not
include it. The `include` line is commented out there: the filenames from `exp2` to
`exp8` were once shifted by one against their contents, and the file now named
`exp8_single_problem.jl` is a collection of helpers with no entry point of its own, so
the experiment-8 script is genuinely missing rather than merely unlisted.

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
