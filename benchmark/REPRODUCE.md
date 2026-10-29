# Reproducing the numerical study

*A numerical study of radius update rules in trust-region algorithms*
([arXiv:2610.26134](https://arxiv.org/abs/2610.26134)) rests on seven campaigns. Each wrote one
archive, and the seven are deposited at
[doi:10.5281/zenodo.22655838](https://doi.org/10.5281/zenodo.22655838), one zip per campaign.

## Which archive feeds what

| in the article | archive | script | entry point | commit |
|---|---|---|---|---|
| like-for-like comparison, its profiles | `exp_2026-08-29_02-28-45_comparison` | `exp1_comparison.jl` | `comparison()` | `5b959ce` |
| ζ sweep of R-DFO | `exp_2026-08-29_05-05-13_zeta_sweep` | `exp3_zeta_sweep.jl` | `zeta_sweep()` | `5b959ce` |
| tail activity, eventual inactivity | `exp_2026-08-29_07-14-47_inactivity` | `exp5_inactivity.jl` | `inactivity()` | `5245c30` |
| rule × model grid, additivity | `exp_2026-08-29_09-37-36_interaction` | `exp6_interaction.jl` | `interaction()` | `13d26d2` |
| local convergence order | `exp_2026-08-29_11-49-49_convergence_rate` | `exp7_convergence_rate.jl` | `CVRate()` | `5b959ce` |
| μ̄ sweep of capped R-grad | `exp_2026-08-29_18-21-22_mu_sweep` | `exp4_mu_sweep.jl` | `mu_sweep()` | `a9e4926` |
| second-order rules on CUTEst | `exp_2026-08-31_10-38-22_second_order_profiles` | `exp17_second_order_profiles.jl` | `second_order_profiles()` | `2969ac7` |

The commit is the one each archive's `experiment_config.toml` records, with `git_dirty = false`
in all seven. The inactivity and interaction archives keep their August names, but their
configuration was written on 5 September, at the commits shown. All seven ran under Julia 1.12.1 on one
thread, with `package_version = "0.1.0"`.

## The library across the five commits

`src/` is identical at `5b959ce`, `a9e4926` and `2969ac7`, and identical at `5245c30`,
`13d26d2` and the tag `v0.1.0`. Between the two groups it changes only in the stochastic
sampling code, the likelihood models, the exported names and docstrings; the deterministic
path the seven campaigns run is the same. The drivers under `benchmark/` do differ, so to
rerun a campaign as it was run, check out the commit its archive records.

## Running a campaign

From the repository root:

```bash
git checkout 5b959ce        # the commit of the campaign, from the table
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark -e 'include("benchmark/initialisation.jl"); comparison()'
```

Each campaign writes a new archive under `benchmark/results/`. CUTEst problems are opened
through `CUTEst.jl`; without it, `default_problems()` falls back to the eight analytic
problems with only a warning, and `benchmark/run_campaign.jl` refuses to start.

## Dependency versions

The repository ships no `Manifest.toml`, so `Pkg.instantiate()` resolves the dependencies
within the `[compat]` bounds of `Project.toml`, and the versions can differ from those of the
runs. The archives record a content hash of the root `Project.toml` and `Manifest.toml`
(`f2aa70c00c0418a4` and `b5e9c5e29c995cb9`, the same in all seven); the manifest of the
`benchmark` environment, the one active during the runs, was not hashed.

## Reading an archive

Each zip unpacks to the directory the campaign wrote: `experiment_config.toml` (what was
run, including every rule, model and subsolver with its parameters),
`experiment_summary.md`, `figures/`, `tables/`, and `data/` with one JLD2 file per
problem and configuration. `load_config(dir)` reads the configuration back; see
`docs/src/benchmarking.md`.
