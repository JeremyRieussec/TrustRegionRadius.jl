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
git show main:Manifest.toml > Manifest.toml
git show main:benchmark/Manifest.toml > benchmark/Manifest.toml
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark -e 'include("benchmark/initialisation.jl"); comparison()'
```

The manifests were committed after the campaigns, so the campaign commits lack them; the
two `git show` lines copy them in, and since those commits ignore `Manifest.toml` the tree
stays clean. Use `Pkg.instantiate()` alone: `Pkg.develop` or `Pkg.update` rewrites the
manifest, and on `main`, where it is tracked, leaves the tree dirty.

Each campaign writes a new archive under `benchmark/results/`. CUTEst problems are opened
through `CUTEst.jl`, which decodes each SIF file and compiles it with `gfortran`: install it
first (`apt install gfortran` on Debian or Ubuntu), since the `CUTEst.jl` of the manifest
(1.4.0) fails to load without it. Check that the log reports
`CUTEst: 196 problems selected`, the number in every archive; a different SIF collection can
select a different set.

## A quick check

Rerunning everything takes about a day (below). `quick_check.jl` checks the package against
the comparison archive in a fraction of that. From the repository root, on `main`:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
# unzip exp_2026-08-29_02-28-45_comparison.zip from the Zenodo record, then
julia --project=benchmark benchmark/quick_check.jl exp_2026-08-29_02-28-45_comparison
```

It re-solves every problem of the comparison on which all archived runs ended inside the
time budget and in under one second on the campaign machine: 176 of the 196 problems,
1056 runs, under the six rules that `RULES` still holds (`RRTR` and `RAdaptiveStep` were
commented out of it after the campaign). The status, the iteration count and the evaluation
counts of each run must agree exactly with the archive; the script lists every run that
does not, says how many differ in status, and exits with 1 if any differs. A second
argument restricts it to the cheapest problems. It also prints the ratio of solve times,
this machine to the campaign machine, over the runs that took at least 0.1 s there; that
ratio scales the durations below.

On a laptop, an Intel Core i5-13500H under WSL with Julia 1.12.6 and gfortran 13.3, the
check took about three minutes, compilation included, and gave a ratio of 2.0. 1050 of the
1056 runs agreed exactly. The six that did not are all on HUMPS, a two-variable problem on
which the rules creep for up to ten thousand iterations, and the status agreed in all six.
They differ by the same counts whether the environment is instantiated from the manifests
of the campaign or resolved afresh, so the package versions are not the cause. What still
differs from the campaign machine is the processor, the Julia patch release (1.12.6 against
1.12.1) and the gfortran that compiles the SIF file; a rounding difference from any of them
is enough to move a run that creeps for thousands of iterations.

## How long it takes

All seven campaigns ran on one machine, an Intel Core i9-13900, on a single thread. The
sum of the archived solve times:

| campaign | runs | solve time | ended on the 120 s budget |
|---|---|---|---|
| comparison | 1568 | 2.5 h | 67 |
| ζ sweep | 1176 | 2.1 h | 52 |
| inactivity | 1176 | 2.0 h | 51 |
| interaction | 3136 | 2.1 h | 34 |
| convergence order | 1568 | 2.9 h | 75 |
| μ̄ sweep | 1372 | 2.5 h | 63 |
| second order | 4704 | 7.7 h | 198 |
| **total** | **14 700** | **21.9 h** | **540** |

Decoding and compiling the problems add little on that machine: the comparison started
2.61 h before the next campaign, against 2.53 h of solve time. Elsewhere, multiply by the
ratio `quick_check.jl` prints: on the laptop above, a full rerun would take some 43 hours.

## What depends on the machine

A run is unsolved when it exhausts 10⁴ iterations or 120 s of wall clock, and only the second
test depends on the machine. On a slower machine, runs that came close to the budget reach
it too, so the counts of solved problems can change. The iteration and evaluation counts of
the runs that end inside the budget do not, rounding on long runs such as those of HUMPS
above apart, and they are what to compare.

## Dependency versions

The repository ships the manifests of the campaign environment: `Manifest.toml` at the
root and `benchmark/Manifest.toml`, both resolved under Julia 1.12.1. The archives record a
content hash of the root `Project.toml` and `Manifest.toml`, `f2aa70c00c0418a4` and
`b5e9c5e29c995cb9` in all seven, and the committed files give exactly those values, so the
root manifest is the one the campaigns ran with. The `benchmark` manifest, the environment
actually active during the runs, was not hashed; it agrees with the root manifest on all
21 packages they share. To check the root hash, under Julia 1.12:

```julia
string(hash(read("Manifest.toml", String)); base = 16)   # b5e9c5e29c995cb9
```

## Reading an archive

Each zip unpacks to the directory the campaign wrote: `experiment_config.toml` (what was
run, including every rule, model and subsolver with its parameters),
`experiment_summary.md`, `figures/`, `tables/`, and `data/` with one JLD2 file per
problem and configuration. `load_config(dir)` reads the configuration back; see
`docs/src/benchmarking.md`.
