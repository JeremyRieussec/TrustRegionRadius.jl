# TrustRegionRadius.jl

[![CI](https://github.com/JeremyRieussec/TrustRegionRadius.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JeremyRieussec/TrustRegionRadius.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://JeremyRieussec.github.io/TrustRegionRadius.jl/dev/)
[![codecov](https://codecov.io/gh/JeremyRieussec/TrustRegionRadius.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JeremyRieussec/TrustRegionRadius.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A testbed for **trust-region radius update mechanisms**, built on
[JuliaSmoothOptimizers](https://jso.dev) (`NLPModels`, `SolverCore`, `Krylov`,
`LinearOperators`).

Companion code for *A survey of trust-region radius update mechanisms*, Parts I–III.

## What it is for

Most trust-region codes fix one radius rule and vary everything else. This package does the
opposite: the radius rule is a first-class, swappable component, and so are the two things it
interacts with. Three orthogonal axes:

| axis | choices |
|---|---|
| **radius rule** | `RDelta`, `RStep`, `RDFO`, `RGrad`, `RGradCapped`, `RAdaptiveStep`, `RAdaptiveGrad`, `RAdaptiveFanYuan`, `RRTR`, `RRTRGrad` |
| **model Hessian** | `ExactHessian`, `LBFGSModel`, `SR1Model`, `ScaledIdentity`, `SPDTarget` |
| **subproblem solver** | `SteihaugCG`, `ExactMS`, `KrylovCG`, `KrylovCGLanczos` |

Every combination runs through one driver, so a comparison measures the axis you varied and
not the difference between two code bases.

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/JeremyRieussec/TrustRegionRadius.jl")
```

## Use

```julia
using TrustRegionRadius, ADNLPModels

nlp = ADNLPModel(x -> (1 - x[1])^2 + 100(x[2] - x[1]^2)^2, [-1.2, 1.0])

stats = tr_solve(nlp;
    rule      = RGrad(),
    model     = SR1Model(mem = 5),
    subsolver = SteihaugCG(),
    params    = TRParams(tol = 1e-8),
    trace     = true)

stats.status                                       # :first_order
stats.solver_specific[:delta_trajectory]           # Δ per iteration
count(stats.solver_specific[:active_trajectory])   # iterations with ‖s‖ = Δ
```

`TRResult` is `GenericExecutionStats`, so `SolverBenchmark` and the rest of the JSO ecosystem
accept it unchanged.

Start with [`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb), or the
[documentation](https://JeremyRieussec.github.io/TrustRegionRadius.jl/dev/).

## Adding a rule

One struct and two methods:

```julia
mutable struct MyRule <: RadiusRule
    up::Float64
    down::Float64
end

TrustRegionRadius.initial_radius(::MyRule, Δ₀, g_norm) = Δ₀

TrustRegionRadius.update_radius!(r::MyRule, Δ, ρ, η₁, η₂, s_norm, g_old, g_new) =
    ρ ≥ η₁ ? Δ * r.up : Δ * r.down
```

Then `tr_solve(nlp; rule = MyRule(2.0, 0.5))`. Add `reset_rule!` if the struct carries
mutable state, and `needs_retrospective(::MyRule) = true` to be driven by ρ̃ rather than ρ.

## Benchmarks

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark benchmark/experiments/run_all.jl
julia --project=benchmark benchmark/experiments/exp3_zeta_sweep.jl   # just one
```

Each run writes a self-documenting archive:

```
benchmark/results/exp_2026-04-16_02-08-34_zeta_sweep/
├── experiment_config.toml     what was run
├── experiment_summary.md      what came out
├── figures/  tables/  data/
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

## Two things worth knowing

**`RGrad` is uncapped; `RGradCapped` is not.** For `RGrad` the multiplier μ is exactly the
ratio Δ/‖g‖, and it grows geometrically past any threshold, so the trust-region constraint
eventually stops binding with no side condition. The capped variant supplies the bound the
asymptotic theory assumes, at the cost of needing `μ_max > κ̄ = 4/λ*_min` — a constant that
depends on the solution and so cannot be chosen in advance.

**The activity flag is the interesting observable.** Whether the constraint eventually stops
binding separates mechanisms whose first-order behaviour is indistinguishable: both drive
‖g‖ → 0. `trace = true` records it.

## Tests

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

## Citing

See [`CITATION.cff`](CITATION.cff).

## References

- Conn, Gould & Toint, *Trust-Region Methods*, SIAM 2000.
- Bastin, Malmedy, Mouffe, Toint & Tomanos, *A retrospective trust-region method*,
  Math. Prog. 123 (2010) 395–418.
- Fan, Pan & Song, *A retrospective trust region algorithm with trust region converging to
  zero*, J. Comput. Math. 34 (2016) 421–436.
- Dolan & Moré, *Benchmarking optimization software with performance profiles*,
  Math. Prog. 91 (2002) 201–213.
- Moré & Wild, *Benchmarking derivative-free optimization algorithms*, SIAM J. Optim. 20
  (2009) 172–191.
