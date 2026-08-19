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
interacts with. Four orthogonal axes:

| axis | choices |
|---|---|
| **radius rule** | `RDelta`, `RStep`, `RDFO`, `RGrad`, `RGradCapped`, `RAdaptiveStep`, `RAdaptiveGrad`, `RRTR`, `RRTRGrad` |
| **model Hessian** | `ExactHessian`, `LBFGSModel`, `SR1Model`, `ScaledIdentity`, `SPDTarget`, `BHHHModel`, `BHHH2Model`, `GaussNewtonModel` |
| **subproblem solver** | `SteihaugCG`, `ExactMS`, `KrylovCG`, `KrylovCR`, `EigenPoint` |
| **sampling rule** | `FullBatch`, `FixedSample`, `RadiusProportional`, `NormTest`, `GeometricSample`, `InnerProductTest`, `OrthogonalityTest`, `AugmentedInnerProduct`, `SequentialEstimation`, `CertifiedDecrease` |

The nine radius rules are the first-order ones; `SecondOrder(inner)` wraps any
criticality-anchored rule, with the aliases `RGradTau`, `RGradCappedTau`, `RDFOTau`,
`RAdaptiveGradTau` and `RRTRGradTau`.

The sampling rule belongs to the *oracle*, not to `tr_solve`: build a `FiniteSumNLP`
or an `ExpectationNLP` around it and pass that. There is no `sampling` keyword.

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
struct MyRule <: RadiusRule
    γ1::Float64
    γ2::Float64
    γ3::Float64
    function MyRule(; γ1 = 0.25, γ2 = 0.5, γ3 = 2.0)
        check_factors(:MyRule; γ1 = γ1, γ2 = γ2, γ3 = γ3)
        new(γ1, γ2, γ3)
    end
end

function TrustRegionRadius.update_radius!(r::MyRule, Δ::Float64, ρ::Float64, ::Bool,
                                          η1::Float64, η2::Float64,
                                          ::Float64, ::Float64, ::Float64)
    ρ < η1  && return r.γ1 * Δ        # unsuccessful: the result MUST be < Δ
    ρ >= η2 && return r.γ3 * Δ
    return Δ
end

TrustRegionRadius.asymptotic_regime(::MyRule) = :bounded_below
```

Then `tr_solve(nlp; rule = MyRule())`. Add `reset_rule!` if the struct carries mutable
state, and `needs_retrospective(::MyRule) = true` to be driven by ρ̃ rather than ρ.

Three things the earlier version of this template got wrong, all of which stop it
dispatching or constructing:

- `update_radius!` takes **nine** arguments, with `accepted::Bool` third. An
  eight-argument method never dispatches and the rule is silently ignored.
- `check_factors` takes **ASCII** keywords `γ1, γ2, γ3`. The Unicode aliases `η₁, η₂,
  Δ₀` exist on `TRParams` but not on the rule validators.
- Do **not** define `initial_radius` unless your rule ignores `Δ0`. The fallback
  already returns `Δ0`, and an unannotated `initial_radius(::MyRule, Δ0, g_norm) = Δ0`
  is *ambiguous* with it rather than an override.

Every rule must strictly contract on an unsuccessful iteration. A rule that can return
`Δ` unchanged after a rejection re-solves an identical subproblem for ever, since a
rejected step leaves both the model and the iterate unchanged.

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
