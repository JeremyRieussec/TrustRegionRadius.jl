# TrustRegionRadius.jl

A testbed for **trust-region radius update mechanisms**, built on the
[JuliaSmoothOptimizers](https://jso.dev) stack.

Companion code for *A survey of trust-region radius update mechanisms*, Parts I–III.

## Why this package exists

Most trust-region codes fix one radius rule and vary everything else. This package does the
opposite: the radius rule is a first-class, swappable component, and so are the three things it
interacts with. Four orthogonal axes:

| axis | choices |
|---|---|
| [radius rule](rules.md) | `RDelta`, `RStep`, `RDFO`, `RGrad`, `RGradCapped`, `RAdaptiveStep`, `RAdaptiveGrad`, `RAdaptiveFanYuan`, `RRTR`, `RRTRGrad` |
| [model Hessian](models.md) | `ExactHessian`, `LBFGSModel`, `SR1Model`, `ScaledIdentity`, `SPDTarget`, `BHHHModel`, `BHHH2Model`, `GaussNewtonModel` |
| [subproblem solver](subsolvers.md) | `SteihaugCG`, `ExactMS`, `KrylovCG`, `KrylovCGLanczos`, `EigenPoint` |
| [sampling rule](stochastic.md) | `FullBatch`, `FixedSample`, `RadiusProportional`, `NormTest`, `GeometricSample`, `InnerProductTest`, `OrthogonalityTest`, `AugmentedInnerProduct`, `SequentialEstimation` |

Every combination runs through one driver *per regime*, so a comparison measures the axis you
varied rather than the difference between two code bases. That single-driver design is the main
methodological control behind the numerical results in Part III.

## Three regimes, three solvers

The fourth axis exists only where there is something to sample, so the driver is chosen by the
[problem class](problem_classes.md) rather than by a keyword:

| class | solver | cap on `N_k` | full batch? |
|---|---|---|---|
| deterministic | `DeterministicTRSolver` | — | always |
| expectation, `f = E[F(x,ξ)]` | `ExpectationTRSolver` | your `budget` | **never** |
| finite sum, `f = (1/M) Σ fᵢ` | `FiniteSumTRSolver` | `M`, imposed | reachable |

`tr_solve` dispatches for you. Keeping the three apart is not tidiness: a deterministic run
should not carry a resampling branch, an expectation has no `M` and no exact iterate to fall
back on, and a finite sum's `N_max` is a property of the problem rather than a knob. The
[problem classes](problem_classes.md) page has the details, including why `BHHHModel` is
rejected on anything that is not a likelihood.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/USER/TrustRegionRadius.jl")
```

## Thirty seconds

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
count(stats.solver_specific[:active_trajectory])   # iterations with ‖s‖ = Δ
```

Continue with [Getting started](quickstart.md).

## Three results worth knowing before you choose a rule

**`RGrad` is uncapped; `RGradCapped` is not.** For `RGrad` the multiplier ``\mu_k`` is exactly
the ratio ``\Delta_k/\|g_k\|``, and it grows geometrically past any threshold, so the
trust-region constraint eventually stops binding with no side condition. `RGradCapped`
supplies the bound the asymptotic theory assumes, at the cost of requiring
``\bar\mu > \bar\kappa = 4/\lambda^*_{\min}`` — a constant that depends on the solution and
so cannot be chosen in advance. Below that threshold the constraint binds forever and the
method degrades to linear convergence while every first-order diagnostic looks healthy.

**The activity flag is the observable that matters.** Whether the constraint eventually stops
binding separates mechanisms whose first-order behaviour is indistinguishable: both groups
drive ``\|g_k\| \to 0``. Pass `trace = true` and read `:active_trajectory`; for *when* it
stopped rather than *whether*, count the active iterations still ahead of each index, as in
[Getting started](quickstart.md#When-the-constraint-stopped-binding).

**Acceptance and scaling answer different questions.** ``\eta`` decides whether a step is
worth taking; ``\eta_1`` and ``\eta_2`` decide only whether the region should shrink or grow,
and are the only two any rule sees. Coupling them, as the classical statement does, hides the
fact that ``\eta = 0`` is admissible and that ``\rho_k \in [\eta, \eta_1)`` accepts a step
while still contracting the radius. See [Thresholds and factors](thresholds.md).

## Interface changes

Acceptance is decoupled from scaling, `update_radius!` takes an `accepted::Bool` third
argument, and every rule obeys ``0 < \gamma_1 \le \gamma_2 < 1 < \gamma_3``. Because ``\eta``
defaults to ``\eta_1``, existing `TRParams` calls are unaffected; four rules had their
factors renumbered and will now throw rather than be silently reinterpreted.
[Thresholds and factors](thresholds.md) explains both conventions.

The solver was then split in three by problem class, `TRSolver` becoming
`DeterministicTRSolver`, and `SampledNLP` split into `ExpectationNLP` and `FiniteSumNLP`.
`TRParams` keywords are ASCII (`η1`, `η2`, `Δ0`) with the subscript spellings kept as aliases.
`MIGRATION.md` in the repository root lists what to edit.

## Citing

See `CITATION.cff`, or cite Part I of the survey directly.
