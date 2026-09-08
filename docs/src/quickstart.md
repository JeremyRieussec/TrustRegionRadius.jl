# Getting started

## Installing

```julia
using Pkg
Pkg.add(url = "https://github.com/USER/TrustRegionRadius.jl")
```

!!! note "Not executed"
    This block is shown for reference and is not run when the documentation is
    built.

For development, clone and `Pkg.develop(path = ".")`.

## A first solve

```@example quickstart
using TrustRegionRadius, ADNLPModels, NLPModels

rosen(x) = (1 - x[1])^2 + 100(x[2] - x[1]^2)^2
nlp = ADNLPModel(rosen, [-1.2, 1.0])

stats = tr_solve(nlp; rule = RDelta())
(stats.status, stats.solution, stats.iter)
```

`tr_solve` returns a [`TRResult`](@ref), which is JSO's `GenericExecutionStats`, so
`SolverBenchmark` and the rest of the ecosystem accept it unchanged. Evaluation counts live
on the model — `neval_obj(nlp)`, `neval_grad(nlp)`, `neval_hprod(nlp)` — not on the result.

## Choosing the axes

```@example quickstart
stats = tr_solve(nlp;
    rule      = RGrad(μ = 1.0),      # how Δ moves
    model     = SR1Model(mem = 5),   # what curvature the model reports
    subsolver = SteihaugCG(),        # how the subproblem is solved
    params    = TRParams(η = 0.1, η1 = 0.1, η2 = 0.9, Δ0 = 1.0, tol = 1e-8))
(stats.status, stats.iter)
```

Defaults are `RDelta()`, `ExactHessian()`, `SteihaugCG()`. The threshold keywords are
ASCII (`η1`, `η2`, `Δ0`); the subscript spellings `η₁`, `η₂`, `Δ₀` are accepted as
aliases and readable as properties either way.

The fourth axis, the sampling rule, belongs to the *oracle* rather than to `tr_solve`,
because it decides what the oracle returns:

```@example quickstart
prob = LogisticRegression(K = 5, M = 2_000)

exact   = tr_solve(FullBatchNLP(prob);                       model = BHHHModel())
sampled = tr_solve(FiniteSumNLP(prob, RadiusProportional()); model = BHHHModel())
(exact.status, exact.iter, sampled.status, sampled.iter)
```

`tr_solve` picks the solver from the oracle: [`DeterministicTRSolver`](@ref),
[`FiniteSumTRSolver`](@ref) or [`ExpectationTRSolver`](@ref). See
[Problem classes](problem_classes.md) for what each one does differently and why the
three are kept apart.

!!! note "Hold `TRParams` fixed across mechanisms"
    In any comparison the thresholds and factors must be identical for every rule. Tuning
    them per rule measures tuning effort rather than algorithmic merit and makes a
    performance profile uninterpretable. The acceptance threshold `η` is the exception worth
    varying deliberately: it belongs to the *algorithm* rather than to any one mechanism, so
    changing it moves every column at once.

## The three thresholds

`η` decides whether a step is accepted; `η₁` and `η₂` decide only how the radius is scaled,
and are the only two a rule ever sees:

```@example quickstart
p1 = TRParams(η1 = 0.1, η2 = 0.9)           # η defaults to η1: the classical algorithm
p2 = TRParams(η = 0.0, η1 = 0.1, η2 = 0.9)  # accept every step with positive pred. reduction
(p1.η, p2.η)
```

They must satisfy `0 ≤ η ≤ η1 ≤ η2 < 1`, checked in the constructor. Setting `η < η₁` opens a
band where a step is accepted while the radius still contracts. [Thresholds and
factors](thresholds.md) covers the consequences, and the matching convention
`0 < γ₁ ≤ γ₂ < 1 < γ₃` on the scaling factors.

## Tracing

`trace = true` attaches per-iteration trajectories to `stats.solver_specific`:

```@example quickstart
stats = tr_solve(nlp; rule = RGrad(), trace = true)
ss = stats.solver_specific

# Δₖ, ‖gₖ‖, f(xₖ) are state trajectories, of length k+1;
# ρₖ, ‖sₖ‖ and the two Bool records are per-iteration, of length k.
[k => length(ss[k]) for k in (:delta_trajectory, :grad_trajectory, :obj_trajectory,
                              :ratio_trajectory, :step_trajectory,
                              :active_trajectory, :accepted_trajectory)]
```

The one-entry difference is the alignment convention, and it is set out in full on the
[Diagnostics](diagnostics.md) page.

A sampled run adds `:grad_sample_trajectory`, `:obj_sample_trajectory`,
`:samples_total`, `:sample_cap_hits` and — where a truth exists —
`:true_grad_trajectory`. A finite-sum run adds `:full_batch_trajectory`, marking the
iterations at which `Nₖ = M` and the step was therefore exact.

The first three are one entry longer than the rest, since they have a value before the first
iteration; align on the tail when plotting them together.

Two are worth recording even when nothing else is. `:active_trajectory` decides whether the
constraint eventually stops binding: two mechanisms can have identical iteration counts and
identical first-order behaviour while one keeps the constraint permanently active and the
other does not, and nothing else in a standard diagnostic distinguishes them.
`:accepted_trajectory` cannot be reconstructed from `:ratio_trajectory` and `η₁` once
`η < η₁`, so it is the only record of which iterations belong to ``\mathcal{S}``.

### When the constraint stopped binding

The tail fraction answers "is it still binding at the end?" but not "when did it stop?". For
that, count the active iterations still ahead of each index:

[`inactivity_index`](@ref) answers it directly. It returns the first iteration after
which the constraint never binds again, counted from zero, or `nothing` if it was still
binding at the end.

```@example quickstart
(inactivity_index(stats), active_fraction(stats; tail = 0.1))
```

A `nothing` there means the run never entered the regime where the local rate is
available. If you want the staircase itself for a plot, count the active iterations
still ahead of each index:

```@example quickstart
a = ss[:active_trajectory]
K = length(a)
R = [count(@view a[k+1:end]) for k in 0:K-1]     # Rₖ, a staircase down to zero
(first(R), last(R))
```

`R` starts at the total number of active iterations, steps down by one at each active
iteration, and is flat on inactive ones. It reaches zero at the last active iteration `k*`,
after which the step is the unconstrained model minimiser. The range stops at `K-1` on
purpose: `R_K = 0` for every run, so including it would make every curve reach zero and erase
the distinction.

Because the run stops at `‖g‖ ≤ tol` rather than at infinity, a positive tail is evidence of
inactivity *over the iterations observed*, not a proof of eventual inactivity — and a tail of
zero may mean only that the budget ran out first. Experiment 8 in `benchmark/` plots this
against Δ, `‖g‖` and ρ on a shared axis.

## Statuses

| status | meaning |
|---|---|
| `:first_order` | `‖g‖ ≤ tol` — the only status counted as solved in a profile |
| `:max_iter` | iteration budget exhausted |
| `:max_time` | wall-clock budget exhausted |
| `:stalled` | the step fell below the level at which `f(x) − f(x+s)` carries information, so ρ became noise |
| `:exception` | the model could not be built at the current iterate |
| `:user` | stopped by a callback |

`:stalled` deserves a word. Once the radius collapses far enough, the subtraction
`f(xₖ) − f(xₖ + sₖ)` is pure rounding, ρ becomes noise, every step is rejected, and the
radius contracts forever. This is a genuine numerical limitation of the criticality-anchored
rules in floating point rather than a bug, and it is reported separately so that it is not
silently folded into "failure".

!!! note "`:first_order` at iteration 0"
    A problem already critical at `x₀` returns `:first_order` with `iter == 0`. That is
    correct but carries no information about any mechanism, and it reaches the two reporting
    paths inconsistently — a profile treats a cost of zero as a failure while a success table
    counts it as solved. Screen such problems out of a test set before building a cost matrix;
    several CUTEst `*NE` variants have a null objective and behave this way on every rule.

## Adding your own rule

One struct and two methods:

```@example quickstart
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

tr_solve(nlp; rule = MyRule(), params = TRParams(tol = 1e-8)).status
```

Two things in that template are easy to get wrong, and both were wrong in an earlier
version of this page.

`check_factors` takes **ASCII** keywords, `γ1, γ2, γ3`. The Unicode subscripts
`γ₁, γ₂, γ₃` are accepted as aliases on [`TRParams`](@ref) (`η₁, η₂, Δ₀`) but **not**
by the rule validators, so `check_factors(:MyRule; γ₁ = 0.25)` raises
`MethodError: no method matching check_factors(::Symbol; γ₁::Float64)`.

There is deliberately **no `initial_radius` method** above. The fallback
[`initial_radius`](@ref)`(::RadiusRule, Δ0::Float64, ::Float64)` already returns `Δ0`,
and adding an unannotated `initial_radius(::MyRule, Δ0, g_norm) = Δ0` makes the two
ambiguous rather than overriding one:

```
MethodError: initial_radius(::MyRule, ::Float64, ::Float64) is ambiguous.
```

Define it only when your rule ignores `Δ0`, as the `μ`-scaled rules do, and annotate
the argument types when you do.

Then `tr_solve(nlp; rule = MyRule())`. `update_radius!` is exported, so extending it needs the
qualified name; `using` brings it into scope without making it extensible. Arguments the rule
ignores are written as bare type annotations.

Add [`reset_rule!`](@ref) if the struct carries mutable state, and
`needs_retrospective(::MyRule) = true` if it should be driven by ρ̃ rather than ρ — the solver
then computes the retrospective ratio, at the cost of one extra Hessian-vector product per
accepted iteration. Add a [`validate_thresholds`](@ref) method if the rule cannot accept every
admissible `(η, η₁, η₂)`.

### The three obligations

**Contract on unsuccessful iterations.** If `ρ < η₁` can leave `Δ` unchanged, the rule falls
outside weak admissibility, the convergence theorem of Part I no longer covers it, and — since
a rejected step changes neither the model nor the iterate — the solver re-solves an identical
subproblem until the budget runs out. `test/test_thresholds.jl` checks this for every rule in
the package at five values of ρ below `η`; add new rules to it.

**Return a strictly positive radius.** A rule anchored to `‖sₖ‖` can collapse to zero. That is
what a `Δmin` field is for.

**Declare the regime.** [`asymptotic_regime`](@ref) and, where they apply,
[`needs_retrospective`](@ref) and [`is_criticality_anchored`](@ref) are what the experiment
scripts group by.

### The arguments in order

```julia
update_radius!(rule, Δ, ρ, accepted, η1, η2, s_norm, g_norm_old, g_norm_new)
```

`accepted` is third. It is passed rather than recomputed because a rule cannot derive it:
acceptance is decided by `η`, which the rule never receives, and a retrospective rule is
handed ρ̃ in the `ρ` slot and so never sees ρ at all.

Note the two gradient-norm arguments: `g_norm_old` is ‖gₖ‖ *before* the accept/reject
decision and `g_norm_new` is ‖gₖ₊₁‖ *after* it. Rules anchored to criticality at the current
iterate (`RDFO`) use the former; rules that set the next radius from the next gradient
(`RGrad` and relatives) use the latter. Getting this backwards is the easiest mistake to make
here, and it shifts every threshold silently.
