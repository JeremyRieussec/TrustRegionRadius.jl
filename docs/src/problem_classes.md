# Problem classes

Three regimes, kept apart in the type system rather than in comments, because they
need three different solvers and admit three different sets of model Hessians and
sampling rules.

```
AbstractProblem
├── DeterministicProblem      f(x) evaluated exactly
└── SampledProblem            f(x) estimated from a sample
    ├── ExpectationProblem    f(x) = E_ξ[F(x,ξ)];  population unbounded
    └── FiniteSumProblem      f(x) = (1/M) Σ f_i(x);  M finite and known
        └── ScoredProblem     per-observation gradients available
            └── LikelihoodProblem   f = −(1/M) Σ ln p_i
                └── NLSProblem      f = (1/2M) Σ r_i²
```

| class | solver | oracle | cap on `N_k` | full batch? | truth? |
|---|---|---|---|---|---|
| deterministic | [`DeterministicTRSolver`](@ref) | any `AbstractNLPModel`, [`FullBatchNLP`](@ref) | — | always | always |
| expectation | [`ExpectationTRSolver`](@ref) | [`ExpectationNLP`](@ref) | `budget` | **never** | only if supplied |
| finite sum | [`FiniteSumTRSolver`](@ref) | [`FiniteSumNLP`](@ref) | `min(budget, M)` | reachable | always |

[`tr_solve`](@ref) dispatches on the oracle, so the regime is decided by what you
built rather than by a keyword you might forget.

## Why an expectation is not a finite sum with a large `M`

**The population cap.** An expectation has no largest sample, so `N_max` is a
*budget the user chooses*. A finite sum has exactly `M` terms, so `N_k ≤ M` is
imposed by the problem, and a user `N_max` there is either redundant (`≥ M`) or a
deliberate sub-population budget (`< M`) — two different intentions that should not
share a keyword. [`FiniteSumNLP`](@ref) therefore **rejects** a rule carrying an
`N_max` and takes `budget` explicitly:

```@example classes
using TrustRegionRadius, ADNLPModels, NLPModels, LinearAlgebra

base = ADNLPModel(x -> 0.5 * sum(abs2, x .- 1), [0.0, 0.0])
prob = PerturbedSum(base, 500; σg = 1.0, seed = 1)

try
    FiniteSumNLP(prob, RadiusProportional(N_max = 500))
catch e
    showerror(stdout, e)
end
```

```@example classes
eprob = PerturbedExpectation(base; σg = 1.0)
(FiniteSumNLP(prob, RadiusProportional())               isa FiniteSumNLP,   # cap is M
 FiniteSumNLP(prob, RadiusProportional(); budget = 500) isa FiniteSumNLP,   # deliberate
 ExpectationNLP(eprob, RadiusProportional(N_max = 500)) isa ExpectationNLP) # N_max IS the budget
```

**The full-batch limit.** At `N_k = M` a finite-sum iteration is *exactly*
deterministic: `ρ̂ = ρ`, the stopping test is the real one, and the accuracy
hypotheses of the stochastic theory are discharged rather than assumed. An
expectation has no such limit, so its hypotheses can never be discharged by
sampling harder. [`FullBatch`](@ref) makes the limit reachable, is rejected on an
expectation, and gives the sharpest regression test in the package:

```@example classes
det = tr_solve(FullBatchNLP(prob);              rule = RDelta())
fs  = tr_solve(FiniteSumNLP(prob, FullBatch()); rule = RDelta())
(det.iter, fs.iter, det.solution ≈ fs.solution)
```

That is the only place the sampled and exact code paths can be compared directly,
so it pins the resampling and re-evaluation logic against a reference that has
none.

**The truth.** A finite sum can always be evaluated exactly, at one pass, so
[`true_gradient`](@ref) is available for the whole class and `true_stop = true`
works. For an expectation it exists only when the construction supplies the mean —
[`PerturbedExpectation`](@ref) does, a general one does not — so [`has_truth`](@ref)
is a trait and `true_stop` on a problem without truth is an `ArgumentError`, not a
silent fallback to the batch test.

`:full_batch_trajectory` records which finite-sum iterations were exact. A run whose
tail is all `true` has entered the deterministic regime and its final `‖ĝ‖` is the
real one; a run whose tail is all `false` never did, and its status is a statement
about one batch.

## Deterministic is not "finite sum with `M = 1`"

[`DeterministicTRSolver`](@ref) has no `resample!`, no variance estimate, no sample
counters and no `sampling` keyword. Handing it a sampled oracle is a constructor
error, and so is `TRParams(true_stop = true)` — there the batch test *is* the true
test, so confirming it is meaningless rather than merely redundant.

[`FullBatchNLP`](@ref) is the bridge: deterministic, while
[`underlying_problem`](@ref) still reports the finite-sum problem underneath, so
`BHHHModel` over a likelihood remains legal. That is why it is a separate type
rather than a `FiniteSumNLP` with `FullBatch()`: the two are numerically identical
and structurally different, and the second still carries a sampling rule, a batch, a
variance estimate and an RNG that mean nothing.

## What the classes buy the model Hessians

[`required_problem`](@ref) declares the narrowest class a model is defined for, and
[`check_model_problem`](@ref) enforces it at solver construction:

| model | requires | because |
|---|---|---|
| [`BHHHModel`](@ref), [`BHHH2Model`](@ref) | `LikelihoodProblem` | the information identity is a statement about a negative log-likelihood |
| [`GaussNewtonModel`](@ref) | `NLSProblem` | it needs the residual Jacobian |
| everything else | `AbstractProblem` | — |

```@example classes
try
    tr_solve(ADNLPModel(x -> sum(abs2, x), [1.0, 2.0]); model = BHHHModel())
catch e
    showerror(stdout, e)
end
```

```@example classes
logistic = LogisticRegression(K = 3, M = 200, seed = 1)
tr_solve(FullBatchNLP(logistic); model = BHHHModel()).status
```

This matters because the failure it prevents is silent. Applied to a problem that
is not a likelihood, BHHH still produces a positive semidefinite matrix and the run
still converges; nothing in `ρ`, `‖g‖` or the radius trace reveals that the model is
approximating nothing in particular.

!!! note "`NLSProblem <: LikelihoodProblem` is deliberate"
    Least squares *is* maximum likelihood under i.i.d. Gaussian errors, and the
    score `rₙ∇rₙ` is the log-likelihood score up to `σ²`. So BHHH applies to it as
    well as Gauss–Newton, and the two can be compared on one problem — worth having,
    since they discard different terms and fail for different reasons: BHHH on
    *specification*, Gauss–Newton on *fit*.

Belonging to `LikelihoodProblem` asserts that `f` is a negative log-likelihood. It
does **not** assert correct specification, which is a property of the data that no
type can carry — [`information_identity_error`](@ref) is how to check that half.

## What the classes buy the sampling rules

- [`needs_scores`](@ref) against [`has_scores`](@ref): the inner-product family needs
  `Var(∇Fᵢᵀĝ)` and cannot estimate it from a batch gradient. Checked at oracle
  construction, so `InnerProductTest` on an unscored problem fails immediately rather
  than at iteration 1.
- [`requires_finite_population`](@ref): `true` only for [`FullBatch`](@ref).
- [`check_population_cap`](@ref): the `N_max` rule above.

## API

```@docs
AbstractProblem
DeterministicProblem
SampledProblem
ExpectationProblem
FiniteSumProblem
ScoredProblem
LikelihoodProblem
NLSProblem
problem_class
population
n_terms
has_scores
has_truth
full_batch
underlying_problem
required_problem
check_model_problem
check_rule_problem
check_population_cap
user_cap
DeterministicTRSolver
ExpectationTRSolver
FiniteSumTRSolver
FullBatchNLP
ExpectationNLP
FiniteSumNLP
SampledNLP
population_cap
```
