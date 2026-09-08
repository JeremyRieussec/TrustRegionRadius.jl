# Sampling rules

The sampling rule is the **fourth axis**, alongside the radius mechanism, the model
Hessian and the subproblem solver. It applies in the two sampled
[problem classes](problem_classes.md) and not in the deterministic one, so which
solver runs is decided by the oracle you build:

```@example stochastic
using TrustRegionRadius, ADNLPModels, NLPModels, LinearAlgebra

# finite sum: N_k ≤ M, imposed by the problem
base_nlp = ADNLPModel(x -> 0.5 * sum(abs2, x .- 1), [0.0, 0.0])
prob = PerturbedSum(base_nlp, 4_000; σg = 1.0, seed = 1)   # mean is exactly base_nlp
nlp  = FiniteSumNLP(prob, RadiusProportional(κ_g = 1.0))

stats = tr_solve(nlp; rule = RDelta(), model = ExactHessian(),
                 subsolver = ExactMS(), trace = true)

(stats.solver_specific[:samples_total],                   # the cost measure that matters
 count(stats.solver_specific[:full_batch_trajectory]),    # iterations that were exact
 norm(true_gradient(prob, stats.solution)))               # score on the truth, not on ĝ
```

```@example stochastic
# expectation: no M, so the cap is a budget you choose
eprob = PerturbedExpectation(base_nlp; σg = 1.0)
enlp  = ExpectationNLP(eprob, RadiusProportional(); budget = 50_000)
est = tr_solve(enlp; rule = RDelta(), subsolver = ExactMS(), trace = true)
(est.status, est.iter, est.solver_specific[:samples_total])
```

Both oracles satisfy the ordinary NLP interface, so every mechanism, model and
subsolver runs over them unchanged. What differs is the cap on `N_k` and whether a
full batch exists at all — see [Problem classes](problem_classes.md).

## The cost inversion

The accuracy requirements for convergence (Chen, Menickelly & Scheinberg 2018) are
stated in terms of the radius: gradient error `O(Δ_k)`, function error `O(Δ_k²)`.
For a Monte Carlo estimator that means

```math
N_k^{\text{grad}} = \Theta(\Delta_k^{-2}), \qquad
N_k^{\text{obj}}  = \Theta(\Delta_k^{-4}),
```

so the total work is `Σ_k Δ_k^{-2}` — the **reciprocal** of the `Σ_k Δ_k²` tables
of Part II. The deterministic ranking of the mechanisms therefore inverts. On the
running example of Part II, over 60 iterations:

| rule | `Σ Δ_k²` | `Σ Δ_k^{-2}` | `Σ Δ_k^{-4}` |
|---|---|---|---|
| `RDelta` | 4.4e+35 | **1.3e+00** | **1.1e+00** |
| `RStep` | 2.4e+02 | 1.6e+01 | 4.7e+00 |
| `RDFO` | **1.4e+00** | 2.0e+05 | 2.0e+09 |
| `RGrad` | **1.4e+00** | 7.5e+04 | 1.7e+08 |

The mechanisms that the summability criteria favour are the most expensive to run
stochastically, by five to nine orders of magnitude.

!!! warning "Iteration counts assume `FixedSample`"
    Every profile in Parts I–II counts iterations, which is proportional to work
    only when `N_k` is constant. Under any adaptive rule the same runs order
    differently. Report `:samples_total`.

## Where the cap comes from

A rule computes a *requirement*; it does not decide the *budget*. The two were
previously conflated in an `N_max` field, which meant the same keyword expressed two
different things depending on the class:

| class | population | `N_max` on the rule |
|---|---|---|
| expectation | unbounded | **is** your budget — allowed and meaningful |
| finite sum | `M` | **rejected**: the cap belongs to the problem |

```@example stochastic
try
    FiniteSumNLP(prob, RadiusProportional(N_max = 500))
catch e
    showerror(stdout, e)
end
```

```@example stochastic
(FiniteSumNLP(prob, RadiusProportional(); budget = 500) isa FiniteSumNLP,
 ExpectationNLP(eprob, RadiusProportional(N_max = 500)) isa ExpectationNLP)
```

The effective cap is `min(sample_cap(rule), budget, population(prob))`, and hitting it
is counted in `:sample_cap_hits`. A run spent at the cap is no longer meeting the
accuracy requirement the convergence theory assumes, so it is not a run the theory
describes.

`N_min` stays on the rule: a floor is a property of the estimator, not of the budget —
you cannot estimate a variance from one sample.

## The full-batch reference

On a finite sum, [`FullBatch`](@ref) gives `N_k = M` at every iteration, which is the
deterministic run exactly: `ρ̂ = ρ`, the stopping test is the real one, and the accuracy
hypotheses are discharged rather than assumed. It is the reference every other sampling
rule should be scored against.

It is **rejected on an expectation**, where there is no full batch and the hypotheses
can never be discharged by sampling harder. That refusal is the type system carrying the
substantive difference between the two classes.

## The feedback loop

[`couples_to_radius`](@ref) marks the rules whose `N_k` depends on `Δ_k`. When it
does, the radius mechanism and the sampling rule stop being independent axes: for
a criticality-anchored rule, `ĝ_k` sets `Δ_k` sets `N_k` sets the accuracy of
`ĝ_{k+1}`. A noisy gradient shrinks the radius, which demands more samples, which
are spent recovering the accuracy the shrinking presumed. `RDelta` has no such
loop. [`NormTest`](@ref) is the control: it also drives `N_k → ∞`, but through
`‖ĝ_k‖ → 0` rather than through the radius, so the coupling can be switched off
without switching off the growth.

## The rules

| rule | `N_k` from | reads `Δ_k`? |
|---|---|---|
| [`FullBatch`](@ref) | `M` — finite sums only | no |
| [`FixedSample`](@ref) | nothing | no |
| [`RadiusProportional`](@ref) | `(σ/(κΔ_k))²` | **yes** |
| [`NormTest`](@ref) | `σ_g²/(θ²‖ĝ‖²)` | no |
| [`InnerProductTest`](@ref) | `Var(∇Fᵢᵀĝ)/(θ²‖ĝ‖⁴)` | no |
| [`OrthogonalityTest`](@ref) | `E‖proj⊥∇Fᵢ‖²/(ν²‖ĝ‖²)` | no |
| [`AugmentedInnerProduct`](@ref) | the maximum of the previous two | no |
| [`GeometricSample`](@ref) | `N₀·rate^k`, fixed in advance | no |
| [`SequentialEstimation`](@ref) | `2z²σ_f²/(κ²·pred²)` | **yes**, through `pred` |
| [`CertifiedDecrease`](@ref) | `z_p²σ̂_N²/δ̂_N²`, paired | **yes**, through the step |

### Certifying the decrease instead of the gradient

[`CertifiedDecrease`](@ref) is the one rule that sizes the sample against the achieved
decrease rather than against the gradient, using **paired differences** under common
random numbers: with the same realisation at both ends of the step,

```math
D_i = F(x_k, \\xi_i) - F(x_k + s_k, \\xi_i)
```

and `N_{k+1}` is the ceiling of `z_p² σ̂_N² / δ̂_N²`, where `δ̂_N` and `σ̂_N²` are the
sample mean and variance of the `D_i`.

Holding the realisation fixed at both ends is the whole point: it makes `D_i → 0`
pathwise as `s_k → 0`, so the variance of `D` is `O(‖s_k‖²)`. Two independent batches
give `O(1)` instead, and the rule is then unusable near a solution.

The machinery this needs is separate from the rest of the sampling interface, because
only this rule uses it: [`needs_paired`](@ref) declares that a rule wants paired
differences, [`supports_paired`](@ref) declares that a problem can supply them,
[`obs_objective`](@ref) evaluates one term at one realisation,
[`paired_decrease_stats`](@ref) forms the two statistics in one pass, and
[`record_paired!`](@ref) hands them back to the rule. A run under this rule records
`:paired_decrease_trajectory` and `:paired_variance_trajectory`.

### Why the inner-product tests are cheaper

Split the gradient variance along and across the estimated direction:

```math
\sigma_g^2 = \frac{\mathrm{ip}^2}{\|\hat g\|^2} + \mathrm{orth}^2,
\qquad
\mathrm{ip}^2 = \operatorname{Var}(\nabla F_i^\top \hat g),
\quad
\mathrm{orth}^2 = \mathbb E\bigl\|\text{proj}_\perp \nabla F_i\bigr\|^2 .
```

The norm test bounds the sum. But only the component **along** `ĝ` decides whether the
sampled gradient still points downhill — error orthogonal to it rotates the direction
without threatening descent. So bounding the sum is stricter than the descent property
requires, and [`InnerProductTest`](@ref) reaches the same guarantee at a smaller sample.
[`OrthogonalityTest`](@ref) caps the rotation, which the inner-product test alone does
not; [`AugmentedInnerProduct`](@ref) applies both.

[`batch_stats`](@ref) computes all four moments in one pass over the score matrix, and
the identity above holds to rounding — so the tests are comparable on the same batch.

### Sequential estimation

[`SequentialEstimation`](@ref) sizes the sample against the **predicted decrease**
rather than the gradient: enough samples that the noise in the estimated decrease is
small beside the decrease the model claims,

```math
N_k \;\ge\; \frac{2 z_{\alpha/2}^2\,\hat\sigma_f^2}{\kappa^2\,\mathrm{pred}_{k-1}^2}.
```

The accuracy demanded therefore tracks *progress* rather than *criticality*. It is
monotone by default, because a batch that shrinks makes `f̂` jump for reasons unrelated
to the step and ρ̂ then measures the change of estimator rather than of objective; the
`growth` factor bounds the rise per iteration, measured from `N_start` on the first
adaptive call, so one small `pred` cannot demand the whole population at once.

## Common random numbers

The batch is drawn once per iteration and held fixed, so `f̂(x_k)`, `f̂(x_k + s_k)`
and any retrospective evaluation use the same realisations. This is not an
optimisation. With independent draws the error in `f̂(x_k) − f̂(x_k + s_k)` does not
shrink with the step, so ρ̂ is pure noise once `‖s_k‖` falls below the sampling
error, and every mechanism stalls for a reason external to it:

| `‖s‖` | shared batch | independent batches |
|---|---|---|
| `1e-1` | 1.3e-02 | 1.8e-01 |
| `1e-3` | 1.3e-04 | 1.8e-01 |

The solver re-evaluates `f` and `g` at the incumbent after every resample, so the
numerator and denominator of ρ̂ always come from one batch.

## Score on the truth

`‖ĝ_k‖ ≤ tol` is a statement about one batch, and a mechanism that shrinks the
radius fast enough will meet it on noise alone. [`PerturbedSum`](@ref) has mean exactly
the base model, so [`true_gradient`](@ref) is available at every iterate; use it for
every reported number, and set `TRParams(true_stop = true)` when the *status* itself
needs to mean something.

Every finite sum has a truth, at one full pass. An expectation has one only when the
construction supplies it — [`PerturbedExpectation`](@ref) does, a general one does not —
so [`has_truth`](@ref) is a trait and `true_stop` on a problem without truth is an
`ArgumentError` rather than a silent fallback to the batch test.

## API

```@docs
FiniteSum
PerturbedSum
PerturbedExpectation
GaussianDraw
batch_obj
batch_grad!
batch_hess
grad_variance
obj_variance
SamplingRule
FullBatch
SamplingState
SampleStats
batch_stats
FixedSample
RadiusProportional
NormTest
InnerProductTest
OrthogonalityTest
AugmentedInnerProduct
GeometricSample
SequentialEstimation
CertifiedDecrease
SmoothedSize
record_prediction!
reset_sampling_rule!
couples_to_radius
needs_scores
requires_finite_population
sample_cap
grad_sample_size
obj_sample_size
resample!
update_variances!
samples_used
reset_sampling!
draw_batch
sample_schemes
sample_scheme
true_objective
true_gradient
needs_paired
record_paired!
paired_decrease_stats
paired_variance_kind
paired_op_variance
supports_paired
obs_objective
confirm_gradient_norm!
grad_standard_error
```
