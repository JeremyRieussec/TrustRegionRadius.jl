# Outer-product Hessians

Three `ModelHessian`s that approximate `∇²f` from first derivatives alone:

| model | approximation | requires | assumption it rests on |
|---|---|---|---|
| [`BHHHModel`](@ref) | `B = (1/N) Σ sₙsₙᵀ` | `LikelihoodProblem` | information identity: correct specification, at the true parameters |
| [`BHHH2Model`](@ref) | `W = (1/N) Σ (sₙ−ḡ)(sₙ−ḡ)ᵀ` | `LikelihoodProblem` | the same |
| [`GaussNewtonModel`](@ref) | `B = (1/N) JᵀJ` | `NLSProblem` | small residuals, or nearly linear `rₙ` |

All three are positive semidefinite by construction, and all three obtain that
property by discarding a term that vanishes only under their assumption.

The **requires** column is enforced, not documented: [`required_problem`](@ref)
declares it and the solver constructor checks it, so `BHHHModel` over anything that
is not a negative log-likelihood is an `ArgumentError` naming both types.

```@example likelihood
using TrustRegionRadius, ADNLPModels

try
    tr_solve(ADNLPModel(x -> sum(abs2, x), [1.0, 2.0]); model = BHHHModel())
catch e
    showerror(stdout, e)
end
```

```@example likelihood
logistic = LogisticRegression(K = 3, M = 200, seed = 1)
tr_solve(FullBatchNLP(logistic); model = BHHHModel()).status
```

That check exists because the failure it prevents is silent: applied to the wrong
problem BHHH still produces a positive semidefinite matrix and the run still
converges, and nothing in `ρ`, `‖g‖` or the radius trace reveals that the model is
approximating nothing in particular. See [Problem classes](problem_classes.md).

!!! note "`NLSProblem <: LikelihoodProblem`"
    Least squares *is* maximum likelihood under i.i.d. Gaussian errors, and the score
    `rₙ∇rₙ` is the log-likelihood score up to `σ²`. So BHHH applies to a least-squares
    problem as well as Gauss–Newton, and the two can be compared on one — which is
    the comparison worth making. The factor `σ²` matters to BHHH, and
    [`LinearRegression`](@ref) is the least-squares problem that carries it.

Belonging to `LikelihoodProblem` asserts that `f` is a negative log-likelihood. It
does **not** assert correct specification, which is a property of the data that no
type can carry — that half is what [`information_identity_error`](@ref) measures.

## The information identity

For a correctly specified model at the true parameters, `V = −H`: the population
covariance of the scores equals the negative average population Hessian. Minimising
the average negative log-likelihood, this says `B → ∇²f`.

**Both conditions are load-bearing.** [`information_identity_error`](@ref) measures
the gap rather than assuming it away. On logistic regression with `K = 5`:

| `M` | at `β*` | `×√M` | at `β* + 1.5` |
|---|---|---|---|
| 500 | 0.204 | 4.56 | 2.28 |
| 2 000 | 0.071 | 3.20 | 2.69 |
| 32 000 | 0.017 | 2.99 | 2.69 |
| 128 000 | 0.008 | 2.71 | 2.71 |

At `β*` the error decays like `M^{-1/2}` — the `×√M` column is flat. At a displaced
point it is `O(1)` and does not move. That is the whole of BHHH's reputation for
poor early steps and good late ones: away from the optimum `B` is not approximating
the Hessian badly, it is not approximating it at all.

`B` and `W` differ by exactly the rank-one `ḡḡᵀ`, so they coincide at a stationary
point and can differ sharply away from one.

## Two links on one design

[`Probit`](@ref) and [`LogisticRegression`](@ref) draw the same design matrix and the
same `β*` from one seed, so the identity can be measured for both links on the same
data. This block is executed when the page is built.

```@example likelihood
using Printf

for M in (2_000, 32_000), (name, make) in (("logit", LogisticRegression), ("probit", Probit))
    p = make(K = 5, M = M, seed = 1)
    β = β_true(p)
    @printf("%-6s  M = %6d   at β*: %.3f   at β* + 1.5: %6.3f\n", name, M,
            information_identity_error(p, β).B_err,
            information_identity_error(p, β .+ 1.5).B_err)
end
```

At `β*` the error shrinks with `M` under both links, and at the displaced point it keeps
its size under both. The lower values printed there for the larger sample are one
sample differing from another, as the section on the third likelihood measures. The
displaced error is several times larger for the probit, and the reason is
in the score. The logit residual `p − y` is bounded by 1. The probit score carries the
inverse Mills ratio `φ/Φ`, which grows linearly in the index of an observation the
current `β` gets wrong, and `B` squares it, while the weight of the exact Hessian never
exceeds 1. Far from the solution BHHH therefore overstates the curvature of a probit by
a factor that grows with the distance.

The origin is a special point for both links. At `β = 0` every index is zero, so each
score outer product equals the per-observation Hessian, `¼ xxᵀ` for the logit and
`(2/π) xxᵀ` for the probit, and `B` is the exact Hessian whatever the data:

```@example likelihood
[information_identity_error(make(K = 5, M = 2_000, seed = 1), zeros(5)).B_err
 for make in (LogisticRegression, Probit)]
```

Runs start at zero by default. The first BHHH model of such a run is therefore exact,
which says nothing about the models that follow.

The two links differ in a second way. The logit link is canonical, so its
per-observation Hessian `p(1−p) xxᵀ` does not involve the outcome. The probit Hessian
does: it is the *observed* information, random in the outcomes, and the Fisher
information is its expectation over `y` given `x`. On a probit the identity ties three
matrices together in expectation, the outer product of the scores, the observed
information and the Fisher information, where the logit has two.

`Probit(noise = σ)` generates the outcomes from coefficients that vary across
observations. The generating process is then a heteroscedastic probit, the fitted model
is misspecified, and the identity fails at `β*` itself at every sample size. The
minimiser of `f` is then far from `β*`. At that minimiser the identity fails by much
less, and the gap still does not vanish as `M` grows.

## A third likelihood, with a closed-form reference

[`LinearRegression`](@ref) is the Gaussian likelihood of a linear model whose error
standard deviation `σ` is known. One seed gives it the design matrix and the `β*` of the
two binary models, and at `σ = 1` its outcome is the latent variable whose sign the
probit observes. The loop of the previous section then runs over three likelihoods.

```@example likelihood
for M in (2_000, 32_000),
    (name, make) in (("logit", LogisticRegression), ("probit", Probit),
                     ("linear", LinearRegression))
    p = make(K = 5, M = M, seed = 1)
    β = β_true(p)
    @printf("%-6s  M = %6d   at β*: %.3f   at β* + 1.5: %6.3f\n", name, M,
            information_identity_error(p, β).B_err,
            information_identity_error(p, β .+ 1.5).B_err)
end
```

At `β*` the error is divided by about four under all three, which is the rate
`M^{-1/2}` for a sample sixteen times larger. At the displaced point it keeps its size.
The values printed there for the larger sample are lower by 3 to 11 per cent, and that
is one sample differing from another and not a decay: over twenty seeds the larger
sample gave the smaller value 12, 11 and 9 times for the three likelihoods.

The displaced error is largest for the regression. Its score is linear in the
residual, so the outer product grows with the square of the distance to `β*`, while the
weight of its Hessian is the constant `1/σ²`.

What the regression adds is a reference known before anything is run. Its Hessian is the
constant `XᵀX/(Mσ²)`, and with `rₙ = (xₙᵀβ − yₙ)/σ` the standardised residual,

```math
B - H = \frac1M \sum_n \bigl(r_n^2 - 1\bigr)\, \frac{x_n x_n^\top}{\sigma^2}
```

at every `β` and for any data. The identity is therefore a statement about the second
moment of the standardised residuals, and normality plays no part in it. At `β*` the gap
has mean zero given the design, and under Gaussian errors its root-mean-square size is a
function of the design alone. At `β* + δ` its mean is `(1/M) Σ (xₙᵀδ)² xₙxₙᵀ/σ⁴`, which
no sample size removes. The next block computes both from `X` and sets them beside the
measured error.

```@example likelihood
using LinearAlgebra

p = LinearRegression(K = 5, M = 2_000, seed = 1)
β = β_true(p); δ = fill(1.5, 5)
H = p.X' * p.X ./ p.M
at_truth  = sqrt(2 * sum(sum(abs2, view(p.X, i, :))^2 for i in 1:p.M)) / norm(p.X' * p.X)
displaced = norm(p.X' * (((p.X * δ) .^ 2) .* p.X) ./ p.M) / norm(H)
@printf("at β*:        measured %6.3f   root mean square from the design %6.3f\n",
        information_identity_error(p, β).B_err, at_truth)
@printf("at β* + 1.5:  measured %6.3f   mean from the design             %6.3f\n",
        information_identity_error(p, β .+ δ).B_err, displaced)
```

At `β*` the two numbers agree in size and not in value, as one draw and a root mean
square should. At the displaced point they agree to within one per cent.

The origin is not the special point it is for the two links. At `β = 0` the residual is
the outcome itself, and the error is of the order of `‖β*‖²/σ²`. The block prints the
two.

```@example likelihood
(information_identity_error(p, zeros(5)).B_err, sum(abs2, β_true(p)) / p.σ^2)
```

A run started at zero therefore meets a BHHH model that is exact under the two binary
links and far from exact under the regression.

### The scale

The Hessian divides by `σ²` and the outer product of the scores by `σ⁴`, so the two can
agree for one value of `σ` only. [`linear_least_squares`](@ref) minimises the same sum
of squares with the scale fixed at 1, whatever generated the data. Its default errors
have standard deviation 0.1, and the block below poses its data a second time with that
value carried.

```@example likelihood
ls = linear_least_squares()
re = LinearRegression(ls.t, ls.y; σ = 0.1, β_star = x_true(ls))
@printf("σ fixed at 1:    %.3f\n", information_identity_error(ls, x_true(ls)).B_err)
@printf("σ = 0.1 carried: %.3f\n", information_identity_error(re, β_true(re)).B_err)
```

With the scale fixed at 1 the outer product at the solution is the error variance times
the Hessian, 0.01 times here, and the error of 0.99 is the distance of that factor
from 1. No sample size removes it. With `σ` carried the same data leave the sampling
error alone.

`LinearRegression(noise = ν)` generates the outcomes from coefficients that vary across
observations, as the `noise` keyword of [`Probit`](@ref) does. The errors then have variance
`σ² + ν²‖xₙ‖²` while the model holds it at `σ²`, and the identity fails at `β*` at every
sample size. The errors keep mean zero given `x`, so `β*` remains the minimiser of the
population objective, which under the probit it does not.

Part of that failure is a wrong scale, the same thing as the previous block, and with
six coefficients it is most of it. The next block poses the noisy data a second time,
with the scale of their own residuals.

```@example likelihood
noisy = LinearRegression(K = 6, M = 32_000, seed = 3, noise = 0.5)
s = sqrt(sum(abs2, noisy.y - noisy.X * (noisy.X \ noisy.y)) / noisy.M)
reposed = LinearRegression(noisy.X, noisy.y; σ = s, β_star = β_true(noisy), noise = 0.5)
@printf("posed with σ = 1:     %.3f\n", information_identity_error(noisy, β_true(noisy)).B_err)
@printf("posed with σ = %.3f: %.3f\n", s, information_identity_error(reposed, β_true(noisy)).B_err)
```

What is left in the second line is the heteroscedasticity, the part of the gap that
comes from `‖xₙ‖` varying across observations and that no single scale removes. With
`K = 1` and an intercept every `xₙ` is 1, nothing is left, and the noisy data are a
correctly specified regression with a larger `σ`.

### What the three share

The logit, the probit and the regression are single-index likelihoods on one design
matrix. The loss of an observation is a function `ℓ` of the index `z = xᵀβ` and of the
outcome, so the score is `ℓ′ x`, the Hessian is `ℓ″ xxᵀ`, and

```math
B - H = \frac1N \sum_n \bigl(\ell_n'^2 - \ell_n''\bigr)\, x_n x_n^\top .
```

The information identity is then the scalar statement `E[ℓ′² − ℓ″ | x] = 0` at `β*`.

| likelihood | `ℓ′` | `ℓ″` | `ℓ″` depends on |
|---|---|---|---|
| logit | `p − y` | `p(1 − p)` | the index |
| probit | `−q λ(qz)` | `λ(qz)(λ(qz) + qz)` | the index and the outcome |
| regression | `(z − y)/σ²` | `1/σ²` | neither |

Here `p` is the logistic function of `z`, `q = 2y − 1` and `λ = φ/Φ`. The last column is
what separates them. The regression is the control, with a curvature that nothing
moves. The logit has a curvature that moves with `β`, and the probit one that moves with
the outcomes as well.

## The blind spot

`B ⪰ 0` always. The model therefore **cannot report negative curvature**, so a run
over it converges contentedly to a saddle while every first-order diagnostic looks
healthy. This is the same failure [`LBFGSModel`](@ref) has, reached by a different
route, and the caution in [Model Hessians](models.md) applies unchanged: no radius
mechanism repairs a model that misreports curvature.

!!! warning "`SecondOrder` over an outer-product model certifies nothing"
    `λ_min(B) ≥ 0` identically, so `τ = max{‖g‖, −λ_min} ≡ ‖g‖` and a
    `:second_order` status means only that the gradient is small. When the
    second-order question is the question, use `ExactHessian`.

## Neural networks: the justification is absent, not weakened

[`MLPClassifier`](@ref) fits a one-hidden-layer softmax network with per-observation
scores, so BHHH applies mechanically. But the model is misspecified and there are no
true parameters, so the identity fails at every sample size and the error does not
shrink with `N`. `B` remains a usable positive semidefinite preconditioner; calling
it an approximate Hessian there is unjustified.

### Scale

A dense `n × n` Hessian is out of reach; the `n × N` score matrix is not. This
asymmetry is why the outer-product models work at this scale and `ExactHessian`
does not:

| configuration | `n` | dense `∇²f` | scores, `N = 256` |
|---|---|---|---|
| MNIST 784→16→10 | 12 730 | 1.21 GiB | 24.9 MiB |
| MNIST 196→16→10 (2× downsample) | 3 322 | 0.08 GiB | 6.5 MiB |
| CIFAR-100 3072→32→100 | 101 636 | 76.96 GiB | 198.5 MiB |
| CIFAR-100 256→32→100 (16×16 grey) | 11 524 | 0.99 GiB | 22.5 MiB |

Use `hessian_op`, which returns an [`OuterProductOperator`](@ref) above
`dense_max` and computes `Bv = (1/N) S(Sᵀv)` without forming `B`. Pair it with
[`SteihaugCG`](@ref); `ExactMS` needs a dense eigendecomposition and is unavailable
here, which is itself a finding — the model axis constrains the subsolver axis.

`ridge > 0` matters when `N < n`: `B` is a sum of `N` rank-one terms, so a small
batch makes it singular regardless of the problem. Under sampling, the ridge and the
batch size are not independent choices.

## Least squares

[`LeastSquares`](@ref) is an `NLSProblem`, so it runs deterministically through
[`FullBatchNLP`](@ref), stochastically through [`FiniteSumNLP`](@ref) with any sampling
rule, and reports per-observation scores — meaning `GaussNewtonModel`, `BHHHModel` and
`ExactHessian` can all be compared on one problem.

- [`linear_least_squares`](@ref): `∇²rₙ = 0`, so **Gauss–Newton is exact**, not merely
  accurate, whatever the residual size. The control. [`LinearRegression`](@ref) is
  the same problem read as a likelihood, with the scale of the errors carried.
- [`exponential_fit`](@ref): sum-of-exponentials fitting with a `misfit` knob that adds
  a component the model cannot represent, moving the problem continuously from
  small-residual to large-residual.

[`gauss_newton_error`](@ref) is the counterpart of [`information_identity_error`](@ref):
it measures the discarded `(1/N)Σ rₙ∇²rₙ` and reports it beside the RMS residual, which
is what controls it. So the two outer-product models fail for different reasons — BHHH on
*specification*, Gauss–Newton on *fit* — and both keep positive semidefiniteness, and
with it the inability to report negative curvature, after their justification has gone.

## API

```@docs
BHHHModel
BHHH2Model
GaussNewtonModel
OuterProductOperator
information_identity_error
scores
loss_terms
score_matrix
LikelihoodNLP
residuals
jacobian
LogisticRegression
batch_hess(::LogisticRegression, ::Any, ::Any)
Probit
scores(::Probit, ::Any, ::Any)
batch_hess(::Probit, ::Any, ::Any)
LinearRegression
batch_hess(::LinearRegression, ::Any, ::Any)
MultinomialLogit
scores(::MultinomialLogit, ::Any, ::Any)
batch_hess(::MultinomialLogit, ::Any, ::Any)
MLPClassifier
scores(::MLPClassifier, ::Any, ::Any)
batch_grad!(::MLPClassifier, ::Any, ::Any, ::Any)
batch_hess(::MLPClassifier, ::Any, ::Any)
LeastSquares
batch_hess(::LeastSquares, ::Any, ::Any)
linear_least_squares
exponential_fit
gauss_newton_error
x_true
init_params
accuracy
β_true
```
