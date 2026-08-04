# Outer-product Hessians

Three `ModelHessian`s that approximate `∇²f` from first derivatives alone:

| model | approximation | assumption it rests on |
|---|---|---|
| [`BHHHModel`](@ref) | `B = (1/N) Σ sₙsₙᵀ` | information identity: correct specification, at the true parameters |
| [`BHHH2Model`](@ref) | `W = (1/N) Σ (sₙ−ḡ)(sₙ−ḡ)ᵀ` | the same |
| [`GaussNewtonModel`](@ref) | `B = (1/N) JᵀJ` | small residuals, or nearly linear `rₙ` |

All three are positive semidefinite by construction, and all three obtain that
property by discarding a term that vanishes only under their assumption.

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

[`LeastSquares`](@ref) is a `ScoredProblem`, so it runs deterministically through
`LikelihoodNLP`, stochastically through `SampledNLP` with any sampling rule, and reports
per-observation scores — meaning `GaussNewtonModel`, `BHHHModel` and `ExactHessian` can
all be compared on one problem.

- [`linear_least_squares`](@ref): `∇²rₙ = 0`, so **Gauss–Newton is exact**, not merely
  accurate, whatever the residual size. The control.
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
NLSProblem
residuals
jacobian
LogisticRegression
MLPClassifier
LeastSquares
linear_least_squares
exponential_fit
gauss_newton_error
x_true
init_params
accuracy
β_true
```
