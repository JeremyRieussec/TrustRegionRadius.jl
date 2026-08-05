# =============================================================================
# src/Likelihood/likelihood.jl
#
# Outer-product approximations to the Hessian, and the score access they need.
#
#   BHHHModel          B = (1/N) Σ sₙ sₙᵀ            requires LikelihoodProblem
#   BHHH2Model         W = (1/N) Σ (sₙ−ḡ)(sₙ−ḡ)ᵀ    requires LikelihoodProblem
#   GaussNewtonModel   B = (1/N) Jᵀ J                requires NLSProblem
#
# ---------------------------------------------------------------------------
# Why the requirement is a type
#
# BHHH rests on the information identity `V = −H`, which is a statement about a
# *negative log-likelihood*: for a correctly specified model at the true
# parameters, the population covariance of the scores equals the negative average
# population Hessian, so `B → ∇²f`.
#
# Applied to anything else the arithmetic still runs and produces a positive
# semidefinite matrix — which is exactly the failure mode worth preventing, because
# nothing in ρ, ‖g‖ or the radius trace reveals that the model is approximating
# nothing in particular. `required_problem(::BHHHModel) === LikelihoodProblem` makes
# it a constructor error naming both types.
#
# `NLSProblem <: LikelihoodProblem` deliberately: least squares is maximum
# likelihood under i.i.d. Gaussian errors, so BHHH applies to it as well as
# Gauss-Newton and the two can be compared on one problem. They discard *different*
# terms — BHHH fails on specification, Gauss-Newton on fit — and both keep positive
# semidefiniteness, and with it the inability to report negative curvature, after
# their justification has gone.
# =============================================================================

"""
    score_matrix(nlp, x) -> Matrix

The `n × N` matrix of per-observation gradients at `x`, which the outer-product
models consume.

Defined for [`FullBatchNLP`](@ref) (all terms) and for the two sampled oracles over
a [`ScoredProblem`](@ref) (the current batch — so under sampling, BHHH is formed
from the same realisations as the gradient, which is what keeps `B` and `g`
consistent within an iteration).
"""
function score_matrix end

score_matrix(m::FullBatchNLP, x) = scores(m.prob, x, m.all)
score_matrix(m::SampledNLP, x)   = scores(m.prob, x, m.batch_g)
score_matrix(nlp, ::Any) = throw(ArgumentError(
    "score_matrix: $(typeof(nlp)) does not expose per-observation scores. Wrap a " *
    "ScoredProblem in FullBatchNLP or FiniteSumNLP, or use ExactHessian."))

# -----------------------------------------------------------------------------
# Outer-product models
# -----------------------------------------------------------------------------

"""
    BHHHModel(; ridge = 0.0, dense_max = 5_000)

`B = (1/N) Σₙ sₙ sₙᵀ`, the average outer product of the per-observation scores.

Two properties, and the trade between them is the whole story:

- **Cheap.** The scores are already computed for the gradient, so `B` costs one
  rank-`N` product and no second derivatives.
- **Positive semidefinite by construction.** Every step is a descent direction,
  even where the true Hessian is indefinite.

The second is a guarantee and a blind spot at once: `B ⪰ 0` always, so the model can
never report negative curvature, and a run over it converges contentedly to a saddle
while every first-order diagnostic looks healthy. Declared through
[`reports_negative_curvature`](@ref); the solver warns when it meets `SecondOrder` or
`tol_H`.

**Requires a [`LikelihoodProblem`](@ref)** — see the file header. Use
[`information_identity_error`](@ref) to measure how far the identity is from holding
on the problem at hand rather than assuming it.

`ridge > 0` matters when `N < n`: `B` is a sum of `N` rank-one terms, so a small
batch makes it singular regardless of the problem, and under sampling the ridge and
the batch size are not independent choices. It is also the right place for an L2
penalty on the problem, whose Hessian is `λI` — folding `λθ` into each score instead
gives a rank-one `λ²θθᵀ`.
"""
struct BHHHModel <: ModelHessian
    ridge::Float64
    dense_max::Int
    function BHHHModel(; ridge::Real = 0.0, dense_max::Int = 5_000)
        ridge >= 0 || throw(ArgumentError("BHHHModel: need ridge ≥ 0"))
        new(float(ridge), dense_max)
    end
end

"""
    BHHH2Model(; ridge = 0.0, dense_max = 5_000)

`W = (1/N) Σₙ (sₙ − ḡ)(sₙ − ḡ)ᵀ`: the sample **covariance** of the scores rather than
their average outer product.

`B = W + ḡḡᵀ`, so the two coincide exactly at a stationary point and differ by a
rank-one term that is largest early in a run — precisely where the identity does not
hold and neither matrix approximates the Hessian. Which is better there is an
empirical question this package can answer, which is the reason to carry both.

Requires a [`LikelihoodProblem`](@ref).
"""
struct BHHH2Model <: ModelHessian
    ridge::Float64
    dense_max::Int
    function BHHH2Model(; ridge::Real = 0.0, dense_max::Int = 5_000)
        ridge >= 0 || throw(ArgumentError("BHHH2Model: need ridge ≥ 0"))
        new(float(ridge), dense_max)
    end
end

"""
    GaussNewtonModel(; ridge = 0.0, dense_max = 5_000)

`B = (1/N) JᵀJ` for `f(x) = (1/2N) Σ rₙ(x)²`, whose exact Hessian is
`(1/N)(JᵀJ + Σ rₙ ∇²rₙ)`.

Gauss-Newton drops the second term. The approximation is good exactly when that term
is small — small residuals at the solution, or nearly linear `rₙ` — and can be
arbitrarily bad otherwise, which is the large-residual case where it loses its
superlinear rate. [`gauss_newton_error`](@ref) measures the discarded term beside the
RMS residual that controls it.

**Requires an [`NLSProblem`](@ref)**: the residual Jacobian is what it needs and
nothing above that type has one.
"""
struct GaussNewtonModel <: ModelHessian
    ridge::Float64
    dense_max::Int
    function GaussNewtonModel(; ridge::Real = 0.0, dense_max::Int = 5_000)
        ridge >= 0 || throw(ArgumentError("GaussNewtonModel: need ridge ≥ 0"))
        new(float(ridge), dense_max)
    end
end

reports_negative_curvature(::Union{BHHHModel, BHHH2Model, GaussNewtonModel}) = false

required_problem(::BHHHModel)        = LikelihoodProblem
required_problem(::BHHH2Model)       = LikelihoodProblem
required_problem(::GaussNewtonModel) = NLSProblem

reset_model!(::Union{BHHHModel, BHHH2Model, GaussNewtonModel}, ::Int) = nothing
update_model!(::Union{BHHHModel, BHHH2Model, GaussNewtonModel}, s, y) = nothing

"""
    _outer_product(S, centred, ridge) -> Symmetric

`(1/N) A Aᵀ + ridge·I` with `A = S` or `A = S − ḡ1ᵀ`.

Takes the score matrix as an argument rather than fetching it, so callers evaluate it
once: `_op` previously called `score_matrix` and then handed off to `dense_hessian`,
which called it again — on `MLPClassifier` that is one backward pass per observation,
twice.
"""
function _outer_product(S::AbstractMatrix, centred::Bool, ridge::Real)
    n, N = size(S)
    ḡ = vec(sum(S; dims = 2)) ./ N
    A = centred ? (S .- ḡ) : S
    B = (A * A') ./ N
    ridge > 0 && (B .+= ridge .* I(n))
    return Symmetric((B .+ B') ./ 2)
end

dense_hessian(m::BHHHModel, nlp, x)  = _outer_product(score_matrix(nlp, x), false, m.ridge)
dense_hessian(m::BHHH2Model, nlp, x) = _outer_product(score_matrix(nlp, x), true,  m.ridge)

"""
    OuterProductOperator(S, ḡ, centred, ridge)

Matrix-free `B·v` for the outer-product models: `Bv = (1/N) S(Sᵀv) + ridge·v`, two
matrix–vector products against the `n × N` score matrix and no `n × n` array
anywhere.

This is what makes the models usable at neural-network scale. Forming `B` densely
costs `n²` — 25 450 parameters is 648 million entries — while `S` costs `n·N` and
stays in memory for a batch of a few hundred. Pair it with [`SteihaugCG`](@ref).

`S` is `AbstractMatrix{T}` and `eltype` is defined. Both matter: the field was
`Matrix{T}`, so an `Adjoint` (which NLS scores used to be) failed to match the
constructor, and `Krylov.cg` queries `eltype`, so `KrylovCG` over a large
`BHHHModel` could not run at all.
"""
struct OuterProductOperator{T, M <: AbstractMatrix{T}}
    S::M
    ḡ::Vector{T}
    centred::Bool
    ridge::T
end

OuterProductOperator(S::AbstractMatrix{T}, ḡ::AbstractVector, centred::Bool,
                     ridge::Real) where {T} =
    OuterProductOperator{T, typeof(S)}(S, Vector{T}(ḡ), centred, T(ridge))

function Base.:*(op::OuterProductOperator, v::AbstractVector)
    N = size(op.S, 2)
    out = if op.centred
        w = op.S' * v .- dot(op.ḡ, v)
        (op.S * w) ./ N .- op.ḡ .* (sum(w) / N)
    else
        (op.S * (op.S' * v)) ./ N
    end
    op.ridge > 0 && (out .+= op.ridge .* v)
    return out
end

Base.eltype(::OuterProductOperator{T}) where {T} = T
Base.size(op::OuterProductOperator) = (size(op.S, 1), size(op.S, 1))
Base.size(op::OuterProductOperator, ::Int) = size(op.S, 1)
LinearAlgebra.issymmetric(::OuterProductOperator) = true
LinearAlgebra.ishermitian(::OuterProductOperator) = true
LinearAlgebra.mul!(y::AbstractVector, op::OuterProductOperator, v::AbstractVector) =
    copyto!(y, op * v)

function _op(m, nlp, x, centred::Bool)
    S = score_matrix(nlp, x)
    size(S, 1) <= m.dense_max && return _outer_product(S, centred, m.ridge)
    ḡ = vec(sum(S; dims = 2)) ./ size(S, 2)
    return OuterProductOperator(S, ḡ, centred, m.ridge)
end

hessian_op(m::BHHHModel, nlp, x)  = _op(m, nlp, x, false)
hessian_op(m::BHHH2Model, nlp, x) = _op(m, nlp, x, true)

# -----------------------------------------------------------------------------
# Nonlinear least squares
# -----------------------------------------------------------------------------

"""
    residuals(prob, x, batch) -> Vector

The residual vector `r(x)` of an [`NLSProblem`](@ref), restricted to `batch`.
"""
function residuals end

"""
    jacobian(prob, x, batch) -> Matrix

The `|batch| × n` Jacobian `∂r/∂x`. Only first derivatives appear: dropping the
`Σ rₙ ∇²rₙ` term is the whole of the Gauss-Newton approximation.
"""
function jacobian end

loss_terms(p::NLSProblem, x, batch) = 0.5 .* residuals(p, x, batch) .^ 2

"""
    scores(p::NLSProblem, x, batch) -> Matrix

Per-observation scores `rₙ ∇rₙ`, as an `n × |batch|` **`Matrix`**.

`permutedims`, not `'`: the adjoint is lazy, and `OuterProductOperator` stores what
it is given.
"""
function scores(p::NLSProblem, x, batch)
    r = residuals(p, x, batch)
    J = jacobian(p, x, batch)
    return permutedims(J .* r)
end

_jacobian_of(m::FullBatchNLP, x) = jacobian(m.prob, x, m.all)
_jacobian_of(m::SampledNLP, x)   = jacobian(m.prob, x, m.batch_g)
_jacobian_of(nlp, ::Any) = throw(ArgumentError(
    "GaussNewtonModel needs `jacobian`; wrap an NLSProblem in FullBatchNLP or " *
    "FiniteSumNLP."))

function _gauss_newton(J::AbstractMatrix, ridge::Real)
    N, n = size(J)
    B = (J' * J) ./ N
    ridge > 0 && (B .+= ridge .* I(n))
    return Symmetric((B .+ B') ./ 2)
end

dense_hessian(m::GaussNewtonModel, nlp, x) = _gauss_newton(_jacobian_of(nlp, x), m.ridge)

function hessian_op(m::GaussNewtonModel, nlp, x)
    J = _jacobian_of(nlp, x)                  # fetched once, not twice
    size(J, 2) <= m.dense_max && return _gauss_newton(J, m.ridge)
    return OuterProductOperator(Matrix(J'), zeros(size(J, 2)), false, m.ridge)
end

# -----------------------------------------------------------------------------
# The identity, measured rather than assumed
# -----------------------------------------------------------------------------

"""
    information_identity_error(prob, x; batch = full_batch(prob)) -> NamedTuple

How badly `V = −H` fails at `x`, in relative Frobenius norm:

```julia
(B_err = ‖B − ∇²f‖ / ‖∇²f‖,
 W_err = ‖W − ∇²f‖ / ‖∇²f‖,
 grad_norm = ‖ḡ‖,
 BW_gap = ‖B − W‖ / ‖∇²f‖)     # equals ‖ḡḡᵀ‖/‖∇²f‖
```

For a correctly specified model at the true parameters these decay like `1/√N`;
anywhere else they do not decay at all. Belonging to [`LikelihoodProblem`](@ref)
asserts that `f` is a negative log-likelihood; whether the model is *correctly
specified* is a statement about the data that no type can carry, and this is how to
check it.

!!! warning "Regularised problems"
    `batch_hess` includes the Hessian of any L2 penalty (`λI`) while the scores do
    not, so with `λ > 0` these numbers measure the identity *plus* the penalty.
    Compare `BHHHModel(ridge = λ)` against the Hessian instead, or evaluate at
    `λ = 0`.
"""
function information_identity_error(prob::ScoredProblem, x; batch = full_batch(prob))
    b = collect(batch)
    S = scores(prob, x, b); N = size(S, 2)
    ḡ = vec(sum(S; dims = 2)) ./ N
    B = (S * S') ./ N
    A = S .- ḡ
    W = (A * A') ./ N
    H = Matrix(batch_hess(prob, x, b))
    nH = max(norm(H), eps())
    return (B_err = norm(B .- H) / nH, W_err = norm(W .- H) / nH,
            grad_norm = norm(ḡ), BW_gap = norm(B .- W) / nH)
end
