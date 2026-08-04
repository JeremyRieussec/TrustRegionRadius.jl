# =============================================================================
# src/Likelihood/likelihood.jl
#
# Outer-product approximations to the Hessian, and the problems that supply the
# per-observation quantities they need.
#
#   BHHHModel          B = (1/N) Σ sₙ sₙᵀ            (Berndt, Hall, Hall & Hausman 1974)
#   BHHH2Model         W = (1/N) Σ (sₙ−ḡ)(sₙ−ḡ)ᵀ    (the same, centred)
#   GaussNewtonModel   B = (1/N) Jᵀ J                (nonlinear least squares)
#
# All three are `ModelHessian`s, so they slot into the existing axis and every
# radius mechanism and subsolver runs over them unchanged.
#
# ---------------------------------------------------------------------------
# What justifies them, and where the justification stops
#
# BHHH rests on the **information identity**: for a correctly specified model at
# the true parameters,
#
#     V = −H,
#
# the population covariance of the scores equalling the negative average
# population Hessian. Minimising the average negative log-likelihood
# f = −(1/N) Σ ln Pₙ, this says B → ∇²f as N → ∞ — so B is an approximation to
# the Hessian that costs nothing beyond the scores already computed for the
# gradient, and is positive semidefinite by construction.
#
# Both conditions in that sentence are load-bearing, and neither is a technicality:
#
#   * **at the true parameters.** Far from the optimum the identity simply does
#     not hold, and B is not an approximation to anything in particular. This is
#     visible in a few lines — see [`information_identity_error`](@ref) — and is
#     why BHHH is known to take small steps early and good ones late.
#   * **correctly specified.** Under misspecification V ≠ −H however large the
#     sample, and the gap does not close. A neural network on MNIST or CIFAR-100
#     is misspecified by construction, so BHHH there is a positive semidefinite
#     preconditioner with no approximation guarantee behind it. It may still work
#     well. It is no longer the Hessian.
#
# For this package that distinction is not an aside. `models.md` already makes the
# point that the model Hessian decides which critical point is reached and that no
# radius mechanism repairs a model reporting the wrong curvature; BHHH under
# misspecification is exactly that situation, with the added trap that it is
# *provably* positive semidefinite and so reports no negative curvature at all.
# Pairing it with `SecondOrder` gives τ ≡ ‖g‖ and a second-order status that means
# nothing. The tests pin both halves.
# =============================================================================

# -----------------------------------------------------------------------------
# Score access
#
# `ScoredProblem` itself lives in the stochastic layer, since it is a refinement
# of `StochasticProblem` and the adaptive sampling rules depend on it.
# -----------------------------------------------------------------------------

"""
    score_matrix(nlp, x) -> Matrix

The `n × N` matrix of per-observation gradients at `x`, which the outer-product
models consume.

Defined for a [`LikelihoodNLP`](@ref) (all terms) and for a [`SampledNLP`](@ref)
over a `ScoredProblem` (the current batch — so under sampling, BHHH is formed from
the same realisations as the gradient, which is what keeps `B` and `g` consistent
within an iteration).
"""
function score_matrix end

"""
    LikelihoodNLP(prob; x0)

The full-sample view of a [`ScoredProblem`](@ref): an `AbstractNLPModel` that
evaluates every term at every call.

Use it for the deterministic runs; wrap the same `prob` in a [`SampledNLP`](@ref)
for the stochastic ones. The problem object is shared, so the two differ only in
how many terms are touched.
"""
mutable struct LikelihoodNLP{P <: ScoredProblem} <: AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModelMeta{Float64, Vector{Float64}}
    counters::Counters
    prob::P
    all::Vector{Int}
end

function LikelihoodNLP(prob::ScoredProblem; x0::AbstractVector = zeros(prob.n))
    meta = NLPModelMeta(prob.n; x0 = Vector{Float64}(x0), name = "LikelihoodNLP")
    return LikelihoodNLP{typeof(prob)}(meta, Counters(), prob, collect(1:n_terms(prob)))
end

NLPModels.obj(m::LikelihoodNLP, x::AbstractVector) =
    (m.counters.neval_obj += 1; batch_obj(m.prob, x, m.all))

function NLPModels.grad!(m::LikelihoodNLP, x::AbstractVector, g::AbstractVector)
    m.counters.neval_grad += 1
    batch_grad!(m.prob, x, m.all, g)
    return g
end

NLPModels.hess(m::LikelihoodNLP, x::AbstractVector) =
    (m.counters.neval_hess += 1; Symmetric(batch_hess(m.prob, x, m.all)))

function NLPModels.hprod!(m::LikelihoodNLP, x::AbstractVector, v::AbstractVector,
                          Hv::AbstractVector; obj_weight = 1.0)
    m.counters.neval_hprod += 1
    mul!(Hv, batch_hess(m.prob, x, m.all), v)
    obj_weight == 1 || (Hv .*= obj_weight)
    return Hv
end

score_matrix(m::LikelihoodNLP, x) = scores(m.prob, x, m.all)
score_matrix(m::SampledNLP{<:Any, <:Any, <:ScoredProblem}, x) =
    scores(m.prob, x, m.batch_g)
score_matrix(nlp, ::Any) = throw(ArgumentError(
    "score_matrix: $(typeof(nlp)) does not expose per-observation scores. " *
    "The outer-product models (BHHHModel, BHHH2Model) need them; wrap a " *
    "ScoredProblem in LikelihoodNLP or SampledNLP, or use ExactHessian."))

# -----------------------------------------------------------------------------
# Outer-product models
# -----------------------------------------------------------------------------

"""
    BHHHModel(; ridge = 0.0, dense_max = 5_000)

`B = (1/N) Σₙ sₙ sₙᵀ`, the average outer product of the per-observation scores —
the BHHH approximation to the Hessian of the average negative log-likelihood.

Two properties, and the trade between them is the whole story:

- **Cheap.** The scores are already computed for the gradient, so `B` costs one
  rank-`N` product and no second derivatives at all.
- **Positive semidefinite by construction.** Every step is a descent direction,
  even where the true Hessian is indefinite.

That second property is a guarantee and a blind spot at once. `B ⪰ 0` always, so
the model can never report negative curvature, and a run over it converges happily
to a saddle of the true objective while every first-order diagnostic looks healthy.
It is the same failure `LBFGSModel` has, arrived at differently, and the same
caution applies: `SecondOrder` over a `BHHHModel` gives `τ ≡ ‖g‖` and a
`:second_order` status that certifies nothing.

`ridge > 0` adds `ridge·I`, which is worth having when `N < n`: with fewer
observations than parameters `B` is singular by construction, since it is a sum of
`N` rank-one terms in `n` dimensions. This bites under sampling — a small batch
makes `B` rank-deficient regardless of the problem — so the `ridge` and the batch
size are not independent choices.

# When the approximation is justified

`B → ∇²f` requires the information identity, hence a **correctly specified model**
evaluated **at the true parameters**. Far from the optimum, or under
misspecification, `B` approximates nothing in particular; see
[`information_identity_error`](@ref) to measure how far off it is on a given
problem, rather than assuming.
"""
struct BHHHModel <: ModelHessian
    ridge::Float64
    dense_max::Int
    BHHHModel(; ridge::Real = 0.0, dense_max::Int = 5_000) = new(float(ridge), dense_max)
end

"""
    BHHH2Model(; ridge = 0.0, dense_max = 5_000)

`W = (1/N) Σₙ (sₙ − ḡ)(sₙ − ḡ)ᵀ`: the sample **covariance** of the scores, rather
than their average outer product.

`W` and `B` coincide exactly when `ḡ = 0`, i.e. at the maximiser, and differ away
from it by the rank-one term `ḡ ḡᵀ`:

```math
B = W + \\bar g \\bar g^\\top .
```

So `B ⪰ W`, and the difference is largest early in a run when the gradient is
large — precisely where the identity does not hold and neither matrix approximates
the Hessian. Which of the two is better in that regime is an empirical question,
and one this package can answer directly, which is the reason to have both.
"""
struct BHHH2Model <: ModelHessian
    ridge::Float64
    dense_max::Int
    BHHH2Model(; ridge::Real = 0.0, dense_max::Int = 5_000) = new(float(ridge), dense_max)
end

function _outer_product(S::AbstractMatrix, centred::Bool, ridge::Real)
    n, N = size(S)
    ḡ = vec(sum(S; dims = 2)) ./ N
    A = centred ? (S .- ḡ) : S
    B = (A * A') ./ N
    ridge > 0 && (B .+= ridge .* I(n))
    return Symmetric((B .+ B') ./ 2)          # symmetrise against rounding
end

dense_hessian(m::BHHHModel, nlp, x)  = _outer_product(score_matrix(nlp, x), false, m.ridge)
dense_hessian(m::BHHH2Model, nlp, x) = _outer_product(score_matrix(nlp, x), true,  m.ridge)

"""
    OuterProductOperator(S, ḡ, centred, ridge)

Matrix-free `B·v` for the outer-product models: `B v = (1/N) S (Sᵀ v) + ridge·v`,
two matrix–vector products against the `n × N` score matrix and no `n × n` array
anywhere.

This is what makes the models usable at neural-network scale. Forming `B` densely
costs `n²`, which for even a small MLP is out of reach — 25 450 parameters is
648 million entries — while `S` costs `n·N` and stays in memory for a batch of a
few hundred. Pair it with [`SteihaugCG`](@ref), which needs only the product.
"""
struct OuterProductOperator{T}
    S::Matrix{T}
    ḡ::Vector{T}
    centred::Bool
    ridge::T
end

function Base.:*(op::OuterProductOperator, v::AbstractVector)
    N = size(op.S, 2)
    if op.centred
        w = op.S' * v .- dot(op.ḡ, v)
        out = (op.S * w) ./ N .- op.ḡ .* (sum(w) / N)
    else
        out = (op.S * (op.S' * v)) ./ N
    end
    op.ridge > 0 && (out .+= op.ridge .* v)
    return out
end
Base.size(op::OuterProductOperator) = (size(op.S, 1), size(op.S, 1))
Base.size(op::OuterProductOperator, i::Int) = size(op.S, 1)
LinearAlgebra.mul!(y::AbstractVector, op::OuterProductOperator, v::AbstractVector) =
    copyto!(y, op * v)

function _op(m, nlp, x, centred::Bool)
    S = score_matrix(nlp, x)
    n = size(S, 1)
    n <= m.dense_max && return dense_hessian(m, nlp, x)
    ḡ = vec(sum(S; dims = 2)) ./ size(S, 2)
    return OuterProductOperator(S, ḡ, centred, m.ridge)
end

hessian_op(m::BHHHModel, nlp, x)  = _op(m, nlp, x, false)
hessian_op(m::BHHH2Model, nlp, x) = _op(m, nlp, x, true)

"""
    GaussNewtonModel(; ridge = 0.0, dense_max = 5_000)

`B = (1/N) Jᵀ J` for a nonlinear least-squares objective
`f(x) = (1/2N) Σₙ rₙ(x)²`, whose exact Hessian is

```math
\\nabla^2 f = \\frac1N\\Bigl( J^\\top J + \\sum_n r_n \\nabla^2 r_n \\Bigr) .
```

Gauss–Newton drops the second term. The approximation is good exactly when it is
small — either small residuals at the solution, or nearly linear `rₙ` — and can be
arbitrarily bad otherwise, which is the large-residual case where Gauss–Newton is
known to lose its superlinear rate.

The structural parallel with [`BHHHModel`](@ref) is worth seeing: both discard a
term that vanishes under a modelling assumption (zero residuals; correct
specification at the truth), both are positive semidefinite as a result, and both
therefore cannot report negative curvature. The assumption differs; the
consequence for second-order behaviour is identical.

Requires a problem implementing `residuals(p, x, batch)` and
`jacobian(p, x, batch)`.
"""
struct GaussNewtonModel <: ModelHessian
    ridge::Float64
    dense_max::Int
    GaussNewtonModel(; ridge::Real = 0.0, dense_max::Int = 5_000) = new(float(ridge), dense_max)
end

"""
    residuals(prob, x, batch) -> Vector

The residual vector `r(x)` of a least-squares problem, restricted to `batch`.
"""
function residuals end

"""
    jacobian(prob, x, batch) -> Matrix

The `length(batch) × n` Jacobian `∂r/∂x`, from which [`GaussNewtonModel`](@ref)
forms `JᵀJ/N`. Only the first derivatives of the residuals appear: dropping the
`Σ rₙ ∇²rₙ` term is the whole of the Gauss–Newton approximation.
"""
function jacobian end

"""
    NLSProblem <: ScoredProblem

Nonlinear least squares, `f(x) = (1/2N) Σ rₙ(x)²`, implementing `residuals` and
`jacobian` so that [`GaussNewtonModel`](@ref) applies. The per-observation score
is `rₙ ∇rₙ`, so BHHH applies too and the two can be compared on one problem.
"""
abstract type NLSProblem <: ScoredProblem end

loss_terms(p::NLSProblem, x, batch) = 0.5 .* residuals(p, x, batch) .^ 2
function scores(p::NLSProblem, x, batch)
    r = residuals(p, x, batch)
    J = jacobian(p, x, batch)                 # length(batch) × n
    return (J .* r)'                          # n × length(batch)
end

function dense_hessian(m::GaussNewtonModel, nlp, x)
    J = _jacobian_of(nlp, x)
    n = size(J, 2); N = size(J, 1)
    B = (J' * J) ./ N
    m.ridge > 0 && (B .+= m.ridge .* I(n))
    return Symmetric((B .+ B') ./ 2)
end

_jacobian_of(m::LikelihoodNLP, x) = jacobian(m.prob, x, m.all)
_jacobian_of(m::SampledNLP, x)    = jacobian(m.prob, x, m.batch_g)
_jacobian_of(nlp, ::Any) = throw(ArgumentError(
    "GaussNewtonModel needs `jacobian`; wrap an NLSProblem in LikelihoodNLP or SampledNLP."))

function hessian_op(m::GaussNewtonModel, nlp, x)
    J = _jacobian_of(nlp, x)
    size(J, 2) <= m.dense_max && return dense_hessian(m, nlp, x)
    return OuterProductOperator(Matrix(J'), zeros(size(J, 2)), false, m.ridge)
end

# The outer-product models carry no state between iterations: `B` is a function of
# the current iterate and the current batch alone.
reset_model!(::Union{BHHHModel, BHHH2Model, GaussNewtonModel}, ::Int) = nothing
update_model!(::Union{BHHHModel, BHHH2Model, GaussNewtonModel}, s, y) = nothing

# -----------------------------------------------------------------------------
# The information identity, measured rather than assumed
# -----------------------------------------------------------------------------

"""
    information_identity_error(prob, x; batch = all) -> NamedTuple

How badly the identity `V = −H` fails at `x`, in relative Frobenius norm:

```julia
(B_err = ‖B − ∇²f‖ / ‖∇²f‖,
 W_err = ‖W − ∇²f‖ / ‖∇²f‖,
 grad_norm = ‖ḡ‖,
 BW_gap = ‖B − W‖ / ‖∇²f‖)     # equals ‖ḡḡᵀ‖/‖∇²f‖
```

For a correctly specified model at the true parameters these decay like `1/√N`;
anywhere else they do not decay at all. Running this before trusting a BHHH run is
two lines and settles the question for the problem at hand, which is a better basis
than the general claim.

`BW_gap` is the rank-one `ḡḡᵀ` separating [`BHHHModel`](@ref) from
[`BHHH2Model`](@ref); it goes to zero at a stationary point, which is why the two
agree near the optimum and can disagree sharply away from it.
"""
function information_identity_error(prob::ScoredProblem, x; batch = 1:n_terms(prob))
    b = collect(batch)
    S = scores(prob, x, b); N = size(S, 2)
    ḡ = vec(sum(S; dims = 2)) ./ N
    B = (S * S') ./ N
    A = S .- ḡ
    W = (A * A') ./ N
    H = Matrix(batch_hess(prob, x, b))
    nH = norm(H)
    return (B_err = norm(B .- H) / nH, W_err = norm(W .- H) / nH,
            grad_norm = norm(ḡ), BW_gap = norm(B .- W) / nH)
end
