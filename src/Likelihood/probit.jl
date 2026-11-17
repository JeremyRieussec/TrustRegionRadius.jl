# =============================================================================
# src/Likelihood/probit.jl
#
# Binary probit, the second binary-choice likelihood beside `LogisticRegression`.
#
# The two share a design matrix and differ in the link, and the difference is
# what earns the probit a place here. The logit link is canonical, so the
# per-observation Hessian `p(1−p) x xᵀ` does not involve the outcome: observed
# and expected information are one matrix. The probit link is not canonical, so
# the per-observation Hessian depends on `yₙ` and the two matrices differ in
# every sample. BHHH then stands against an exact Hessian that is itself random
# in the outcomes, and the information identity ties the three together in
# expectation only.
#
# The tails need care. `Φ(t)` underflows to zero near `t = −38.5`, long before
# the index stops being a number a run can visit, so `ln Φ` and `φ/Φ` are
# written through the scaled complementary error function `erfcx`, which has no
# such limit.
# =============================================================================

const _INV_SQRT2      = inv(sqrt(2.0))
const _SQRT_2_OVER_PI = sqrt(2 / π)

"""
    _neglogΦ(t)

`−ln Φ(t)`, the negative log of the standard normal distribution function, finite
for every finite `t` whose square is.

On the right, `Φ(t) = 1 − ½ erfc(t/√2)` and `log1p` keeps the digits. On the left,
`Φ(t) = ½ erfcx(−t/√2) e^{−t²/2}`, so the logarithm splits into `t²/2` and the
logarithm of a factor that decays like `1/|t|` and never underflows.
"""
@inline function _neglogΦ(t::Real)
    t > 0 && return -log1p(-0.5 * erfc(t * _INV_SQRT2))
    return 0.5 * t * t - log(0.5 * erfcx(-t * _INV_SQRT2))
end

"""
    _mills(t)

The inverse Mills ratio `λ(t) = φ(t)/Φ(t)`, written as `√(2/π) / erfcx(−t/√2)`.

The two exponentials of the quotient cancel analytically, so nothing underflows on
the left, where `λ(t) ≈ −t`. Far on the right `erfcx` overflows to `Inf` and the
ratio is returned as zero, which is its limit.
"""
@inline _mills(t::Real) = _SQRT_2_OVER_PI / erfcx(-t * _INV_SQRT2)

"""
    _probit_curvature(t)

`λ(t)(λ(t) + t)`, the second derivative of `−ln Φ` at `t`.

It lies strictly between 0 and 1, tending to 1 on the left and to 0 on the right.
The sum `λ + t` cancels in the far left tail, where `λ ≈ −t`, so the result is
clamped to the interval the mathematics gives it.
"""
@inline function _probit_curvature(t::Real)
    λ = _mills(t)
    return clamp(λ * (λ + t), 0.0, 1.0)
end

"""
    Probit(X, y)
    Probit(; K = 5, M = 2_000, β_star = nothing, seed = 0, intercept = true,
             signal = 1.0, noise = 0.0)

Binary probit: minimise the average negative log-likelihood

```math
f(\\beta) = -\\frac1M \\sum_n \\bigl[ y_n \\ln \\Phi(x_n^\\top\\beta)
                                    + (1-y_n)\\ln \\Phi(-x_n^\\top\\beta) \\bigr] ,
```

with `Φ` the standard normal distribution function. `X` is `M × K` and `y` holds
the outcomes as 0 or 1.

The keyword form generates synthetic data from the latent-variable form of the
model, `yₙ = 1` when `xₙᵀβ* + eₙ > 0` with `eₙ ~ N(0, 1)`, so with `noise = 0` the
model is **correctly specified**. `xₙ ~ N(0, I)`, with the first coordinate set to 1
if `intercept`. The generating parameters are retrieved with [`β_true`](@ref).

For a given `K`, `M` and `seed` the design matrix and `β*` are those of
[`LogisticRegression`](@ref) with the same arguments, so the two links can be
compared on one design.

# Scores and curvature

Write `qₙ = 2yₙ − 1` for the sign the outcome gives the index, `tₙ = qₙ xₙᵀβ`, and
`λ(t) = φ(t)/Φ(t)` for the inverse Mills ratio. The objective is the mean of
`−ln Φ(tₙ)`, and the per-observation score and Hessian are

```math
s_n = -q_n \\lambda(t_n)\\, x_n ,
\\qquad
H_n = \\lambda(t_n)\\bigl(\\lambda(t_n) + t_n\\bigr)\\, x_n x_n^\\top .
```

The weight `λ(t)(λ(t) + t)` lies strictly between 0 and 1, so each `Hₙ` is positive
semidefinite and the objective is convex in `β`.

# Observed and expected information differ

`Hₙ` depends on the outcome through `tₙ`. In [`LogisticRegression`](@ref) it does
not: the per-observation Hessian there is `pₙ(1−pₙ) xₙxₙᵀ` whatever `yₙ` is,
because the logit link is canonical. Here the exact Hessian is random in the
outcomes, and only its expectation over `y` given `x` is the Fisher information.

At `β*`, with `z = xᵀβ*`,

```math
E\\bigl[s s^\\top \\mid x\\bigr] = E\\bigl[H \\mid x\\bigr]
  = \\frac{\\varphi(z)^2}{\\Phi(z)\\,\\Phi(-z)}\\; x x^\\top ,
```

so the information identity holds and BHHH is justified, as for the logit. In a
sample of `M` the two sides agree to `O(M^{-1/2})`, which
[`information_identity_error`](@ref) measures. Away from `β*` the first equality
fails and the gap does not shrink with `M`.

!!! warning "`noise > 0` breaks the identity"
    `noise = σ` draws a separate coefficient vector `βₙ = β* + εₙ` with
    `εₙ ~ N(0, σ²I)` for each observation and generates its outcome from it. The
    index then carries the extra term `xₙᵀεₙ`, whose variance `σ²‖xₙ‖²` changes
    with `xₙ`. The generating process is a heteroscedastic probit while the fitted
    model has unit variance. That is misspecification: the identity fails at every
    `β`, [`β_true`](@ref) is no longer the minimiser of `f` even as `M → ∞`, and
    the gap does not shrink with `M`.

    The default is `noise = 0.0`. Set it deliberately, and read
    [`information_identity_error`](@ref) instead of assuming either way.

# The tails

`Φ(t)` underflows to zero near `t = −38.5`, so `ln Φ` evaluated as written returns
`−Inf` there and the score divides zero by zero. The loss, the score and the
curvature are therefore computed through `erfcx`, and stay finite wherever the
square of the index does.
"""
struct Probit <: LikelihoodProblem
    n::Int
    M::Int
    X::Matrix{Float64}          # M × n
    y::Vector{Float64}          # 0 or 1
    β_star::Union{Vector{Float64}, Nothing}
    noise::Float64              # the σ used to generate, 0 when specified correctly
end

function Probit(X::AbstractMatrix, y::AbstractVector; β_star = nothing,
                noise::Real = 0.0)
    size(X, 1) == length(y) || throw(DimensionMismatch(
        "Probit: X has $(size(X, 1)) rows but y has $(length(y)) entries"))
    all(v -> v == 0 || v == 1, y) || throw(ArgumentError("Probit: y must be 0/1"))
    return Probit(size(X, 2), size(X, 1), Matrix{Float64}(X), Vector{Float64}(y),
                  β_star === nothing ? nothing : Vector{Float64}(β_star),
                  float(noise))
end

function Probit(; K::Int = 5, M::Int = 2_000,
                  β_star::Union{AbstractVector, Nothing} = nothing,
                  seed::Int = 0, intercept::Bool = true,
                  signal::Real = 1.0, noise::Real = 0.0)
    K >= 1 || throw(ArgumentError("Probit: need K ≥ 1"))
    M >= 1 || throw(ArgumentError("Probit: need M ≥ 1"))
    noise >= 0 || throw(ArgumentError("Probit: need noise ≥ 0"))
    rng = MersenneTwister(seed)
    # β* and then X, in the order LogisticRegression draws them, so that one seed
    # gives both problems the same design.
    β = β_star === nothing ? signal .* randn(rng, K) : Vector{Float64}(β_star)
    length(β) == K || throw(DimensionMismatch(
        "Probit: β_star must have length K = $K"))
    X = randn(rng, M, K)
    intercept && (X[:, 1] .= 1.0)
    z = X * β
    # βₙ = β* + σεₙ adds xₙᵀεₙ to the index. At noise = 0 nothing is drawn and the
    # model is correctly specified.
    noise > 0 && (z .+= noise .* vec(sum(X .* randn(rng, M, K); dims = 2)))
    y = Float64.(z .+ randn(rng, M) .> 0)
    return Probit(K, M, X, y, β, float(noise))
end

population(p::Probit) = p.M

"""
    β_true(p::Probit) -> Vector or nothing

The generating coefficients. With `noise > 0` these are the **mean** of the
per-observation coefficients, not the minimiser of `f`.
"""
β_true(p::Probit) = p.β_star

function loss_terms(p::Probit, x, batch)
    out = Vector{Float64}(undef, length(batch))
    for (j, i) in enumerate(batch)
        z = dot(view(p.X, i, :), x)
        out[j] = _neglogΦ(p.y[i] == 1 ? z : -z)
    end
    return out
end

"""
    scores(p::Probit, x, batch) -> Matrix (K × |batch|)

Column `j` is `−qₙ λ(tₙ) xₙ` for `n = batch[j]`, the gradient of the negative
log-likelihood of that observation.
"""
function scores(p::Probit, x, batch)
    S = Matrix{Float64}(undef, p.n, length(batch))
    for (j, i) in enumerate(batch)
        xi = view(p.X, i, :)
        q = p.y[i] == 1 ? 1.0 : -1.0
        r = -q * _mills(q * dot(xi, x))          # ∇ of the NEGATIVE log-likelihood
        @. S[:, j] = r * xi
    end
    return S
end

"""
    batch_hess(p::Probit, x, batch)

`(1/N) Σ λ(tₙ)(λ(tₙ) + tₙ) xₙxₙᵀ`, the exact Hessian.

The weights are positive, so this is a weighted Gram matrix as in the logit, built
with one rank-`N` product. What differs is that the weights depend on the outcomes:
this is the observed information, and it is not the Fisher information.
"""
function batch_hess(p::Probit, x, batch)
    N = length(batch)
    Xb = Matrix{Float64}(undef, N, p.n)
    for (j, i) in enumerate(batch)
        xi = view(p.X, i, :)
        q = p.y[i] == 1 ? 1.0 : -1.0
        w = sqrt(_probit_curvature(q * dot(xi, x)))
        @. Xb[j, :] = w * xi
    end
    H = (Xb' * Xb) ./ N
    return (H .+ H') ./ 2
end
