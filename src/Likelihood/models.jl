# =============================================================================
# src/Likelihood/models.jl
#
# Two concrete `ScoredProblem`s.
#
#   LogisticRegression   correctly specified by construction, so the information
#                        identity holds and BHHH is justified. The controlled case.
#   MLPClassifier        softmax cross-entropy over a one-hidden-layer network.
#                        Misspecified by construction, so it is not. The realistic
#                        case, and the contrast is the point.
#
# Both report per-observation scores, so BHHH, BHHH-2, the exact Hessian and the
# quasi-Newton models can all be run on the same problem and compared.
# =============================================================================

# -----------------------------------------------------------------------------
# Logistic regression
# -----------------------------------------------------------------------------

"""
    LogisticRegression(X, y)
    LogisticRegression(; K = 5, M = 2_000, β_true = nothing, seed = 0, intercept = true)

Binary logistic regression: minimise the average negative log-likelihood

```math
f(\\beta) = -\\frac1M \\sum_n \\bigl[ y_n \\ln \\sigma(x_n^\\top\\beta)
                                    + (1-y_n)\\ln(1-\\sigma(x_n^\\top\\beta)) \\bigr] .
```

The keyword form generates synthetic data with the model **correctly specified**:
`xₙ ~ N(0, I)` (first coordinate 1 if `intercept`), and `yₙ ~ Bernoulli(σ(xₙᵀβ*))`
drawn from the very model being fitted. `K` sets the number of parameters, so the
identity can be watched as the dimension grows.

# Why the identity holds here, exactly

The score is `sₙ = (yₙ − pₙ) xₙ` and the per-observation Hessian is
`pₙ(1−pₙ) xₙxₙᵀ`. At `β*`, `E[(y−p)² | x] = Var(y | x) = p(1−p)`, so

```math
V = E\\bigl[(y-p)^2 x x^\\top\\bigr] = E\\bigl[p(1-p) x x^\\top\\bigr] = -H
```

in the population — not approximately, identically. In a sample of size `M` the
two agree to `O(M^{-1/2})`, which is what
[`information_identity_error`](@ref) measures. Away from `β*` the first equality
fails, because `E[(y−p̂)²] = p(1−p) + (p − p̂)²`, and the gap `(p − p̂)²` does not
shrink with `M` at all. That extra term is precisely why BHHH takes poor steps far
from the optimum and good ones near it.

`β_true(prob)` returns the generating parameters, so a run can be scored against
them rather than against its own stopping test.
"""
struct LogisticRegression <: ScoredProblem
    n::Int
    M::Int
    X::Matrix{Float64}          # M × n
    y::Vector{Float64}
    β_true::Union{Vector{Float64}, Nothing}
end

function LogisticRegression(X::AbstractMatrix, y::AbstractVector; β_true = nothing)
    size(X, 1) == length(y) || throw(DimensionMismatch("LogisticRegression: rows of X must match y"))
    all(v -> v == 0 || v == 1, y) || throw(ArgumentError("LogisticRegression: y must be 0/1"))
    return LogisticRegression(size(X, 2), size(X, 1), Matrix{Float64}(X),
                              Vector{Float64}(y), β_true)
end

function LogisticRegression(; K::Int = 5, M::Int = 2_000,
                              β_true::Union{AbstractVector, Nothing} = nothing,
                              seed::Int = 0, intercept::Bool = true,
                              signal::Real = 1.0)
    K >= 1 || throw(ArgumentError("LogisticRegression: need K ≥ 1"))
    rng = MersenneTwister(seed)
    β = β_true === nothing ? signal .* randn(rng, K) : Vector{Float64}(β_true)
    length(β) == K || throw(DimensionMismatch("LogisticRegression: β_true must have length K"))
    X = randn(rng, M, K)
    intercept && (X[:, 1] .= 1.0)
    p = _σ.(X * β)
    y = Float64.(rand(rng, M) .< p)
    return LogisticRegression(K, M, X, y, β)
end

@inline _σ(z) = z >= 0 ? 1 / (1 + exp(-z)) : (e = exp(z); e / (1 + e))

n_terms(p::LogisticRegression) = p.M

"""
    β_true(prob) -> Vector or nothing

The parameters that generated the synthetic data, when there are any.

A run should be scored against these rather than against its own stopping test —
though note the maximum-likelihood estimate for a finite sample differs from them
by the usual `O(M^{-1/2})` sampling error, so `‖β̂ − β*‖` does not go to zero at
the optimum.
"""
β_true(p::LogisticRegression) = p.β_true

function loss_terms(p::LogisticRegression, x, batch)
    out = Vector{Float64}(undef, length(batch))
    for (j, i) in enumerate(batch)
        z = dot(view(p.X, i, :), x)
        # log(1+exp(z)) computed stably; f = −[y z − log(1+e^z)]
        lse = z > 0 ? z + log1p(exp(-z)) : log1p(exp(z))
        out[j] = lse - p.y[i] * z
    end
    return out
end

function scores(p::LogisticRegression, x, batch)
    S = Matrix{Float64}(undef, p.n, length(batch))
    for (j, i) in enumerate(batch)
        xi = view(p.X, i, :)
        r = _σ(dot(xi, x)) - p.y[i]              # ∇ of the NEGATIVE log-likelihood
        @. S[:, j] = r * xi
    end
    return S
end

function batch_hess(p::LogisticRegression, x, batch)
    H = zeros(p.n, p.n)
    for i in batch
        xi = view(p.X, i, :)
        w = (q = _σ(dot(xi, x)); q * (1 - q))
        BLAS.ger!(w, Vector(xi), Vector(xi), H)
    end
    return H ./ length(batch)
end

true_objective(p::LogisticRegression, x) = batch_obj(p, x, 1:p.M)
true_gradient(p::LogisticRegression, x)  =
    (g = zeros(p.n); batch_grad!(p, x, 1:p.M, g); g)

# -----------------------------------------------------------------------------
# One-hidden-layer classifier
# -----------------------------------------------------------------------------

"""
    MLPClassifier(X, y, n_class; hidden = 16, seed = 0, λ = 0.0)

Softmax cross-entropy over a one-hidden-layer `tanh` network, with per-observation
scores so that BHHH applies.

`X` is `M × d`, `y` holds labels in `1:n_class`. The parameter vector packs
`(W₁, b₁, W₂, b₂)` in that order, so `n = hidden·d + hidden + n_class·hidden + n_class`.
`λ > 0` adds `λ‖x‖²/2`, which makes the objective strongly convex in a neighbourhood
and is often what keeps a second-order method well posed here.

!!! warning "The information identity does not hold"
    This model is misspecified — the data were not generated by a tanh network — and
    the run is nowhere near "the true parameters", which do not exist. So BHHH's
    justification is absent, not merely weakened. `B` remains positive semidefinite
    and usable as a preconditioner, but it is not an approximation to `∇²f` and
    should not be described as one. [`information_identity_error`](@ref) reports
    the gap; on a network it does not shrink with the sample.

    Two consequences follow for this package specifically. `B ⪰ 0` means the model
    can never report negative curvature, so a run converges contentedly to saddles
    — which for a network are the dominant critical points. And `SecondOrder` over
    a `BHHHModel` gives `τ ≡ ‖g‖`, so a `:second_order` status certifies nothing.
    Use `ExactHessian` when the second-order question is the question.

!!! note "Dimensions"
    A dense `n × n` Hessian needs `n²` entries: MNIST at `784 → 16 → 10` is 12 730
    parameters and 1.3 GB dense. Use `hessian_op` with [`SteihaugCG`](@ref), which
    needs only `B·v` and costs `n·N` for the score matrix. `dense_max` on the model
    controls the switch. `ExactMS` is out of the question at this scale, which is
    itself worth reporting: the subsolver axis is constrained by the model axis.
"""
struct MLPClassifier <: ScoredProblem
    n::Int
    M::Int
    d::Int
    h::Int
    c::Int
    X::Matrix{Float64}          # M × d
    y::Vector{Int}
    λ::Float64
end

function MLPClassifier(X::AbstractMatrix, y::AbstractVector{<:Integer}, n_class::Int;
                       hidden::Int = 16, λ::Real = 0.0)
    M, d = size(X)
    M == length(y) || throw(DimensionMismatch("MLPClassifier: rows of X must match y"))
    all(v -> 1 <= v <= n_class, y) || throw(ArgumentError("MLPClassifier: labels must lie in 1:n_class"))
    n = hidden * d + hidden + n_class * hidden + n_class
    return MLPClassifier(n, M, d, hidden, n_class, Matrix{Float64}(X),
                         Vector{Int}(y), float(λ))
end

n_terms(p::MLPClassifier) = p.M

"Unpack the flat parameter vector into (W₁, b₁, W₂, b₂) as views."
function _unpack(p::MLPClassifier, θ)
    o = 0
    W1 = reshape(view(θ, o+1:o+p.h*p.d), p.h, p.d); o += p.h * p.d
    b1 = view(θ, o+1:o+p.h);                        o += p.h
    W2 = reshape(view(θ, o+1:o+p.c*p.h), p.c, p.h); o += p.c * p.h
    b2 = view(θ, o+1:o+p.c)
    return W1, b1, W2, b2
end

function _softmax!(z)
    m = maximum(z); z .= exp.(z .- m); z ./= sum(z); return z
end

function loss_terms(p::MLPClassifier, θ, batch)
    W1, b1, W2, b2 = _unpack(p, θ)
    out = Vector{Float64}(undef, length(batch))
    reg = p.λ > 0 ? 0.5 * p.λ * sum(abs2, θ) : 0.0
    a1 = Vector{Float64}(undef, p.h); z2 = Vector{Float64}(undef, p.c)
    for (j, i) in enumerate(batch)
        xi = view(p.X, i, :)
        mul!(a1, W1, xi); a1 .+= b1; a1 .= tanh.(a1)
        mul!(z2, W2, a1); z2 .+= b2
        m = maximum(z2)
        out[j] = (m + log(sum(exp, z2 .- m))) - z2[p.y[i]] + reg
    end
    return out
end

"""
    scores(p::MLPClassifier, θ, batch) -> Matrix (n × |batch|)

Per-observation gradients by explicit backpropagation.

Standard backprop accumulates the *sum* over a batch; BHHH needs the terms kept
apart, so each column is formed from one forward and one backward pass. That is
the price of the outer product: the gradient itself would cost one batched pass,
and the score matrix costs `|batch|` unbatched ones plus `n·|batch|` storage.
"""
function scores(p::MLPClassifier, θ, batch)
    W1, b1, W2, b2 = _unpack(p, θ)
    S = zeros(p.n, length(batch))
    a1 = Vector{Float64}(undef, p.h); z2 = Vector{Float64}(undef, p.c)
    dz1 = Vector{Float64}(undef, p.h)
    oW1 = 0; ob1 = p.h * p.d; oW2 = ob1 + p.h; ob2 = oW2 + p.c * p.h
    for (j, i) in enumerate(batch)
        xi = view(p.X, i, :)
        mul!(a1, W1, xi); a1 .+= b1; a1 .= tanh.(a1)
        mul!(z2, W2, a1); z2 .+= b2
        dz2 = _softmax!(copy(z2)); dz2[p.y[i]] -= 1.0        # ∂ℓ/∂z₂
        col = view(S, :, j)
        @views mul!(reshape(col[oW2+1:oW2+p.c*p.h], p.c, p.h), dz2, a1')
        @views col[ob2+1:ob2+p.c] .= dz2
        mul!(dz1, W2', dz2); @. dz1 *= (1 - a1^2)
        @views mul!(reshape(col[oW1+1:oW1+p.h*p.d], p.h, p.d), dz1, xi')
        @views col[ob1+1:ob1+p.h] .= dz1
        p.λ > 0 && (col .+= p.λ .* θ)
    end
    return S
end

"""
    batch_hess(p::MLPClassifier, θ, batch)

The exact Hessian by finite differences of the analytic gradient.

`O(n)` gradient evaluations, so it is for the small configurations only — which is
exactly the regime where the comparison against BHHH is worth making, since it is
the only regime where the truth is available at all. For anything larger, run
`ExactHessian` against a Hessian-vector product or accept that BHHH is being used
without a reference.
"""
function batch_hess(p::MLPClassifier, θ, batch)
    n = p.n
    n <= 2_000 || throw(ArgumentError(
        "MLPClassifier: exact Hessian at n = $n is impractical (O(n) gradient " *
        "evaluations and an n×n matrix). Use BHHHModel/GaussNewtonModel with " *
        "SteihaugCG, or shrink `hidden` / the input dimension."))
    H = zeros(n, n)
    g0 = zeros(n); batch_grad!(p, θ, batch, g0)
    ε = cbrt(eps()) * max(1.0, norm(θ))
    gp = zeros(n); θp = Vector{Float64}(θ)
    for j in 1:n
        old = θp[j]; θp[j] = old + ε
        batch_grad!(p, θp, batch, gp)
        @. H[:, j] = (gp - g0) / ε
        θp[j] = old
    end
    return (H .+ H') ./ 2
end

"""
    init_params(p::MLPClassifier; seed = 0) -> Vector

Random initial weights, scaled by `1/√fan_in` so the `tanh` units start
unsaturated. Biases start at zero.
"""
function init_params(p::MLPClassifier; seed::Int = 0)
    rng = MersenneTwister(seed)
    θ = zeros(p.n)
    W1, b1, W2, b2 = _unpack(p, θ)
    W1 .= randn(rng, p.h, p.d) ./ sqrt(p.d)
    W2 .= randn(rng, p.c, p.h) ./ sqrt(p.h)
    return θ
end

"""
    accuracy(p::MLPClassifier, θ, batch = 1:p.M) -> Float64

Fraction of `batch` classified correctly at `θ`. Reported alongside the loss
because a trust-region run reports objective values, and on a classification
problem the objective is not the quantity anyone cares about.
"""
function accuracy(p::MLPClassifier, θ, batch = 1:p.M)
    W1, b1, W2, b2 = _unpack(p, θ)
    correct = 0
    a1 = Vector{Float64}(undef, p.h); z2 = Vector{Float64}(undef, p.c)
    for i in batch
        xi = view(p.X, i, :)
        mul!(a1, W1, xi); a1 .+= b1; a1 .= tanh.(a1)
        mul!(z2, W2, a1); z2 .+= b2
        correct += (argmax(z2) == p.y[i])
    end
    return correct / length(batch)
end
