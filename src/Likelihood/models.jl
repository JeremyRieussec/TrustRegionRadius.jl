# =============================================================================
# src/Likelihood/models.jl
#
# Two concrete `ScoredProblem`s.
#
#   LogisticRegression   correctly specified by construction, so the information
#                        identity holds and BHHH is justified. The controlled case.
#
# Both are `LikelihoodProblem`s, which is what makes `BHHHModel` legal over them:
# the requirement is a statement about f being a negative log-likelihood, and the
# type carries it. Whether the model is *correctly specified* — the other half of
# the justification — is a property of the data that no type can assert, which is
# what `information_identity_error` is for.
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
    LogisticRegression(; K = 5, M = 2_000, β_star = nothing, seed = 0,
                         intercept = true, signal = 1.0)

Binary logistic regression: minimise the average negative log-likelihood

```math
f(\\beta) = -\\frac1M \\sum_n \\bigl[ y_n \\ln \\sigma(x_n^\\top\\beta)
                                    + (1-y_n)\\ln(1-\\sigma(x_n^\\top\\beta)) \\bigr] .
```

The keyword form generates synthetic data with the model **correctly specified**:
`xₙ ~ N(0, I)` (first coordinate 1 if `intercept`), and `yₙ ~ Bernoulli(σ(xₙᵀβ*))`
drawn from the very model being fitted. `K` sets the number of parameters, so the
identity can be watched as the dimension grows.

The generating parameters are passed as `β_star` and retrieved with
[`β_true`](@ref); the keyword was previously `β_true` as well, colliding with the
accessor of the same name.

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
"""
struct LogisticRegression <: LikelihoodProblem
    n::Int
    M::Int
    X::Matrix{Float64}          # M × n
    y::Vector{Float64}
    β_star::Union{Vector{Float64}, Nothing}
end

function LogisticRegression(X::AbstractMatrix, y::AbstractVector; β_star = nothing)
    size(X, 1) == length(y) || throw(DimensionMismatch("LogisticRegression: rows of X must match y"))
    all(v -> v == 0 || v == 1, y) || throw(ArgumentError("LogisticRegression: y must be 0/1"))
    return LogisticRegression(size(X, 2), size(X, 1), Matrix{Float64}(X),
                              Vector{Float64}(y),
                              β_star === nothing ? nothing : Vector{Float64}(β_star))
end

function LogisticRegression(; K::Int = 5, M::Int = 2_000,
                              β_star::Union{AbstractVector, Nothing} = nothing,
                              seed::Int = 0, intercept::Bool = true,
                              signal::Real = 1.0)
    K >= 1 || throw(ArgumentError("LogisticRegression: need K ≥ 1"))
    rng = MersenneTwister(seed)
    β = β_star === nothing ? signal .* randn(rng, K) : Vector{Float64}(β_star)
    length(β) == K || throw(DimensionMismatch("LogisticRegression: β_star must have length K"))
    X = randn(rng, M, K)
    intercept && (X[:, 1] .= 1.0)
    p = _σ.(X * β)
    y = Float64.(rand(rng, M) .< p)
    return LogisticRegression(K, M, X, y, β)
end

@inline _σ(z) = z >= 0 ? 1 / (1 + exp(-z)) : (e = exp(z); e / (1 + e))

population(p::LogisticRegression) = p.M

"""
    β_true(prob) -> Vector or nothing

The parameters that generated the synthetic data, when there are any.

A run should be scored against these rather than against its own stopping test —
though note the maximum-likelihood estimate for a finite sample differs from them
by the usual `O(M^{-1/2})` sampling error, so `‖β̂ − β*‖` does not go to zero at
the optimum.
"""
β_true(p::LogisticRegression) = p.β_star

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

"""
    batch_hess(p::LogisticRegression, x, batch)

`(1/N) Σ pₙ(1−pₙ) xₙxₙᵀ`, the exact Hessian.

Built with one `syrk`-style rank-`N` update rather than `N` `ger!` calls, each of
which previously allocated two copies of the row.
"""
function batch_hess(p::LogisticRegression, x, batch)
    N = length(batch)
    Xb = Matrix{Float64}(undef, N, p.n)
    for (j, i) in enumerate(batch)
        xi = view(p.X, i, :)
        q = _σ(dot(xi, x))
        w = sqrt(max(q * (1 - q), 0.0))
        @. Xb[j, :] = w * xi
    end
    H = (Xb' * Xb) ./ N
    return (H .+ H') ./ 2
end

# -----------------------------------------------------------------------------
# One-hidden-layer classifier
# -----------------------------------------------------------------------------

"""
    MLPClassifier(X, y, n_class; hidden = 16, λ = 0.0)

Softmax cross-entropy over a one-hidden-layer `tanh` network, with per-observation
scores so that BHHH applies.

`X` is `M × d`, `y` holds labels in `1:n_class`. The parameter vector packs
`(W₁, b₁, W₂, b₂)` in that order, so `n = hidden·d + hidden + n_class·hidden + n_class`.
`λ > 0` adds `λ‖θ‖²/2`.

!!! note "The penalty is not in the score matrix"
    `scores` returns the **likelihood** scores only; the `λθ` term is added to the
    gradient in `batch_grad!`, after averaging. Folding it into every column
    instead — which is what this file used to do — leaves the objective and
    gradient correct but makes BHHH form

        B = (1/N) Σ (sᵢ + λθ)(sᵢ + λθ)ᵀ,

    whose regularisation contribution is a rank-one `λ²θθᵀ` plus cross terms,
    where the true Hessian contribution is `λI`. The BHHH-versus-exact comparison
    on a regularised network then measures the discrepancy rather than the
    information identity. To regularise the *model*, use `BHHHModel(ridge = λ)`.

!!! warning "The information identity does not hold"
    This model is misspecified — the data were not generated by a tanh network — and
    the run is nowhere near "the true parameters", which do not exist. So BHHH's
    justification is absent, not merely weakened. `B` remains positive semidefinite
    and usable as a preconditioner, but it is not an approximation to `∇²f` and
    should not be described as one. [`information_identity_error`](@ref) reports
    the gap; on a network it does not shrink with the sample.

    Two consequences follow. `B ⪰ 0` means the model can never report negative
    curvature, so a run converges contentedly to saddles — which for a network are
    the dominant critical points. And `SecondOrder` over a `BHHHModel` gives
    `τ ≡ ‖g‖`, so a `:second_order` status certifies nothing; the solver now warns.
    Use `ExactHessian` when the second-order question is the question.

!!! note "Dimensions"
    A dense `n × n` Hessian needs `n²` entries: MNIST at `784 → 16 → 10` is 12 730
    parameters and 1.3 GB dense. Use `hessian_op` with [`SteihaugCG`](@ref), which
    needs only `B·v` and costs `n·N` for the score matrix. `dense_max` on the model
    controls the switch. `ExactMS` is out of the question at this scale, which is
    itself worth reporting: the subsolver axis is constrained by the model axis.
"""
struct MLPClassifier <: LikelihoodProblem
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
    λ >= 0 || throw(ArgumentError("MLPClassifier: need λ ≥ 0"))
    n = hidden * d + hidden + n_class * hidden + n_class
    return MLPClassifier(n, M, d, hidden, n_class, Matrix{Float64}(X),
                         Vector{Int}(y), float(λ))
end

population(p::MLPClassifier) = p.M

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

Per-observation **likelihood** gradients by explicit backpropagation. The L2
penalty is *not* included here; see the note on [`MLPClassifier`](@ref).

Standard backprop accumulates the *sum* over a batch; BHHH needs the terms kept
apart, so each column is formed from one forward and one backward pass. That is
the price of the outer product: the gradient itself would cost one batched pass,
and the score matrix costs `|batch|` unbatched ones plus `n·|batch|` storage.
"""
function scores(p::MLPClassifier, θ, batch)
    W1, b1, W2, b2 = _unpack(p, θ)
    S = zeros(p.n, length(batch))
    a1 = Vector{Float64}(undef, p.h); z2 = Vector{Float64}(undef, p.c)
    dz1 = Vector{Float64}(undef, p.h); dz2 = Vector{Float64}(undef, p.c)
    oW1 = 0; ob1 = p.h * p.d; oW2 = ob1 + p.h; ob2 = oW2 + p.c * p.h
    for (j, i) in enumerate(batch)
        xi = view(p.X, i, :)
        mul!(a1, W1, xi); a1 .+= b1; a1 .= tanh.(a1)
        mul!(z2, W2, a1); z2 .+= b2
        copyto!(dz2, z2); _softmax!(dz2); dz2[p.y[i]] -= 1.0        # ∂ℓ/∂z₂
        col = view(S, :, j)
        @views mul!(reshape(col[oW2+1:oW2+p.c*p.h], p.c, p.h), dz2, a1')
        @views col[ob2+1:ob2+p.c] .= dz2
        mul!(dz1, W2', dz2); @. dz1 *= (1 - a1^2)
        @views mul!(reshape(col[oW1+1:oW1+p.h*p.d], p.h, p.d), dz1, xi')
        @views col[ob1+1:ob1+p.h] .= dz1
    end
    return S
end

"""
    batch_grad!(p::MLPClassifier, θ, batch, g)

The mean of the likelihood scores, **plus** `λθ`.

Overrides the `ScoredProblem` default so that the penalty enters the gradient
once, after averaging, rather than being folded into every score column.

One batched forward and backward pass, not `|batch|` unbatched ones. This used to
call [`scores`](@ref) and sum the result, which built the whole `n × |batch|`
score matrix in order to reduce it away immediately — the cost [`scores`](@ref)'s
own docstring warns about, paid on every gradient rather than only where the
per-observation terms are actually wanted. On the `n = 1156` network of
experiment 11 that is 1.7 million entries materialised per gradient, and
`batch_hess` asks for `2n = 2312` gradients to build one Hessian.

Measured against the previous implementation: identical to `1.3e-15` relative,
and between `6.8` and `7.9` times faster. It also turns the inner work from
`|batch|` BLAS-2 calls into four BLAS-3 ones, which is what makes the layer
portable to an array type that is not a `Matrix{Float64}`.

!!! note "`scores` is unchanged and still needed"
    BHHH and BHHH-2 are outer products of the per-observation scores, so they
    need the columns kept apart. Only the *gradient*, which sums them, can take
    the batched route.
"""
function batch_grad!(p::MLPClassifier, θ, batch, g)
    W1, b1, W2, b2 = _unpack(p, θ)
    N  = length(batch)
    Xb = @view p.X[batch, :]                       # N × d

    A1 = W1 * transpose(Xb)                        # h × N
    A1 .+= b1
    A1 .= tanh.(A1)

    Z2 = W2 * A1                                   # c × N
    Z2 .+= b2
    Z2 .= exp.(Z2 .- maximum(Z2; dims = 1))        # same max-shift as `_softmax!`
    Z2 ./= sum(Z2; dims = 1)
    @inbounds for j in 1:N                         # dz₂ = softmax − onehot(y)
        Z2[p.y[batch[j]], j] -= 1.0
    end
    DZ2 = Z2                                       # c × N
    DZ1 = (transpose(W2) * DZ2) .* (1 .- A1 .^ 2)  # h × N

    o = 0
    copyto!(view(g, o+1 : o+p.h*p.d), vec(DZ1 * Xb));            o += p.h * p.d
    copyto!(view(g, o+1 : o+p.h),     vec(sum(DZ1; dims = 2)));  o += p.h
    copyto!(view(g, o+1 : o+p.c*p.h), vec(DZ2 * transpose(A1))); o += p.c * p.h
    copyto!(view(g, o+1 : o+p.c),     vec(sum(DZ2; dims = 2)))
    g ./= N
    p.λ > 0 && (g .+= p.λ .* θ)
    return nothing
end

"""
    batch_hess(p::MLPClassifier, θ, batch)

The exact Hessian by central differences of the analytic gradient.

`O(n)` gradient evaluations, so it is for the small configurations only — which is
exactly the regime where the comparison against BHHH is worth making, since it is
the only regime where the truth is available at all. `LikelihoodNLP` and
`SampledNLP` cache the result per iterate, so it is paid once per iteration rather
than once per Hessian-vector product.

Central rather than forward differences: the step `ε = ∛eps · max(1, ‖θ‖)` is the
optimal choice for the central formula, and was previously being used with the
one-sided one.
"""
function batch_hess(p::MLPClassifier, θ, batch)
    n = p.n
    n <= 2_000 || throw(ArgumentError(
        "MLPClassifier: exact Hessian at n = $n is impractical (O(n) gradient " *
        "evaluations and an n×n matrix). Use BHHHModel/GaussNewtonModel with " *
        "SteihaugCG, or shrink `hidden` / the input dimension."))
    H = zeros(n, n)
    ε = cbrt(eps()) * max(1.0, norm(θ))
    gp = zeros(n); gm = zeros(n); θw = Vector{Float64}(θ)
    for j in 1:n
        old = θw[j]
        θw[j] = old + ε; batch_grad!(p, θw, batch, gp)
        θw[j] = old - ε; batch_grad!(p, θw, batch, gm)
        θw[j] = old
        @. H[:, j] = (gp - gm) / (2ε)
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
    W1, _, W2, _ = _unpack(p, θ)
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

# -----------------------------------------------------------------------------
# Multinomial logit
# -----------------------------------------------------------------------------

"""
    MultinomialLogit(X, y)
    MultinomialLogit(; K = 10, M = 100_000, J = 5, β_star = nothing, seed = 0,
                       signal = 1.0, noise = 0.0)

Multinomial logit over `J` alternatives with utilities linear in the parameters.

Individual `n` faces `J` alternatives; alternative `j` carries the attribute vector
`x_{nj} ∈ R^K` and the representative utility `V_{nj} = β' x_{nj}`. Under
independent Gumbel errors the choice probabilities are

```math
P_{nj} = \\frac{e^{\\beta^\\top x_{nj}}}{\\sum_{l} e^{\\beta^\\top x_{nl}}} ,
```

and the objective is the average negative log-likelihood

```math
f(\\beta) = -\\frac1M \\sum_{n=1}^{M} \\sum_{j=1}^{J} y_{nj} \\ln P_{nj} ,
\\qquad y_{nj} = 1 \\text{ iff } n \\text{ chose } j .
```

`X` is `M × J × K` and `y` holds the chosen alternative in `1:J`. There is no
alternative-specific constant: the utility is exactly `β' x_{nj}`, with one generic
coefficient vector shared across alternatives.

This is the genuine multiple-alternative case, not `J` binary fits.
[`LogisticRegression`](@ref) is the binary problem in its own parameterisation and
stays the cheaper choice when `J = 2`.

# Scores and curvature

Writing `p_n` for the probability vector and `x̄_n = Σ_j P_{nj} x_{nj}` for the
probability-weighted mean attribute, the per-observation score and Hessian are

```math
s_n = \\bar x_n - x_{n c_n} ,
\\qquad
H_n = \\sum_j P_{nj} x_{nj} x_{nj}^\\top - \\bar x_n \\bar x_n^\\top ,
```

so `H_n` is the covariance of the attribute vector under the choice probabilities:
positive semidefinite, and singular exactly when the attributes carry no variation
across alternatives that the current `β` can see. The objective is therefore convex
in `β`, which the binary case shares.

# The information identity

At `β*` with `noise = 0` the data were generated by the very model being fitted, so
`E[s s'] = -E[H]` in the population and BHHH is justified. In a sample of `M` the
two agree to `O(M^{-1/2})`; [`information_identity_error`](@ref) measures the gap.

!!! warning "`noise > 0` breaks it, and the break is the point"
    `noise = σ` draws a **separate** coefficient vector `β_n = β* + ε_n` with
    `ε_n ~ N(0, σ²I)` for each individual and generates that individual's choice
    from it. The mean perturbation is zero, so `E[β_n] = β*`, but the generating
    process is then a **mixed logit** while the fitted model is a plain logit. That
    is misspecification: the identity does not hold at any `β`, and the gap does not
    shrink with `M`.

    The default is `noise = 0.0`, which is the correctly specified case. Set it
    deliberately, and read [`information_identity_error`](@ref) rather than assuming
    either way.

# Cost

`X` holds `M·J·K` entries: the `100_000 × 5 × 10` default is 40 MB. `loss_terms`
and `scores` cost `O(N·J·K)` over a batch of `N` individuals, `batch_hess` costs
`O(N·J·K²)`.
"""
struct MultinomialLogit <: LikelihoodProblem
    n::Int                      # K, the number of parameters
    M::Int                      # individuals
    J::Int                      # alternatives
    X::Array{Float64, 3}        # M × J × K
    y::Vector{Int}              # chosen alternative, in 1:J
    β_star::Union{Vector{Float64}, Nothing}
    noise::Float64              # the σ used to generate, 0 when specified correctly
end

function MultinomialLogit(X::AbstractArray{<:Real, 3}, y::AbstractVector{<:Integer};
                          β_star = nothing, noise::Real = 0.0)
    M, J, K = size(X)
    M == length(y) || throw(DimensionMismatch(
        "MultinomialLogit: X has $M individuals but y has $(length(y))"))
    J >= 2 || throw(ArgumentError("MultinomialLogit: need J ≥ 2 alternatives, got $J"))
    all(v -> 1 <= v <= J, y) || throw(ArgumentError(
        "MultinomialLogit: choices must lie in 1:$J"))
    return MultinomialLogit(K, M, J, Array{Float64, 3}(X), Vector{Int}(y),
                            β_star === nothing ? nothing : Vector{Float64}(β_star),
                            float(noise))
end

function MultinomialLogit(; K::Int = 10, M::Int = 100_000, J::Int = 5,
                            β_star::Union{AbstractVector, Nothing} = nothing,
                            seed::Int = 0, signal::Real = 1.0, noise::Real = 0.0)
    K >= 1 || throw(ArgumentError("MultinomialLogit: need K ≥ 1"))
    J >= 2 || throw(ArgumentError("MultinomialLogit: need J ≥ 2"))
    M >= 1 || throw(ArgumentError("MultinomialLogit: need M ≥ 1"))
    noise >= 0 || throw(ArgumentError("MultinomialLogit: need noise ≥ 0"))
    rng = MersenneTwister(seed)
    β = β_star === nothing ? signal .* randn(rng, K) : Vector{Float64}(β_star)
    length(β) == K || throw(DimensionMismatch(
        "MultinomialLogit: β_star must have length K = $K"))
    X = randn(rng, M, J, K)
    y = Vector{Int}(undef, M)
    V = Vector{Float64}(undef, J)
    βn = Vector{Float64}(undef, K)
    for i in 1:M
        # A mean-zero perturbation of β* per individual. At noise = 0 this is β*
        # itself and the model is correctly specified.
        if noise > 0
            @inbounds for k in 1:K
                βn[k] = β[k] + noise * randn(rng)
            end
        else
            copyto!(βn, β)
        end
        @inbounds for j in 1:J
            acc = 0.0
            for k in 1:K
                acc += X[i, j, k] * βn[k]
            end
            V[j] = acc
        end
        _softmax!(V)
        u = rand(rng); c = J; acc = 0.0
        @inbounds for j in 1:J
            acc += V[j]
            if u < acc
                c = j
                break
            end
        end
        y[i] = c
    end
    return MultinomialLogit(K, M, J, X, y, β, float(noise))
end

population(p::MultinomialLogit) = p.M

"""
    β_true(p::MultinomialLogit) -> Vector or nothing

The generating coefficients. With `noise > 0` these are the **mean** of the
per-individual coefficients, not the minimiser of `f`: the plain-logit maximum
likelihood estimate of a mixed-logit population is not `β*` even as `M → ∞`.
"""
β_true(p::MultinomialLogit) = p.β_star

"Utilities and choice probabilities for individual `i` at `β`, written into `P`."
@inline function _mnl_probs!(P, p::MultinomialLogit, β, i::Int)
    @inbounds for j in 1:p.J
        acc = 0.0
        for k in 1:p.n
            acc += p.X[i, j, k] * β[k]
        end
        P[j] = acc
    end
    return _softmax!(P)
end

function loss_terms(p::MultinomialLogit, β, batch)
    out = Vector{Float64}(undef, length(batch))
    V = Vector{Float64}(undef, p.J)
    for (t, i) in enumerate(batch)
        @inbounds for j in 1:p.J
            acc = 0.0
            for k in 1:p.n
                acc += p.X[i, j, k] * β[k]
            end
            V[j] = acc
        end
        m = maximum(V)
        # −log P_{i,y_i} = logsumexp(V) − V[y_i], shifted for stability.
        se = 0.0
        @inbounds for j in 1:p.J
            se += exp(V[j] - m)
        end
        out[t] = m + log(se) - V[p.y[i]]
    end
    return out
end

"""
    scores(p::MultinomialLogit, β, batch) -> Matrix (K × |batch|)

Column `t` is `x̄_n − x_{n c_n}`, the gradient of the negative log-likelihood of
individual `n = batch[t]`. Its mean over the population is zero at the true
parameters, which is the first half of the information identity.
"""
function scores(p::MultinomialLogit, β, batch)
    S = Matrix{Float64}(undef, p.n, length(batch))
    P = Vector{Float64}(undef, p.J)
    for (t, i) in enumerate(batch)
        _mnl_probs!(P, p, β, i)
        c = p.y[i]
        @inbounds for k in 1:p.n
            acc = 0.0
            for j in 1:p.J
                acc += P[j] * p.X[i, j, k]
            end
            S[k, t] = acc - p.X[i, c, k]
        end
    end
    return S
end

"""
    batch_hess(p::MultinomialLogit, β, batch)

`(1/N) Σ_n [ Σ_j P_{nj} x_{nj} x_{nj}' − x̄_n x̄_n' ]`, the exact Hessian.

Each summand is a covariance under the choice probabilities, so the result is
positive semidefinite and the problem is convex. Formed as a weighted Gram matrix
minus a rank-one correction per individual, at `O(N·J·K²)`.
"""
function batch_hess(p::MultinomialLogit, β, batch)
    K, N = p.n, length(batch)
    H = zeros(K, K)
    P = Vector{Float64}(undef, p.J)
    xbar = Vector{Float64}(undef, K)
    for i in batch
        _mnl_probs!(P, p, β, i)
        fill!(xbar, 0.0)
        @inbounds for j in 1:p.J
            w = P[j]
            for k in 1:K
                xbar[k] += w * p.X[i, j, k]
            end
        end
        @inbounds for j in 1:p.J
            w = P[j]
            w == 0 && continue
            for a in 1:K
                xa = w * p.X[i, j, a]
                for b in a:K
                    H[a, b] += xa * p.X[i, j, b]
                end
            end
        end
        @inbounds for a in 1:K, b in a:K
            H[a, b] -= xbar[a] * xbar[b]
        end
    end
    @inbounds for a in 1:K, b in 1:a-1
        H[a, b] = H[b, a]
    end
    H ./= N
    return (H .+ H') ./ 2
end

"""
    init_params(p::MultinomialLogit) -> Vector

`zeros(K)`, the point at which every alternative is equally likely.

The PreDoc's logit runs start from `x₀ = 0`, and it is the natural origin here: at
`β = 0` all `P_{nj} = 1/J`, so `f = log J` whatever the data, and the gradient is
the difference between the mean attribute and the mean chosen attribute.
"""
init_params(p::MultinomialLogit) = zeros(p.n)

"""
    accuracy(p::MultinomialLogit, β, batch = 1:p.M) -> Float64

Fraction of `batch` whose chosen alternative has the largest utility at `β`.

Reported alongside the loss because the loss is not the quantity a discrete-choice
model is judged on. `1/J` is the value at `β = 0`.
"""
function accuracy(p::MultinomialLogit, β, batch = 1:p.M)
    correct = 0
    P = Vector{Float64}(undef, p.J)
    for i in batch
        _mnl_probs!(P, p, β, i)
        correct += (argmax(P) == p.y[i])
    end
    return correct / length(batch)
end
