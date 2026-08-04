# =============================================================================
# src/Likelihood/least_squares.jl
#
# Concrete nonlinear least squares, so that `GaussNewtonModel` has something to
# run on and the three worked examples of Part III — least squares, likelihood,
# classification — all go through the same solver and the same sampling layer.
#
#     f(x) = (1/2M) Σₙ rₙ(x)²,   rₙ(x) = φ(tₙ, x) − yₙ .
#
# `LeastSquares` is a `ScoredProblem`, so it is simultaneously
#   * an ordinary objective, via `LikelihoodNLP`;
#   * a sampled objective, via `SampledNLP` with any sampling rule;
#   * a source of per-observation scores, so BHHH applies as well as Gauss–Newton
#     and the two outer-product approximations can be compared on one problem.
# =============================================================================

"""
    LeastSquares(t, y, φ, ∇φ, n; x_true = nothing)

Nonlinear least squares with `M` observations and `n` parameters:

- `t` is `M × d`, row `n` being the design point `tₙ`;
- `φ(tₙ, x) -> Real` is the model;
- `∇φ(tₙ, x) -> AbstractVector` of length `n` is its gradient in `x`.

The per-observation score is `rₙ ∇φₙ` and the Jacobian row is `∇φₙᵀ`, so
[`GaussNewtonModel`](@ref), [`BHHHModel`](@ref) and [`ExactHessian`](@ref) can all
be run against the same problem — which is the comparison worth making, since
Gauss–Newton and BHHH discard *different* terms and their failure modes separate
on residual size.

Use [`linear_least_squares`](@ref) or [`exponential_fit`](@ref) for the two cases
this package exercises.
"""
struct LeastSquares{F, J} <: NLSProblem
    n::Int
    M::Int
    t::Matrix{Float64}
    y::Vector{Float64}
    φ::F
    ∇φ::J
    x_true::Union{Vector{Float64}, Nothing}
end

function LeastSquares(t::AbstractMatrix, y::AbstractVector, φ, ∇φ, n::Int;
                      x_true = nothing)
    size(t, 1) == length(y) ||
        throw(DimensionMismatch("LeastSquares: rows of t must match y"))
    return LeastSquares{typeof(φ), typeof(∇φ)}(
        n, length(y), Matrix{Float64}(t), Vector{Float64}(y), φ, ∇φ,
        x_true === nothing ? nothing : Vector{Float64}(x_true))
end

n_terms(p::LeastSquares) = p.M

"""
    x_true(prob) -> Vector or nothing

The parameters that generated the data, when the problem was constructed
synthetically. As with [`β_true`](@ref), the least-squares estimate for a finite
sample differs from them by the sampling error, so `‖x̂ − x*‖` does not vanish at
the optimum.
"""
x_true(p::LeastSquares) = p.x_true

residuals(p::LeastSquares, x, batch) =
    [p.φ(view(p.t, i, :), x) - p.y[i] for i in batch]

function jacobian(p::LeastSquares, x, batch)
    J = Matrix{Float64}(undef, length(batch), p.n)
    for (j, i) in enumerate(batch)
        J[j, :] .= p.∇φ(view(p.t, i, :), x)
    end
    return J
end

# `loss_terms` and `scores` come from the NLSProblem defaults: rₙ²/2 and rₙ∇φₙ.

"""
    batch_hess(p::LeastSquares, x, batch)

The exact Hessian `(1/N)(JᵀJ + Σ rₙ ∇²rₙ)`, by central differences of the
analytic gradient.

Available so that the Gauss–Newton approximation can be scored against the truth
rather than assumed adequate: the discarded term is exactly the difference, and it
is what separates a small-residual problem from a large-residual one.
"""
function batch_hess(p::LeastSquares, x, batch)
    n = p.n
    H = zeros(n, n)
    ε = cbrt(eps()) * max(1.0, norm(x))
    gp = zeros(n); gm = zeros(n); xw = Vector{Float64}(x)
    for j in 1:n
        old = xw[j]
        xw[j] = old + ε; batch_grad!(p, xw, batch, gp)
        xw[j] = old - ε; batch_grad!(p, xw, batch, gm)
        xw[j] = old
        @. H[:, j] = (gp - gm) / (2ε)
    end
    return (H .+ H') ./ 2
end

true_objective(p::LeastSquares, x) = batch_obj(p, x, 1:p.M)
true_gradient(p::LeastSquares, x)  = (g = zeros(p.n); batch_grad!(p, x, 1:p.M, g); g)

"""
    gauss_newton_error(p::LeastSquares, x; batch = all) -> NamedTuple

How much the Gauss–Newton approximation discards at `x`:

```julia
(GN_err   = ‖JᵀJ/N − ∇²f‖ / ‖∇²f‖,
 residual = ‖r‖ / √N,          # the RMS residual, which controls the gap
 rank_def = N < n)             # JᵀJ singular by construction when it is
```

The counterpart of [`information_identity_error`](@ref) for the other
outer-product model. The discarded term is `(1/N)Σ rₙ ∇²rₙ`, so `GN_err` should
track `residual`: small at a well-fitting solution, `O(1)` when the model cannot
fit the data. That is a different failure mode from BHHH's — it depends on the
*fit*, not on the *specification* — and it is worth measuring rather than
assuming, for the same reason.
"""
function gauss_newton_error(p::LeastSquares, x; batch = 1:n_terms(p))
    b = collect(batch)
    J = jacobian(p, x, b); r = residuals(p, x, b); N = length(b)
    GN = (J' * J) ./ N
    H  = batch_hess(p, x, b)
    return (GN_err = norm(GN .- H) / max(norm(H), eps()),
            residual = norm(r) / sqrt(N),
            rank_def = N < p.n)
end

# -----------------------------------------------------------------------------
# Constructors for the two cases used in the experiments
# -----------------------------------------------------------------------------

"""
    linear_least_squares(A, b; x_true = nothing)
    linear_least_squares(; n = 5, M = 2_000, noise = 0.1, seed = 0)

`rₙ(x) = aₙᵀx − bₙ`, so `∇²rₙ = 0` and **Gauss–Newton is exact**:
`JᵀJ/M = ∇²f` identically, with no discarded term at all.

That makes it the control for the least-squares half of the comparison, exactly as
a correctly specified logistic regression is the control for the likelihood half.
`gauss_newton_error` should return zero to rounding here whatever the residual
size, which is the property that distinguishes "Gauss–Newton is exact" from
"Gauss–Newton is accurate because the residuals are small".

The keyword form generates `A` with independent normal entries and
`b = A x* + noise·ε`.
"""
function linear_least_squares(A::AbstractMatrix, b::AbstractVector; x_true = nothing)
    n = size(A, 2)
    φ(a, x)  = dot(a, x)
    ∇φ(a, x) = Vector(a)
    return LeastSquares(Matrix(A), Vector(b), φ, ∇φ, n; x_true = x_true)
end

function linear_least_squares(; n::Int = 5, M::Int = 2_000, noise::Real = 0.1,
                                seed::Int = 0)
    rng = MersenneTwister(seed)
    A = randn(rng, M, n)
    x⋆ = randn(rng, n)
    b = A * x⋆ .+ noise .* randn(rng, M)
    return linear_least_squares(A, b; x_true = x⋆)
end

"""
    exponential_fit(; n_terms_model = 2, M = 400, noise = 0.05, misfit = 0.0, seed = 0)

Sum-of-exponentials fitting, the classical nonlinear least-squares testbed:

```math
\\varphi(t, x) = \\sum_{j=1}^{J} x_{2j-1}\\, e^{-x_{2j} t},
\\qquad n = 2J .
```

`misfit > 0` adds a component to the data that the model cannot represent — a term
`misfit·sin(3t)` — which raises the residual at the solution without changing the
parameterisation. That knob is the point of this problem: it moves the fit
continuously from small-residual to large-residual, and

- `misfit = 0`: residuals are noise-sized, `Σ rₙ∇²rₙ` is small, Gauss–Newton is a
  good approximation and converges superlinearly;
- `misfit` large: the residuals do not vanish, the discarded term is `O(1)`, and
  Gauss–Newton degrades to linear convergence while still producing descent
  directions and healthy ratios.

So it exhibits the same pattern as BHHH under misspecification — a positive
semidefinite model whose justification has quietly gone — arrived at from the
other direction. `gauss_newton_error` measures it.
"""
function exponential_fit(; n_terms_model::Int = 2, M::Int = 400, noise::Real = 0.05,
                           misfit::Real = 0.0, seed::Int = 0)
    J = n_terms_model
    n = 2J
    rng = MersenneTwister(seed)
    t = reshape(collect(range(0.0, 4.0; length = M)), M, 1)

    x⋆ = zeros(n)
    for j in 1:J
        x⋆[2j - 1] = 1.0 + 0.5 * (j - 1)          # amplitudes
        x⋆[2j]     = 0.5 * j                       # decay rates, well separated
    end

    function φ(tr, x)
        τ = tr[1]; s = 0.0
        for j in 1:J
            s += x[2j - 1] * exp(-x[2j] * τ)
        end
        return s
    end
    function ∇φ(tr, x)
        τ = tr[1]; g = Vector{Float64}(undef, 2J)
        for j in 1:J
            e = exp(-x[2j] * τ)
            g[2j - 1] = e
            g[2j]     = -x[2j - 1] * τ * e
        end
        return g
    end

    y = [φ(view(t, i, :), x⋆) + misfit * sin(3 * t[i, 1]) + noise * randn(rng)
         for i in 1:M]
    return LeastSquares(t, y, φ, ∇φ, n; x_true = x⋆)
end
