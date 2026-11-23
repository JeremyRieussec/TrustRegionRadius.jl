# =============================================================================
# src/Likelihood/linear_regression.jl
#
# Linear regression with Gaussian errors of known scale, posed as a likelihood.
#
# `linear_least_squares` already minimises (1/2M) Σ (xₙᵀβ − yₙ)², which is the
# Gaussian likelihood with the error standard deviation fixed at 1 whatever
# generated the data. BHHH is not invariant to that scale: dividing the objective
# by σ² divides the Hessian by σ² and the outer product of the scores by σ⁴. The
# information identity can therefore hold for one σ only, and this type carries it.
#
# What earns the problem its place is that everything is known in closed form.
# The Hessian is the constant XᵀX/(Mσ²), the minimiser is the least-squares
# estimate, and the BHHH error is (1/N) Σ (rₙ² − 1) xₙxₙᵀ/σ² at every β, with rₙ
# the standardised residual. The identity is then a statement about the second
# moment of the errors that can be checked exactly and not only as a rate.
#
# The type is an `NLSProblem`, with the standardised residual as its residual, so
# the loss and the scores come from the NLSProblem defaults and Gauss-Newton is
# legal beside BHHH. Every residual is linear in β, so Gauss-Newton is exact here.
# =============================================================================

"""
    LinearRegression(X, y; σ, β_star = nothing, noise = 0.0)
    LinearRegression(; K = 5, M = 2_000, β_star = nothing, seed = 0, intercept = true,
                       signal = 1.0, noise = 0.0, σ = 1.0)

Linear regression with Gaussian errors of known standard deviation `σ`, posed as a
likelihood: minimise

```math
f(\\beta) = \\frac{1}{2M} \\sum_n \\frac{(x_n^\\top\\beta - y_n)^2}{\\sigma^2} ,
```

the average negative log-likelihood of `yₙ | xₙ ~ N(xₙᵀβ, σ²)` less the constant
`½ ln(2πσ²)`. `X` is `M × K` and `y` holds real outcomes.

`σ` is a constant of the model and is not estimated. It has to be carried because
BHHH is not invariant to the scale of the objective. Dividing `f` by `σ²` divides
the Hessian by `σ²` and the outer product of the scores by `σ⁴`, so the two can
agree for one value of `σ` only, the standard deviation of the errors. With observed
data `σ` must be given, and there is no default. A value estimated beforehand may be
passed, for instance `sqrt(sum(abs2, y - X * (X \\ y)) / M)`.

The keyword form generates `yₙ = xₙᵀβ* + σ eₙ` with `eₙ ~ N(0, 1)`, so with
`noise = 0` the model is **correctly specified**. `xₙ ~ N(0, I)`, with the first
coordinate set to 1 if `intercept`. The generating coefficients are retrieved with
[`β_true`](@ref).

For a given `K`, `M` and `seed` the design matrix and `β*` are those of
[`LogisticRegression`](@ref) and [`Probit`](@ref) with the same arguments. At
`σ = 1` the outcome is the latent variable whose sign is the probit outcome, so the
three likelihoods can be compared on one design.

# Scores and curvature

The standardised residual is `rₙ = (xₙᵀβ − yₙ)/σ`, the loss of one observation is
`rₙ²/2`, and the per-observation score and Hessian are

```math
s_n = \\frac{(x_n^\\top\\beta - y_n)\\, x_n}{\\sigma^2} ,
\\qquad
H_n = \\frac{x_n x_n^\\top}{\\sigma^2} .
```

`Hₙ` involves neither `β` nor `yₙ`. The exact Hessian is the constant `XᵀX/(Mσ²)`,
and `X_bᵀX_b/(Nσ²)` on a batch of `N`. The objective is a convex quadratic, and its
minimiser is the least-squares estimate `X \\ y` whatever `σ` is. Every `rₙ` is
linear in `β`, so [`GaussNewtonModel`](@ref) discards nothing and is the exact
Hessian. `batch_hess` forms it as `JᵀJ/N` from the Jacobian of the standardised
residuals, the expression `GaussNewtonModel` evaluates, so without a ridge the two
matrices are equal to the last bit.

# The BHHH error in closed form

With `B = (1/N) Σ sₙsₙᵀ` and `H` the Hessian on the same batch, at any `β` and for
any data,

```math
B - H = \\frac1N \\sum_n \\bigl(r_n^2 - 1\\bigr)\\, \\frac{x_n x_n^\\top}{\\sigma^2} .
```

The information identity is therefore a statement about the second moment of the
standardised residuals, and normality plays no part in it. Suppose the data are
`yₙ = xₙᵀβ* + uₙ` with `E[uₙ | xₙ] = 0` and `Var(uₙ | xₙ) = vₙ`. At `β = β* + δ`,

```math
E\\bigl[B - H \\mid X\\bigr] = \\frac{1}{M\\sigma^4} \\sum_n
    \\bigl[(x_n^\\top\\delta)^2 + v_n - \\sigma^2\\bigr]\\, x_n x_n^\\top .
```

- At `β*` with `vₙ = σ²` the right-hand side is zero for every `M`. What remains is
  sampling error, of relative size `√(2 Σ‖xₙ‖⁴) / ‖XᵀX‖` in root mean square under
  Gaussian errors, which decays like `M^{-1/2}`.
- Away from `β*` the term `(xₙᵀδ)²` is left. It is positive semidefinite and does
  not shrink with `M`, so BHHH overstates the curvature, by an amount that grows
  like `‖δ‖²/σ²`.
- A wrong `σ` leaves `vₙ − σ²`. Data whose errors have variance `σ₀²`, posed with
  `σ`, give `E[B | X] = (σ₀²/σ²) H` at `β*`.
- At `β = 0`, where a run starts by default, the residual is the outcome and the
  gap is of the order of the signal-to-noise ratio `mean((xₙᵀβ*)²)/σ²` times the
  Hessian. On the generated design that ratio is close to `‖β*‖²/σ²`. The two
  binary models have an exact BHHH model at the origin, and this one does not.

[`information_identity_error`](@ref) measures all of this against the closed-form
Hessian.

!!! warning "`noise > 0` breaks the identity and leaves `β*` where it is"
    `noise = ν` draws a separate coefficient vector `βₙ = β* + εₙ` with
    `εₙ ~ N(0, ν²I)` for each observation, as in [`Probit`](@ref). The errors then
    have variance `σ² + ν²‖xₙ‖²` while the model holds it at `σ²`. At `β*` the
    expected gap is `(ν²/(Mσ⁴)) Σ ‖xₙ‖² xₙxₙᵀ` at every sample size. The mean of
    `yₙ` given `xₙ` is still `xₙᵀβ*`, so [`β_true`](@ref) remains the minimiser as
    `M → ∞`. In the probit the same keyword moves the minimiser as well.

    Part of that gap is a wrong scale and not heteroscedasticity. With `m` the mean
    of `‖xₙ‖²`, it is `(ν²m/σ²) H` plus a remainder weighted by `‖xₙ‖² − m`. The
    first part is what a wrong `σ` produces, and posing the same data with the
    scale `√(RSS/M)` of their residuals removes it. The remainder is the
    heteroscedasticity. On the generated design the first part is the larger one
    from three coefficients on, and without an intercept the remainder is `2/K`
    of it. The remainder vanishes where `‖xₙ‖` does not vary: with `K = 1` and an
    intercept every `xₙ` is 1, and the noisy data are a correctly specified
    regression of standard deviation `√(σ² + ν²)`.

!!! warning "`noise` is not the error standard deviation"
    In [`linear_least_squares`](@ref) the keyword `noise` is the standard deviation
    of the errors. Here that is `σ`, and `noise` is the misspecification knob of the
    likelihood siblings. `linear_least_squares(A, b)` is this problem with `σ = 1`,
    and the losses and scores of `LinearRegression(A, b; σ = 1)` are the same
    numbers. When the errors do not have unit variance, its outer product at the
    solution is their variance times the Hessian. Carrying `σ` is what repairs that.
    The generating coefficients are read with [`β_true`](@ref) here, as for the
    likelihood siblings, and [`x_true`](@ref) is not defined.

!!! note "Under sampling an exact model has a ratio of 1"
    The objective on a batch is itself a quadratic with Hessian `X_bᵀX_b/(Nσ²)`.
    With [`ExactHessian`](@ref) or [`GaussNewtonModel`](@ref) the model is the batch
    objective. Whenever the objective and the gradient share the batch, the ratio is
    then 1 to rounding and says nothing about the sample. Four things follow.

    - Every step is read as fully successful. Under [`RDelta`](@ref) and
      [`RDeltaStep`](@ref) the radius is then multiplied by `γ3` at every iteration
      and never shrinks, so give the rule a `Δmax`. Under [`RRTR`](@ref) it never
      shrinks either, and grows only with the longest step taken. The other rules
      set it from the step, from the norm of the batch gradient, or from a
      comparison of the radius with that norm, and still move it both ways.
    - A step that solves the subproblem inside the region lands on the least-squares
      point of the batch, where the batch gradient vanishes, and a stopping test
      read on the batch is then met far from the solution. [`ExactMS`](@ref) does
      this at any batch size. [`SteihaugCG`](@ref) does it when its recurrence runs
      to termination, which takes few observations or few coefficients. Confirm
      the stop with `TRParams(true_stop = :full)`.
    - With `ExactMS`, a rule that scales the radius by the gradient norm reads that
      vanished batch gradient and sets the radius to rounding. A run whose stop is
      confirmed can then end `:stalled` far from the solution.
    - With `ExactMS` and fewer observations than coefficients, the batch Hessian is
      singular and the step takes a component as long as the radius in its null
      space, so the iterates leave. A ridge repairs that case. A batch of about `K`
      observations is not singular and can be nearly so, which a ridge does not
      repair. Use batches of several times `K` observations, or a `Δmax`.

# What a finite sum rules out

Every sampling rule draws observations, so the objective has to be a mean of
per-observation terms. Concentrating `σ` out gives `½ ln(RSS(β)/M)`, the logarithm
of a sum and not a sum: it has no per-observation terms and its value on a batch is
biased. The same holds for `RSS/(M − K)`, for a `σ` re-estimated during a run, and
for correlated errors. Known unequal standard deviations `s` are covered by scaling
the rows, `LinearRegression(X ./ s, y ./ s; σ = 1)`.
"""
struct LinearRegression <: NLSProblem
    n::Int
    M::Int
    X::Matrix{Float64}          # M × n
    y::Vector{Float64}
    β_star::Union{Vector{Float64}, Nothing}
    noise::Float64              # the ν used to generate, 0 when specified correctly
    σ::Float64                  # the error standard deviation the model holds fixed
end

function LinearRegression(X::AbstractMatrix, y::AbstractVector; σ::Real,
                          β_star = nothing, noise::Real = 0.0)
    size(X, 1) == length(y) || throw(DimensionMismatch(
        "LinearRegression: X has $(size(X, 1)) rows but y has $(length(y)) entries"))
    # The checks read what is stored. A number outside the Float64 range is finite
    # as given and is not once converted.
    yf = Vector{Float64}(y)
    σf = Float64(σ)
    νf = Float64(noise)
    all(isfinite, yf) || throw(ArgumentError("LinearRegression: y must be finite"))
    0 < σf < Inf || throw(ArgumentError("LinearRegression: need 0 < σ < Inf, got $σ"))
    νf >= 0 || throw(ArgumentError("LinearRegression: need noise ≥ 0"))
    β_star === nothing || length(β_star) == size(X, 2) || throw(DimensionMismatch(
        "LinearRegression: β_star must have length $(size(X, 2))"))
    return LinearRegression(size(X, 2), size(X, 1), Matrix{Float64}(X), yf,
                            β_star === nothing ? nothing : Vector{Float64}(β_star),
                            νf, σf)
end

function LinearRegression(; K::Int = 5, M::Int = 2_000,
                            β_star::Union{AbstractVector, Nothing} = nothing,
                            seed::Int = 0, intercept::Bool = true,
                            signal::Real = 1.0, noise::Real = 0.0, σ::Real = 1.0)
    K >= 1 || throw(ArgumentError("LinearRegression: need K ≥ 1"))
    M >= 1 || throw(ArgumentError("LinearRegression: need M ≥ 1"))
    σf = Float64(σ)
    νf = Float64(noise)
    νf >= 0 || throw(ArgumentError("LinearRegression: need noise ≥ 0"))
    0 < σf < Inf || throw(ArgumentError("LinearRegression: need 0 < σ < Inf, got $σ"))
    rng = MersenneTwister(seed)
    # β*, then X, then the coefficient perturbations, then the errors: the order in
    # which Probit draws them. One seed then gives the three likelihoods the same
    # design, and at σ = 1 the outcome here is the probit's latent variable.
    β = β_star === nothing ? Float64(signal) .* randn(rng, K) : Vector{Float64}(β_star)
    length(β) == K || throw(DimensionMismatch(
        "LinearRegression: β_star must have length K = $K"))
    X = randn(rng, M, K)
    intercept && (X[:, 1] .= 1.0)
    z = X * β
    # βₙ = β* + νεₙ adds ν xₙᵀεₙ to the outcome. At noise = 0 nothing is drawn and
    # the model is correctly specified.
    νf > 0 && (z .+= νf .* vec(sum(X .* randn(rng, M, K); dims = 2)))
    y = z .+ σf .* randn(rng, M)
    all(isfinite, y) || throw(ArgumentError(
        "LinearRegression: the generated outcomes are not finite, check signal, " *
        "noise, σ and β_star"))
    return LinearRegression(K, M, X, y, β, νf, σf)
end

population(p::LinearRegression) = p.M

"""
    β_true(p::LinearRegression) -> Vector or nothing

The generating coefficients. With `noise > 0` they are the mean of the
per-observation coefficients and remain the minimiser of `f` as `M → ∞`, because
the errors keep mean zero given `x`. The minimiser in the sample is `p.X \\ p.y`.
"""
β_true(p::LinearRegression) = p.β_star

# The standardised residual, fitted minus observed as in `LeastSquares`, and its
# Jacobian X/σ. `loss_terms` and `scores` come from the NLSProblem defaults: rₙ²/2
# and rₙ∇rₙ = (xₙᵀβ − yₙ) xₙ/σ², which is the score of the Gaussian likelihood.
function residuals(p::LinearRegression, x, batch)
    r = Vector{Float64}(undef, length(batch))
    for (j, i) in enumerate(batch)
        r[j] = (dot(view(p.X, i, :), x) - p.y[i]) / p.σ
    end
    return r
end

function jacobian(p::LinearRegression, x, batch)
    J = Matrix{Float64}(undef, length(batch), p.n)
    for (j, i) in enumerate(batch)
        @views J[j, :] .= p.X[i, :] ./ p.σ
    end
    return J
end

"""
    batch_hess(p::LinearRegression, x, batch)

`X_bᵀX_b/(Nσ²)`, the exact Hessian, which involves neither `x` nor the outcomes.

Formed as `JᵀJ/N` from the Jacobian of the standardised residuals. That is the
expression [`GaussNewtonModel`](@ref) evaluates, so without a ridge the two
matrices are equal to the last bit and not only to rounding.
"""
function batch_hess(p::LinearRegression, x, batch)
    J = jacobian(p, x, batch)
    H = (J' * J) ./ size(J, 1)
    return (H .+ H') ./ 2
end
