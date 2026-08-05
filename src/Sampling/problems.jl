# =============================================================================
# src/Sampling/problems.jl
#
# Concrete sampled problems, in the two classes.
#
#   ExpectationProblem       PerturbedExpectation
#   FiniteSumProblem         FiniteSum, PerturbedSum
#     └── ScoredProblem      (LogisticRegression, MLPClassifier, LeastSquares —
#                             in Likelihood/)
#
# A "batch" is whatever the class means by one:
#
#   finite sum   a Vector{Int} of indices into 1:M
#   expectation  an object holding N freshly drawn realisations
#
# Both flow through the same `batch_obj` / `batch_grad!` / `batch_hess`
# interface, so the sampled solvers do not branch on the class for evaluation —
# only for how the batch is produced and how large it is allowed to be.
# =============================================================================

# -----------------------------------------------------------------------------
# The evaluation interface, common to both sampled classes
# -----------------------------------------------------------------------------

"""
    batch_obj(prob, x, batch) -> Float64

Estimate `f(x)` from one batch.

Must be a **deterministic function of `(x, batch)`**: the same batch is reused for
`f̂(x_k)` and `f̂(x_k + s_k)` within an iteration, and that is what makes the noise in
the difference shrink with `‖s_k‖`. An implementation that draws randomness inside
this function defeats it.
"""
function batch_obj end

"""
    batch_grad!(prob, x, batch, g) -> nothing

Estimate `∇f(x)` from one batch, writing it into `g`. Deterministic in
`(x, batch)`, for the reason given under [`batch_obj`](@ref).
"""
function batch_grad! end

"""
    batch_hess(prob, x, batch) -> Matrix

Estimate `∇²f(x)` from one batch. Cached per `(iterate, batch)` by the oracles, since
it is a pure function of both and on the finite-difference problems it costs `O(n)`
gradient evaluations.
"""
function batch_hess end

"""
    grad_variance(prob, x, batch) -> Float64

`tr Var(∇F_i)` over the batch — the `σ_g²` the [`NormTest`](@ref) and
[`RadiusProportional`](@ref) consume.
"""
function grad_variance end

"""
    obj_variance(prob, x, batch) -> Float64

`Var(F_i)` over the batch — the `σ_f²` that [`SequentialEstimation`](@ref) and the
objective half of [`RadiusProportional`](@ref) consume.
"""
function obj_variance end

"""
    draw_batch(prob, rng, N; replace = true)

Produce a batch of size `N`.

For a [`FiniteSumProblem`](@ref) this subsets `1:M` and `replace` decides whether
with or without replacement. For an [`ExpectationProblem`](@ref) it draws `N` fresh
realisations and `replace` is meaningless (an unbounded population is always sampled
with replacement in effect).

!!! note "`N ≥ M` always returns the whole population"
    On a finite sum, `replace` is ignored once `N` reaches `M`: the result is
    `collect(1:M)`, not `rand(1:M, M)`. A bootstrap resample of size `M` omits about
    37% of the terms, so it is not the population and an iteration over it would not
    be exact — which would make [`FullBatch`](@ref) noisy and the equivalence against
    [`DeterministicTRSolver`](@ref) false.
"""
function draw_batch end

"""
    true_objective(prob, x)

The exact `f(x)`, by evaluating every term rather than a batch.

Defined for every [`FiniteSumProblem`](@ref) — one pass over all `M` terms — and for
those expectations whose construction supplies the mean. Guarded by
[`has_truth`](@ref).
"""
function true_objective end

"""
    true_gradient(prob, x)

The exact `∇f(x)`.

The counterpart of [`true_objective`](@ref), and the one that matters for scoring: a
stopping test on `‖ĝ_k‖` is a statement about one batch, and a mechanism that shrinks
the radius fast enough will meet it on noise alone. Every reported number should come
from this, and `TRParams(true_stop = true)` makes the solver's own stopping test do
so.

Costs a full pass, so it belongs in the reporting and in `trace`, not in the inner
loop. A run on a problem without truth cannot use `true_stop` and gets no
`:true_grad_trajectory`.
"""
function true_gradient end

# =============================================================================
# Finite sums
# =============================================================================

"""
    FiniteSum(n, M, Fi, Gi!, Hi)

`f(x) = (1/M) Σ_{i=1}^{M} F_i(x)`, sampled by subsetting the index set.

- `Fi(i, x) -> Real`
- `Gi!(i, x, g) -> nothing`, accumulating `∇F_i(x)` into `g`
- `Hi(i, x) -> AbstractMatrix`

The empirical-risk case. Subsampling the same index set at `x_k` and `x_k + s_k`
gives common random numbers for free, and it is worth having: the noise in
`f̂(x_k) − f̂(x_k + s_k)` then cancels to the extent that the `F_i` are smooth,
which is what keeps ρ̂ usable at small radii.
"""
struct FiniteSum{FF, GG, HH} <: FiniteSumProblem
    n::Int
    M::Int
    Fi::FF
    Gi!::GG
    Hi::HH
end

population(p::FiniteSum) = p.M

batch_obj(p::FiniteSum, x, batch) = sum(p.Fi(i, x) for i in batch) / length(batch)

function batch_grad!(p::FiniteSum, x, batch, g)
    fill!(g, 0); tmp = similar(g)
    for i in batch
        fill!(tmp, 0); p.Gi!(i, x, tmp); g .+= tmp
    end
    g ./= length(batch); return nothing
end

function batch_hess(p::FiniteSum, x, batch)
    H = zeros(eltype(x), p.n, p.n)
    for i in batch; H .+= p.Hi(i, x); end
    return H ./ length(batch)
end

function grad_variance(p::FiniteSum, x, batch)
    length(batch) < 2 && return 0.0
    ḡ = zeros(eltype(x), p.n); batch_grad!(p, x, batch, ḡ)
    tmp = similar(ḡ); acc = 0.0
    for i in batch
        fill!(tmp, 0); p.Gi!(i, x, tmp); acc += sum(abs2, tmp .- ḡ)
    end
    return acc / (length(batch) - 1)
end

function obj_variance(p::FiniteSum, x, batch)
    length(batch) < 2 && return 0.0
    m = batch_obj(p, x, batch)
    return sum((p.Fi(i, x) - m)^2 for i in batch) / (length(batch) - 1)
end

"""
    PerturbedSum(base, M; σg = 1.0, σH = 0.0, seed = 0)

A finite sum whose mean is **exactly** a known model: with mean-zero `b_i` and
symmetric mean-zero `C_i`,

```math
F_i(x) = f_{\\text{base}}(x) + b_i^\\top x + \\tfrac12 x^\\top C_i x ,
\\qquad \\tfrac1M \\textstyle\\sum_i F_i \\equiv f_{\\text{base}} .
```

The perturbations are smooth in `x`, so common random numbers behave as they do on
a real empirical risk rather than cancelling artificially, while the exact `f`,
`∇f` and `∇²f` remain available through `base`.

That combination is what makes a controlled experiment possible: the estimation
error at every iterate is known, not merely bounded.

Contrast [`PerturbedExpectation`](@ref), the same construction without the finite
population — there the perturbations are drawn fresh and their mean is zero only in
expectation, so no sample size makes the estimate exact.
"""
struct PerturbedSum{P} <: FiniteSumProblem
    base::P
    n::Int
    M::Int
    b::Matrix{Float64}
    C::Vector{Matrix{Float64}}
end

function PerturbedSum(base, M::Int; σg::Real = 1.0, σH::Real = 0.0, seed::Int = 0)
    n = base.meta.nvar
    rng = MersenneTwister(seed)
    b = σg .* randn(rng, n, M)
    b .-= sum(b; dims = 2) ./ M                       # exactly mean-zero
    if σH > 0
        raw = [σH .* randn(rng, n, n) for _ in 1:M]
        raw = [(A .+ A') ./ 2 for A in raw]
        m̄ = sum(raw) ./ M
        C = [A .- m̄ for A in raw]
    else
        C = [zeros(n, n) for _ in 1:M]
    end
    return PerturbedSum{typeof(base)}(base, n, M, b, C)
end

population(p::PerturbedSum) = p.M

function batch_obj(p::PerturbedSum, x, batch)
    acc = obj(p.base, x); N = length(batch)
    for i in batch
        acc += (dot(view(p.b, :, i), x) + 0.5 * dot(x, p.C[i] * x)) / N
    end
    return acc
end

function batch_grad!(p::PerturbedSum, x, batch, g)
    grad!(p.base, x, g); N = length(batch)
    for i in batch
        g .+= view(p.b, :, i) ./ N
        g .+= (p.C[i] * x) ./ N
    end
    return nothing
end

"""
    batch_hess(p::PerturbedSum, x, batch)

Uses `_full_hessian` rather than `H + H' - Diagonal(diag(H))`. The old
line assumed `NLPModels.hess` returns a bare lower triangle; for ADNLPModels it
returns a `Symmetric` wrapper whose `Matrix` is already full, so the symmetrisation
**doubled every off-diagonal entry** — silently corrupting the exact Hessian of the
one problem class built to have a known ground truth.
"""
function batch_hess(p::PerturbedSum, x, batch)
    H = _full_hessian(hess(p.base, x)); N = length(batch)
    for i in batch; H .+= p.C[i] ./ N; end
    return H
end

function grad_variance(p::PerturbedSum, x, batch)
    length(batch) < 2 && return 0.0
    N = length(batch)
    cols = [view(p.b, :, i) .+ p.C[i] * x for i in batch]
    m̄ = sum(cols) ./ N
    return sum(sum(abs2, c .- m̄) for c in cols) / (N - 1)
end

function obj_variance(p::PerturbedSum, x, batch)
    length(batch) < 2 && return 0.0
    N = length(batch)
    vals = [dot(view(p.b, :, i), x) + 0.5 * dot(x, p.C[i] * x) for i in batch]
    m̄ = sum(vals) / N
    return sum((v - m̄)^2 for v in vals) / (N - 1)
end

true_objective(p::PerturbedSum, x) = obj(p.base, x)
true_gradient(p::PerturbedSum, x)  = grad(p.base, x)

true_objective(p::FiniteSum, x) = sum(p.Fi(i, x) for i in 1:p.M) / p.M
function true_gradient(p::FiniteSum, x)
    g = zeros(eltype(x), p.n); tmp = similar(g)
    for i in 1:p.M
        fill!(tmp, 0); p.Gi!(i, x, tmp); g .+= tmp
    end
    return g ./ p.M
end

# Finite-sum batches are index vectors.
#
# `N >= M` returns the whole index set, **regardless of `replace`**. This is not a
# convenience: it is what makes the full-batch limit real. Drawing M indices with
# replacement is a bootstrap resample — roughly 37% of the terms are missing and
# others appear twice — so it is *not* the population, and an iteration at N_k = M
# would not be exact. The entire justification for separating FiniteSumProblem from
# ExpectationProblem is that a finite sum can be exhausted; a `rand(1:M, M)` that
# silently fails to exhaust it makes `FullBatch` noisy, `:full_batch_trajectory` a
# lie, and the equivalence against DeterministicTRSolver false.
#
# `resample!` clamps N to `min(budget, M)`, so any rule that saturates the cap lands
# here and gets the exact batch — which is the documented claim.
function draw_batch(p::FiniteSumProblem, rng, N::Int; replace::Bool = true)
    M = population(p)
    N >= M && return collect(1:M)                 # the population, not a resample of it
    return replace ? rand(rng, 1:M, N) : randperm(rng, M)[1:N]
end

"""
    full_batch(p::FiniteSumProblem) -> Vector{Int}

The whole index set `1:M`. A finite-sum iteration on this batch is *exactly*
deterministic, which is the limit an expectation does not have.
"""
full_batch(p::FiniteSumProblem) = collect(1:population(p))

# =============================================================================
# Expectations
# =============================================================================

"""
    GaussianDraw

One expectation batch: `N` realisations `(b_i, C_i)` drawn i.i.d., held as an
`n × N` matrix and an optional vector of matrices.

Unlike a finite-sum batch this is *data*, not indices — which is the whole
difference between the two classes at the evaluation level, and the reason
`batch_obj` and friends take an opaque `batch` argument rather than a
`Vector{Int}`.
"""
struct GaussianDraw
    b::Matrix{Float64}
    C::Union{Nothing, Vector{Matrix{Float64}}}
end

Base.length(d::GaussianDraw) = size(d.b, 2)

"""
    PerturbedExpectation(base; σg = 1.0, σH = 0.0)

`f(x) = E_ξ[ f_base(x) + b(ξ)ᵀx + ½ xᵀC(ξ)x ]` with `b ~ N(0, σg²I)` and `C`
symmetric mean-zero: an expectation whose mean is exactly `base`.

The expectation twin of [`PerturbedSum`](@ref), and the difference between them is
the point of having both:

- `PerturbedSum` **centres** its `M` perturbations, so at `N = M` the estimate is
  exact and the accuracy hypotheses of the stochastic theory are discharged.
- `PerturbedExpectation` draws fresh, so the sample mean of `b` is `O(σ_g/√N)` at
  every `N`. **There is no sample size at which the estimate is exact**, and no
  full-batch iteration to fall back on.

`has_truth` is `true` here — `base` is available — which makes it the right
expectation problem for a controlled experiment. A general expectation has no such
handle, and `TRParams(true_stop = true)` is rejected against it.
"""
struct PerturbedExpectation{P} <: ExpectationProblem
    base::P
    n::Int
    σg::Float64
    σH::Float64
end

function PerturbedExpectation(base; σg::Real = 1.0, σH::Real = 0.0)
    σg >= 0 && σH >= 0 || throw(ArgumentError("PerturbedExpectation: need σg, σH ≥ 0"))
    return PerturbedExpectation{typeof(base)}(base, base.meta.nvar, float(σg), float(σH))
end

has_truth(::PerturbedExpectation) = true

function draw_batch(p::PerturbedExpectation, rng, N::Int; replace::Bool = true)
    b = p.σg .* randn(rng, p.n, N)                    # NOT centred: that is the point
    C = if p.σH > 0
        [(A = p.σH .* randn(rng, p.n, p.n); (A .+ A') ./ 2) for _ in 1:N]
    else
        nothing
    end
    return GaussianDraw(b, C)
end

function batch_obj(p::PerturbedExpectation, x, d::GaussianDraw)
    N = length(d)
    acc = obj(p.base, x) + dot(vec(sum(d.b; dims = 2)) ./ N, x)
    d.C === nothing && return acc
    for Ci in d.C; acc += 0.5 * dot(x, Ci * x) / N; end
    return acc
end

function batch_grad!(p::PerturbedExpectation, x, d::GaussianDraw, g)
    N = length(d)
    grad!(p.base, x, g)
    g .+= vec(sum(d.b; dims = 2)) ./ N
    if d.C !== nothing
        for Ci in d.C; g .+= (Ci * x) ./ N; end
    end
    return nothing
end

function batch_hess(p::PerturbedExpectation, x, d::GaussianDraw)
    H = _full_hessian(hess(p.base, x))
    d.C === nothing && return H
    N = length(d)
    for Ci in d.C; H .+= Ci ./ N; end
    return H
end

function grad_variance(p::PerturbedExpectation, x, d::GaussianDraw)
    N = length(d); N < 2 && return 0.0
    cols = [view(d.b, :, i) .+ (d.C === nothing ? zero(x) : d.C[i] * x) for i in 1:N]
    m̄ = sum(cols) ./ N
    return sum(sum(abs2, c .- m̄) for c in cols) / (N - 1)
end

function obj_variance(p::PerturbedExpectation, x, d::GaussianDraw)
    N = length(d); N < 2 && return 0.0
    vals = [dot(view(d.b, :, i), x) +
            (d.C === nothing ? 0.0 : 0.5 * dot(x, d.C[i] * x)) for i in 1:N]
    m̄ = sum(vals) / N
    return sum((v - m̄)^2 for v in vals) / (N - 1)
end

true_objective(p::PerturbedExpectation, x) = obj(p.base, x)
true_gradient(p::PerturbedExpectation, x)  = grad(p.base, x)

# =============================================================================
# Scored problems
# =============================================================================

"""
    scores(prob, x, batch) -> Matrix

The `n × |batch|` matrix whose column `i` is `∇f_i(x)`.

The column mean is the gradient, so this is strictly more information than a
gradient evaluation and usually costs no more to produce. It is what the
outer-product models consume, and the reason a likelihood admits a Hessian
approximation a general objective does not.

Implementations must return a genuine `Matrix`, not a lazy `Adjoint`:
`OuterProductOperator` stores what it is given.
"""
function scores end

"""
    loss_terms(prob, x, batch) -> Vector

The per-observation values `f_i(x)`, whose mean is `f(x)`. Kept separate from the
sum because their sample variance is what the adaptive rules need.
"""
function loss_terms end

batch_obj(p::ScoredProblem, x, batch) = sum(loss_terms(p, x, batch)) / length(batch)

function batch_grad!(p::ScoredProblem, x, batch, g)
    S = scores(p, x, batch)
    g .= vec(sum(S; dims = 2)) ./ length(batch)
    return nothing
end

function grad_variance(p::ScoredProblem, x, batch)
    length(batch) < 2 && return 0.0
    S = scores(p, x, batch); N = size(S, 2)
    ḡ = vec(sum(S; dims = 2)) ./ N
    return sum(abs2, S .- ḡ) / (N - 1)
end

function obj_variance(p::ScoredProblem, x, batch)
    length(batch) < 2 && return 0.0
    v = loss_terms(p, x, batch); m = sum(v) / length(v)
    return sum((vi - m)^2 for vi in v) / (length(v) - 1)
end

# Every ScoredProblem can be scored on the truth: the full batch is the truth.
true_objective(p::ScoredProblem, x) = batch_obj(p, x, full_batch(p))
function true_gradient(p::ScoredProblem, x)
    g = zeros(eltype(x), p.n); batch_grad!(p, x, full_batch(p), g); return g
end
