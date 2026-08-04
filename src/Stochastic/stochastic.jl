# =============================================================================
# src/Stochastic/stochastic.jl
#
# The sampling layer: a fourth axis for problems given as an expectation,
#
#     f(x) = E_ξ[ F(x, ξ) ],
#
# estimated from a sample of size N_k that may change every iteration.
#
# Three pieces.
#
#   StochasticProblem   where the randomness lives: a finite sum to subsample,
#                       or a smooth perturbation of a known model so that the
#                       truth is available for scoring.
#   SamplingRule        how N_k is chosen. This is the axis under study.
#   SampledNLP          an AbstractNLPModel that answers obj/grad/hess from the
#                       current batch, so the existing solver, every radius
#                       mechanism, every model Hessian and every subsolver run
#                       unchanged over it.
#
# ---------------------------------------------------------------------------
# Why the radius mechanism decides the sampling cost
#
# The standard accuracy requirements for convergence of a stochastic trust-region
# method (Chen, Menickelly & Scheinberg 2018) are stated in terms of the radius:
# the model must be κ_eg-fully-linear on B(x_k, Δ_k), and the function estimates
# accurate to κ_ef Δ_k², both with probability at least 1 − α. For a Monte Carlo
# estimator with per-sample variance σ², meeting them takes
#
#     N_k^grad ≳ (σ_g / (κ_eg Δ_k))²  = Θ(Δ_k^{-2}),
#     N_k^obj  ≳ (σ_f / (κ_ef Δ_k²))² = Θ(Δ_k^{-4}).
#
# So the total work is Σ_k Δ_k^{-2}, the reciprocal of the Σ_k Δ_k² tables of
# Part II, and the deterministic ranking of the mechanisms inverts. A rule with
# Δ_k → 0 pays unboundedly per iteration near the solution; a rule with
# liminf Δ_k > 0 does not. On the running example of Part II, over 60 iterations:
#
#     rule       Σ Δ_k²      Σ Δ_k^{-2}     Σ Δ_k^{-4}
#     R-delta    4.4e+35      1.3e+00        1.1e+00
#     R-step     2.4e+02      1.6e+01        4.7e+00
#     R-DFO      1.4e+00      2.0e+05        2.0e+09
#     R-grad     1.4e+00      7.5e+04        1.7e+08
#
# The mechanisms that look best by the summability criteria of the deterministic
# theory are the most expensive to run stochastically, by five to nine orders of
# magnitude. That is why iteration counts are the wrong cost measure here and
# `samples_used` is recorded alongside them.
# =============================================================================

# -----------------------------------------------------------------------------
# Problems
# -----------------------------------------------------------------------------

"""
    StochasticProblem

A problem given as an expectation, sampled by drawing a batch of realisations.

A concrete subtype implements

    n_terms(prob)                                 -> Int  (or `typemax(Int)`)
    batch_obj(prob, x, batch)                     -> Float64
    batch_grad!(prob, x, batch, g)                -> nothing
    batch_hess(prob, x, batch)                    -> Matrix
    grad_variance(prob, x, batch)                 -> Float64
    obj_variance(prob, x, batch)                  -> Float64

`batch` is a vector of term indices. The same batch is reused for `f(x_k)` and
`f(x_k + s_k)` — see [`SampledNLP`](@ref) on common random numbers — so the
estimator must be a deterministic function of `(x, batch)`.
"""
abstract type StochasticProblem end

"""
    FiniteSum(n, M, Fi, Gi!, Hi)

`f(x) = (1/M) Σ_{i=1}^{M} F_i(x)`, sampled by subsetting the index set.

- `Fi(i, x) -> Real`
- `Gi!(i, x, g) -> nothing`, accumulating `∇F_i(x)` into `g`
- `Hi(i, x) -> AbstractMatrix`

This is the empirical-risk case: logistic regression, maximum likelihood,
sample-average approximation of a simulated objective. Subsampling the same index
set at `x_k` and `x_k + s_k` gives common random numbers for free, and it is
worth having: the noise in `f(x_k) − f(x_k + s_k)` then cancels to the extent
that the `F_i` are smooth, which is what keeps ρ usable at small radii.
"""
struct FiniteSum{FF, GG, HH} <: StochasticProblem
    n::Int
    M::Int
    Fi::FF
    Gi!::GG
    Hi::HH
end

n_terms(p::FiniteSum) = p.M

batch_obj(p::FiniteSum, x, batch) =
    sum(p.Fi(i, x) for i in batch) / length(batch)

function batch_grad!(p::FiniteSum, x, batch, g)
    fill!(g, 0)
    tmp = similar(g)
    for i in batch
        fill!(tmp, 0); p.Gi!(i, x, tmp); g .+= tmp
    end
    g ./= length(batch)
    return nothing
end

function batch_hess(p::FiniteSum, x, batch)
    H = zeros(eltype(x), p.n, p.n)
    for i in batch
        H .+= p.Hi(i, x)
    end
    return H ./ length(batch)
end

"Sample variance of ‖∇F_i(x)‖ over the batch, the σ_g² an adaptive rule needs."
function grad_variance(p::FiniteSum, x, batch)
    length(batch) < 2 && return 0.0
    ḡ = zeros(eltype(x), p.n); batch_grad!(p, x, batch, ḡ)
    tmp = similar(ḡ); acc = 0.0
    for i in batch
        fill!(tmp, 0); p.Gi!(i, x, tmp)
        acc += sum(abs2, tmp .- ḡ)
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

A finite sum whose mean is exactly a known model: with mean-zero `b_i` and
symmetric mean-zero `C_i`,

```math
F_i(x) = f_{\\text{base}}(x) + b_i^\\top x + \\tfrac12 x^\\top C_i x ,
\\qquad \\tfrac1M \\textstyle\\sum_i F_i \\equiv f_{\\text{base}} .
```

The perturbations are smooth in `x`, so common random numbers behave as they do
on a real empirical risk rather than cancelling artificially, while the exact
`f`, `∇f` and `∇²f` remain available through `base` for scoring.

That combination is what makes a controlled experiment possible: the estimation
error at every iterate is known, not merely bounded, so a claim about how a
radius mechanism responds to noise can be checked directly instead of inferred
from the final iterate.

`σg` scales the gradient noise and `σH` the curvature noise; `σH = 0` leaves the
Hessian exact, isolating the effect of gradient and objective noise.
"""
struct PerturbedSum{P} <: StochasticProblem
    base::P
    n::Int
    M::Int
    b::Matrix{Float64}      # n × M, columns summing to zero
    C::Vector{Matrix{Float64}}
end

function PerturbedSum(base, M::Int; σg::Real = 1.0, σH::Real = 0.0, seed::Int = 0)
    n = base.meta.nvar
    rng = MersenneTwister(seed)
    b = σg .* randn(rng, n, M)
    b .-= sum(b; dims = 2) ./ M                       # exactly mean-zero
    C = Matrix{Float64}[]
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

n_terms(p::PerturbedSum) = p.M

function batch_obj(p::PerturbedSum, x, batch)
    acc = obj(p.base, x)
    N = length(batch)
    for i in batch
        acc += (dot(view(p.b, :, i), x) + 0.5 * dot(x, p.C[i] * x)) / N
    end
    return acc
end

function batch_grad!(p::PerturbedSum, x, batch, g)
    grad!(p.base, x, g)
    N = length(batch)
    for i in batch
        g .+= view(p.b, :, i) ./ N
        g .+= (p.C[i] * x) ./ N
    end
    return nothing
end

function batch_hess(p::PerturbedSum, x, batch)
    H = Matrix(hess(p.base, x))
    H = H + H' - Diagonal(diag(H))                    # NLPModels returns a triangle
    N = length(batch)
    for i in batch
        H .+= p.C[i] ./ N
    end
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

"""
    ScoredProblem <: StochasticProblem

A finite-sum problem that can report the gradient of **each term separately**:

    scores(p, x, batch) -> Matrix, n × length(batch), column i = ∇fᵢ(x)
    loss_terms(p, x, batch) -> Vector, entry i = fᵢ(x)

Everything the sampling layer needs follows from those two, so a `ScoredProblem`
is usable both as an ordinary objective and as a [`SampledNLP`](@ref), and the
outer-product models below read the score matrix directly.

Per-observation scores are the extra structure that separates a likelihood from a
general objective. It is exactly what BHHH exploits, and it is free: the average
of the columns is the gradient, so any code that computes the gradient by summing
per-term contributions already has them.
"""
abstract type ScoredProblem <: StochasticProblem end

"""
    scores(prob, x, batch) -> Matrix

The `n × length(batch)` matrix whose column `i` is `∇fᵢ(x)`, the gradient of a
single term.

The average of the columns is the gradient, so this is strictly more information
than a gradient evaluation and usually costs no more to produce. It is what the
outer-product models consume, and the reason a likelihood admits a Hessian
approximation that a general objective does not.
"""
function scores end

"""
    loss_terms(prob, x, batch) -> Vector

The per-observation objective values `fᵢ(x)`, whose mean is `f(x)`.

Kept separate from the sum because their sample variance is what the adaptive
sampling rules need — see [`obj_variance`](@ref).
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

# -----------------------------------------------------------------------------
# Sampling rules — the axis under study
# -----------------------------------------------------------------------------

"""
    SamplingState

What a sampling rule may condition on: the iteration index `k`, the current
radius `Δ`, the current gradient-estimate norm `g_norm`, and the running sample
variances `σg²`, `σf²`.
"""
struct SamplingState
    k::Int
    Δ::Float64
    g_norm::Float64
    σg²::Float64
    σf²::Float64
    ip²::Float64
    orth²::Float64
    pred::Float64
    N_prev::Int
end

SamplingState(k, Δ, g_norm, σg², σf²) =
    SamplingState(k, Float64(Δ), Float64(g_norm), Float64(σg²), Float64(σf²),
                  0.0, 0.0, NaN, 0)

"""
    SampleStats(σg², σf², ip², orth²)

The four second-moment quantities the adaptive rules consume, all estimated from
one batch at one iterate:

- `σg²  = tr Var(∇Fᵢ)`, the total gradient variance — the [`NormTest`](@ref);
- `σf²  = Var(Fᵢ)`, the objective variance — [`SequentialEstimation`](@ref) and
  the objective half of [`RadiusProportional`](@ref);
- `ip²  = Var(∇Fᵢᵀĝ)`, the variance *along* the estimated gradient — the
  [`InnerProductTest`](@ref);
- `orth² = E‖∇Fᵢ − (∇Fᵢᵀĝ/‖ĝ‖²)ĝ‖²`, the variance *orthogonal* to it — the
  [`OrthogonalityTest`](@ref).

Splitting the gradient variance into its parallel and orthogonal parts is the
whole content of the inner-product tests: the norm test controls `σg² = ip²/‖ĝ‖²
+ orth²` as a single number, which is stricter than necessary, because only the
component along `ĝ` decides whether the estimated direction is still a descent
direction.
"""
struct SampleStats
    σg²::Float64
    σf²::Float64
    ip²::Float64
    orth²::Float64
end

"""
    batch_stats(prob, x, batch, g) -> SampleStats

All four moments in one pass. Generic for any problem exposing per-observation
scores; falls back to the gradient and objective variances alone otherwise, which
leaves the inner-product family unusable and says so when it is selected.
"""
function batch_stats(p::StochasticProblem, x, batch, g)
    return SampleStats(grad_variance(p, x, batch), obj_variance(p, x, batch), 0.0, 0.0)
end

"""
    SamplingRule

How the sample sizes `N_k^grad` and `N_k^obj` are chosen. A concrete subtype
implements

    grad_sample_size(rule, st::SamplingState) -> Int
    obj_sample_size(rule, st::SamplingState)  -> Int

and may implement `couples_to_radius(rule) -> Bool`.
"""
abstract type SamplingRule end

"""
    grad_sample_size(rule, st::SamplingState) -> Int

The number of terms to draw for the gradient estimate at the state `st`.

This is the quantity the survey's stochastic experiments vary. Note what it may
depend on: `st` carries `Δ_k`, so a rule *may* couple the sample size to the
radius — and [`RadiusProportional`](@ref) does, which is what makes the sampling
cost a property of the radius mechanism rather than of the problem alone.
"""
function grad_sample_size end

"""
    obj_sample_size(rule, st::SamplingState) -> Int

The number of terms to draw for the objective estimates.

Usually larger than [`grad_sample_size`](@ref): the accuracy requirement on `f` is
`O(Δ_k²)` against `O(Δ_k)` on the gradient, so the sample size scales as
`Δ_k^{-4}` rather than `Δ_k^{-2}`. On a run whose radius becomes small it is the
objective estimates, not the gradients, that dominate the bill.
"""
function obj_sample_size end

"""
    couples_to_radius(rule) -> Bool

Whether `N_k` depends on `Δ_k`. When it does, the radius mechanism and the
sampling rule are no longer independent axes: the mechanism sets `Δ_k`, which
sets `N_k`, which sets the accuracy of `ĝ_k`, which for a criticality-anchored
rule sets `Δ_{k+1}`. That loop is closed for `RGrad` and `RDFO` and open for
`RDelta`, and it is the reason a stochastic comparison cannot be read off the
deterministic one.
"""
couples_to_radius(::SamplingRule) = false

"""
    FixedSample(N; N_obj = N)

Sample-average approximation: the same `N` at every iteration.

The baseline every adaptive rule should be scored against, and the only rule here
for which total work is proportional to the iteration count — which is why
iteration-count comparisons quietly assume it.
"""
struct FixedSample <: SamplingRule
    N::Int
    N_obj::Int
    FixedSample(N::Int; N_obj::Int = N) = new(N, N_obj)
end
grad_sample_size(r::FixedSample, ::SamplingState) = r.N
obj_sample_size(r::FixedSample, ::SamplingState)  = r.N_obj

"""
    RadiusProportional(; κ_g = 1.0, κ_f = 1.0, N_min = 2, N_max = 10^6, σ_floor = 1e-12)

The STORM requirement, made operational:

```math
N_k^{\\mathrm{grad}} = \\Bigl\\lceil (\\sigma_g / (\\kappa_g \\Delta_k))^2 \\Bigr\\rceil,
\\qquad
N_k^{\\mathrm{obj}}  = \\Bigl\\lceil (\\sigma_f / (\\kappa_f \\Delta_k^2))^2 \\Bigr\\rceil,
```

so that the gradient estimate is accurate to `O(Δ_k)` and the objective estimates
to `O(Δ_k²)` with fixed probability.

This is the rule that makes the mechanism pay for its own radius. `N_k` grows like
`Δ_k^{-2}`, so a criticality-anchored mechanism, whose radius vanishes with the
gradient, drives the sample size up without bound near a solution while a
criticality-blind one does not. `N_max` caps that growth and is reported when it
binds, since a run spent at the cap is no longer meeting the accuracy requirement
the convergence theory assumes.
"""
struct RadiusProportional <: SamplingRule
    κ_g::Float64
    κ_f::Float64
    N_min::Int
    N_max::Int
    σ_floor::Float64
    function RadiusProportional(; κ_g::Real = 1.0, κ_f::Real = 1.0,
                                  N_min::Int = 2, N_max::Int = 10^6,
                                  σ_floor::Real = 1e-12)
        κ_g > 0 && κ_f > 0 || throw(ArgumentError("RadiusProportional: need κ_g, κ_f > 0"))
        0 < N_min <= N_max || throw(ArgumentError("RadiusProportional: need 0 < N_min ≤ N_max"))
        new(float(κ_g), float(κ_f), N_min, N_max, float(σ_floor))
    end
end
couples_to_radius(::RadiusProportional) = true

function grad_sample_size(r::RadiusProportional, st::SamplingState)
    σ = sqrt(max(st.σg², r.σ_floor))
    Δ = max(st.Δ, 1e-300)
    return clamp(ceil(Int, (σ / (r.κ_g * Δ))^2), r.N_min, r.N_max)
end
function obj_sample_size(r::RadiusProportional, st::SamplingState)
    σ = sqrt(max(st.σf², r.σ_floor))
    Δ = max(st.Δ, 1e-300)
    return clamp(ceil(Int, (σ / (r.κ_f * Δ^2))^2), r.N_min, r.N_max)
end

"""
    NormTest(; θ = 0.5, N_min = 2, N_max = 10^6, N_obj = nothing)

The norm test of Byrd, Chin, Nocedal & Wu (2012): choose `N_k` so that the
estimated variance of the sampled gradient is a fixed fraction of its own size,

```math
\\frac{\\sigma_g^2}{N_k} \\;\\le\\; \\theta^2 \\|\\hat g_k\\|^2 .
```

Unlike [`RadiusProportional`](@ref) this never consults `Δ_k`, so the sampling
rule and the radius mechanism stay independent — which makes it the right control
for isolating what the coupling in `RadiusProportional` actually does. It still
drives `N_k → ∞`, but through `‖ĝ_k‖ → 0` rather than through the radius, so the
growth is a property of the problem rather than of the mechanism.
"""
struct NormTest <: SamplingRule
    θ::Float64
    N_min::Int
    N_max::Int
    N_obj::Union{Int, Nothing}
    function NormTest(; θ::Real = 0.5, N_min::Int = 2, N_max::Int = 10^6,
                        N_obj::Union{Int, Nothing} = nothing)
        θ > 0 || throw(ArgumentError("NormTest: need θ > 0"))
        0 < N_min <= N_max || throw(ArgumentError("NormTest: need 0 < N_min ≤ N_max"))
        new(float(θ), N_min, N_max, N_obj)
    end
end

function grad_sample_size(r::NormTest, st::SamplingState)
    gn = max(st.g_norm, 1e-300)
    return clamp(ceil(Int, st.σg² / (r.θ^2 * gn^2)), r.N_min, r.N_max)
end
obj_sample_size(r::NormTest, st::SamplingState) =
    r.N_obj === nothing ? grad_sample_size(r, st) : r.N_obj

"""
    GeometricSample(; N₀ = 8, rate = 1.1, N_max = 10^6)

`N_k = ⌈N₀ · rate^k⌉`: a schedule fixed in advance, independent of everything the
run observes.

Included as the other control. It reaches any accuracy eventually and cannot
respond to the radius at all, so comparing it against `RadiusProportional` at
matched total sample budget separates "spending more" from "spending it at the
right iterations".
"""
struct GeometricSample <: SamplingRule
    N₀::Int
    rate::Float64
    N_max::Int
    function GeometricSample(; N₀::Int = 8, rate::Real = 1.1, N_max::Int = 10^6)
        N₀ > 0    || throw(ArgumentError("GeometricSample: need N₀ > 0"))
        rate >= 1 || throw(ArgumentError("GeometricSample: need rate ≥ 1"))
        new(N₀, float(rate), N_max)
    end
end
grad_sample_size(r::GeometricSample, st::SamplingState) =
    clamp(ceil(Int, r.N₀ * r.rate^st.k), 1, r.N_max)
obj_sample_size(r::GeometricSample, st::SamplingState) = grad_sample_size(r, st)

# -----------------------------------------------------------------------------
# The sampled model
# -----------------------------------------------------------------------------

"""
    SampledNLP(prob, rule; seed = 0, replace = true)

An `AbstractNLPModel` answering `obj`, `grad!` and `hess` from the batch drawn at
the current iteration.

Because it satisfies the ordinary NLP interface, every radius mechanism, model
Hessian and subproblem solver in the package runs over it unchanged — the
stochastic setting is a fourth axis rather than a second code path, and a
comparison of mechanisms under noise is the same comparison with one component
swapped.

# Common random numbers

The batch is redrawn once per iteration, in [`resample!`](@ref), and then held
fixed. Every evaluation within the iteration — `f̂(x_k)`, `f̂(x_k + s_k)`, and the
retrospective `f̂` if the rule needs it — therefore uses the *same* realisations.
That is not an optimisation: with independent draws the variance of
`f̂(x_k) − f̂(x_k + s_k)` does not shrink as `‖s_k‖ → 0`, so ρ̂ becomes pure noise
at small radii and every mechanism stalls for a reason that has nothing to do
with the mechanism. With common random numbers the difference is an average of
`F_i(x_k) − F_i(x_k + s_k)`, which shrinks with the step.

# Counters

`samples_used(m)` returns the cumulative number of term evaluations, split by
gradient and objective. It, not the iteration count, is the cost measure for a
stochastic comparison.
"""
mutable struct SampledNLP{T, V, P <: StochasticProblem, S <: SamplingRule} <: AbstractNLPModel{T, V}
    meta::NLPModelMeta{T, V}
    counters::Counters
    prob::P
    rule::S
    rng::MersenneTwister
    seed::Int
    replace::Bool
    batch_g::Vector{Int}
    batch_f::Vector{Int}
    Ng::Int
    Nf::Int
    samples_g::Int
    samples_f::Int
    Ng_hist::Vector{Int}
    Nf_hist::Vector{Int}
    σg²::Float64
    σf²::Float64
    ip²::Float64
    orth²::Float64
    last_pred::Float64
    capped::Int
end

function SampledNLP(prob::StochasticProblem, rule::SamplingRule;
                    x0::AbstractVector = zeros(prob.n), seed::Int = 0,
                    replace::Bool = true, N_init::Int = 8)
    n = prob.n
    meta = NLPModelMeta(n; x0 = Vector{Float64}(x0), name = "SampledNLP")
    M = n_terms(prob)
    N0 = min(N_init, M)
    rng = MersenneTwister(seed)
    b = _draw(rng, M, N0, replace)
    return SampledNLP{Float64, Vector{Float64}, typeof(prob), typeof(rule)}(
        meta, Counters(), prob, rule, rng, seed, replace,
        copy(b), copy(b), N0, N0, 0, 0, Int[], Int[], 1.0, 1.0, 0.0, 0.0, NaN, 0)
end

_draw(rng, M, N, replace) =
    replace ? rand(rng, 1:M, N) : (N >= M ? collect(1:M) : randperm(rng, M)[1:N])

NLPModels.obj(m::SampledNLP, x::AbstractVector) =
    (m.counters.neval_obj += 1; batch_obj(m.prob, x, m.batch_f))

function NLPModels.grad!(m::SampledNLP, x::AbstractVector, g::AbstractVector)
    m.counters.neval_grad += 1
    batch_grad!(m.prob, x, m.batch_g, g)
    return g
end

NLPModels.hess(m::SampledNLP, x::AbstractVector) =
    (m.counters.neval_hess += 1; Symmetric(batch_hess(m.prob, x, m.batch_g)))

function NLPModels.hprod!(m::SampledNLP, x::AbstractVector, v::AbstractVector,
                          Hv::AbstractVector; obj_weight = 1.0)
    m.counters.neval_hprod += 1
    H = batch_hess(m.prob, x, m.batch_g)
    mul!(Hv, H, v)
    obj_weight == 1 || (Hv .*= obj_weight)
    return Hv
end

"""
    resample!(m::SampledNLP, k, Δ, g_norm) -> (Ng, Nf)

Choose `N_k` from the sampling rule and draw the batches for iteration `k`.

Called once per iteration by the solver, before anything is evaluated. The
variance estimates fed to the rule are those of the *previous* batch, since the
new one does not exist yet — the usual plug-in, and the reason `N_min` matters:
a rule cannot estimate a variance from one sample.
"""
function resample!(m::SampledNLP, k::Int, Δ::Real, g_norm::Real)
    st = SamplingState(k, Float64(Δ), Float64(g_norm), m.σg², m.σf²,
                       m.ip², m.orth², m.last_pred, m.Ng)
    M  = n_terms(m.prob)
    Ng = min(grad_sample_size(m.rule, st), M)
    Nf = min(obj_sample_size(m.rule, st), M)

    Nmax = _rule_cap(m.rule)
    (Ng >= Nmax || Nf >= Nmax) && (m.capped += 1)

    m.Ng, m.Nf = Ng, Nf
    m.batch_g = _draw(m.rng, M, Ng, m.replace)
    # Objective and gradient share the batch when the sizes agree, which is the
    # common case and gives the tightest common random numbers.
    m.batch_f = Ng == Nf ? m.batch_g : _draw(m.rng, M, Nf, m.replace)

    m.samples_g += Ng
    m.samples_f += Nf
    push!(m.Ng_hist, Ng); push!(m.Nf_hist, Nf)
    return Ng, Nf
end

_rule_cap(r::SamplingRule) = hasproperty(r, :N_max) ? r.N_max : typemax(Int)

"Refresh the variance estimates from the current batch, at the current iterate."
function update_variances!(m::SampledNLP, x::AbstractVector)
    g = zeros(eltype(x), m.prob.n)
    batch_grad!(m.prob, x, m.batch_g, g)
    stt = batch_stats(m.prob, x, m.batch_g, g)
    m.σg², m.ip², m.orth² = stt.σg², stt.ip², stt.orth²
    m.σf² = obj_variance(m.prob, x, m.batch_f)
    return m.σg², m.σf²
end

"""
    record_prediction!(m::SampledNLP, pred) -> pred

Store the model's predicted reduction for the iteration just computed.

[`SequentialEstimation`](@ref) sizes the next batch against it: the sample must be
large enough that the noise in the estimated decrease is small beside the decrease
the model claims. No other rule reads it, and it is a no-op for a deterministic
model.
"""
record_prediction!(m::SampledNLP, pred::Real) = (m.last_pred = Float64(pred); pred)
record_prediction!(::AbstractNLPModel, pred::Real) = pred

"""
    samples_used(m::SampledNLP) -> (grad = …, obj = …, total = …)

Cumulative term evaluations. The cost measure for a stochastic comparison:
iteration counts are only proportional to work under [`FixedSample`](@ref).
"""
samples_used(m::SampledNLP) =
    (grad = m.samples_g, obj = m.samples_f, total = m.samples_g + m.samples_f)

"""
    reset_sampling!(m::SampledNLP) -> m

Restore the counters, the sample-size history and the random stream to their
initial state.

Called at the start of every solve, so the same `SampledNLP` can be reused across
configurations and each run sees the same realisations — which is what makes a
comparison of mechanisms under noise a comparison of mechanisms rather than of
random seeds.
"""
function reset_sampling!(m::SampledNLP)
    m.rng = MersenneTwister(m.seed)
    m.samples_g = 0; m.samples_f = 0
    empty!(m.Ng_hist); empty!(m.Nf_hist)
    m.σg² = 1.0; m.σf² = 1.0; m.ip² = 0.0; m.orth² = 0.0
    m.last_pred = NaN; m.capped = 0
    reset_sampling_rule!(m.rule)          # SequentialEstimation carries N_last
    return m
end

"""
    prepare_iteration!(nlp, k, Δ, g_norm) -> Bool

Hook called by the solver at the top of every iteration. Returns `true` if the
model changed underneath and the incumbent `f` and `g` must be re-evaluated.

The default is a no-op returning `false`, so a deterministic run is untouched.
"""
prepare_iteration!(::AbstractNLPModel, ::Int, ::Real, ::Real) = false

function prepare_iteration!(m::SampledNLP, k::Int, Δ::Real, g_norm::Real)
    resample!(m, k, Δ, g_norm)
    return true
end

# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------

"""
    true_objective(prob, x)  /  true_objective(m::SampledNLP, x)

The exact `f(x)` when the problem knows it — always for
[`PerturbedSum`](@ref), whose perturbations cancel by construction.

A stochastic run must be scored on the truth, not on its own estimates. `‖ĝ_k‖`
below a tolerance is a statement about one batch, and a mechanism that shrinks the
radius fast enough will meet it on noise alone. Every reported result in an
experiment should use these.
"""
true_objective(p::PerturbedSum, x) = obj(p.base, x)
true_gradient(p::PerturbedSum, x)  = grad(p.base, x)
true_objective(p::FiniteSum, x)    = sum(p.Fi(i, x) for i in 1:p.M) / p.M

"""
    true_gradient(prob, x)  /  true_gradient(m::SampledNLP, x)

The exact `∇f(x)`, by summing every term rather than a batch.

The counterpart of [`true_objective`](@ref), and the one that matters for scoring:
a stopping test on `‖ĝ_k‖` is a statement about one batch, and a mechanism that
shrinks the radius fast enough will satisfy it on noise alone. Costs a full pass
over the problem, so it belongs in the reporting, not in the loop.
"""
function true_gradient(p::FiniteSum, x)
    g = zeros(eltype(x), p.n); tmp = similar(g)
    for i in 1:p.M
        fill!(tmp, 0); p.Gi!(i, x, tmp); g .+= tmp
    end
    return g ./ p.M
end

true_objective(m::SampledNLP, x) = true_objective(m.prob, x)
true_gradient(m::SampledNLP, x)  = true_gradient(m.prob, x)
