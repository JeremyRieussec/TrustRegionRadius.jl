# =============================================================================
# src/Sampling/rules.jl
#
# The sampling rules — the fourth axis.
#
#   FullBatch               N_k = M                        finite sum only
#   FixedSample             N_k = N
#   RadiusProportional      (σ/(κΔ_k))²                    couples to Δ
#   NormTest                σ_g²/(θ²‖ĝ‖²)
#   GeometricSample         N₀·rate^k
#   InnerProductTest        Var(∇Fᵢᵀĝ)/(θ²‖ĝ‖⁴)            needs scores
#   OrthogonalityTest       E‖proj⊥∇Fᵢ‖²/(ν²‖ĝ‖²)          needs scores
#   AugmentedInnerProduct   the maximum of the two          needs scores
#   SequentialEstimation    2z²σ_f²/(κ²pred²)              couples through pred
#
# ---------------------------------------------------------------------------
# Where the cap comes from
#
# A rule computes a *requirement*; it does not decide the *budget*. The two were
# previously conflated in an `N_max` field, which meant the same keyword expressed
# two different things depending on the problem class:
#
#   expectation  the population is unbounded, so N_max IS the user's budget and
#                belongs wherever they want to put it;
#   finite sum   the population is M, so the cap is a property of the PROBLEM.
#                A user N_max there is either redundant (≥ M) or a deliberate
#                sub-population budget (< M) — different intentions.
#
# So `N_max` on a rule now defaults to `nothing`, `user_cap` reports whether it was
# set, and `check_population_cap` rejects it on a finite sum with a message
# pointing at the solver's `budget` keyword. The effective cap the solver applies
# is `min(sample_cap(rule), solver budget, population(prob))`.
#
# `N_min` stays on the rule: a floor is a property of the estimator, not of the
# budget — you cannot estimate a variance from one sample.
# =============================================================================

"""
    _sample_size(v, N_min, N_max) -> Int

Convert a real-valued requirement to an `Int`, **saturating** at `N_max`.

`clamp(ceil(Int, v), lo, hi)` cannot be used: the conversion happens first, so
`v > typemax(Int)` throws `InexactError` before the clamp is reached — and that is
routine here. The header table of the survey quotes `Σ Δ^{-4} ≈ 2e9` for R-DFO; at
`Δ = 1e-8` with `σ_f = 1` the objective requirement is `1e32` against a maximum
representable `9.2e18`.
"""
@inline function _sample_size(v::Real, N_min::Int, N_max::Int)
    isnan(v)     && return N_max
    v <= N_min   && return N_min
    v >= N_max   && return N_max
    return clamp(ceil(Int, v), N_min, N_max)
end

"""
    SamplingState

What a sampling rule may condition on: the iteration index `k`, the radius `Δ`,
the gradient-estimate norm `g_norm`, the running variances `σg²`, `σf²`, their
parallel/orthogonal split `ip²`, `orth²`, the previous predicted reduction `pred`,
the previous sample size `N_prev`, and the population `N_pop`.

`N_pop` is `M` for a finite sum and `typemax(Int)` for an expectation. Only
[`FullBatch`](@ref) reads it, and it raises on an expectation — where "all of it"
does not exist.
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
    N_pop::Int
end

SamplingState(k, Δ, g_norm, σg², σf²) =
    SamplingState(k, Float64(Δ), Float64(g_norm), Float64(σg²), Float64(σf²),
                  0.0, 0.0, NaN, 0, typemax(Int))

SamplingState(k, Δ, g_norm, σg², σf², ip², orth², pred, N_prev) =
    SamplingState(k, Float64(Δ), Float64(g_norm), Float64(σg²), Float64(σf²),
                  Float64(ip²), Float64(orth²), Float64(pred), N_prev, typemax(Int))

"""
    SampleStats(σg², σf², ip², orth²)

The four second moments the adaptive rules consume:

- `σg² = tr Var(∇Fᵢ)` — the [`NormTest`](@ref);
- `σf² = Var(Fᵢ)` — [`SequentialEstimation`](@ref) and the objective half of
  [`RadiusProportional`](@ref);
- `ip² = Var(∇Fᵢᵀĝ)` — the variance *along* the estimated gradient;
- `orth² = E‖∇Fᵢ − (∇Fᵢᵀĝ/‖ĝ‖²)ĝ‖²` — the variance *orthogonal* to it.

Splitting `σg² = ip²/‖ĝ‖² + orth²` is the whole content of the inner-product tests:
only the component along `ĝ` decides whether the estimated direction is still a
descent direction, so bounding the sum is stricter than descent requires.
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
scores; falls back to the gradient and objective variances alone otherwise, in
which case the inner-product family is rejected at oracle construction rather than
at the first iteration.
"""
batch_stats(p::AbstractProblem, x, batch, g) =
    SampleStats(grad_variance(p, x, batch), obj_variance(p, x, batch), 0.0, 0.0)

"""
    batch_stats(p::ScoredProblem, x, batch, g) -> SampleStats

The parallel/orthogonal split, exact rather than estimated separately: with `S` the
`n × N` score matrix and `ĝ` its column mean,

    ip²   = Var_i(Sᵢᵀ ĝ),
    orth² = (1/(N−1)) Σᵢ ‖Sᵢ − ĝ‖² − ip²/‖ĝ‖² ,

so `σ_g² = ip²/‖ĝ‖² + orth²` holds to rounding and the tests are comparable on the
same batch. `σf²` comes back on the same pass, so the caller need not traverse
twice.
"""
function batch_stats(p::ScoredProblem, x, batch, g)
    S = scores(p, x, batch); N = size(S, 2)
    N < 2 && return SampleStats(0.0, 0.0, 0.0, 0.0)
    ḡ  = vec(sum(S; dims = 2)) ./ N
    D  = S .- ḡ
    σg² = sum(abs2, D) / (N - 1)
    σf² = obj_variance(p, x, batch)
    gn² = dot(ḡ, ḡ)
    gn² <= 0 && return SampleStats(σg², σf², 0.0, σg²)
    proj  = vec(ḡ' * D)
    ip²   = sum(abs2, proj) / (N - 1)
    orth² = max(σg² - ip² / gn², 0.0)
    return SampleStats(σg², σf², ip², orth²)
end

# -----------------------------------------------------------------------------
# The rule interface
# -----------------------------------------------------------------------------

"""
    SamplingRule

How `N_k^grad` and `N_k^obj` are chosen. A concrete subtype implements

    grad_sample_size(rule, st::SamplingState) -> Int
    obj_sample_size(rule, st::SamplingState)  -> Int

and may implement [`couples_to_radius`](@ref), [`needs_scores`](@ref),
[`user_cap`](@ref), [`requires_finite_population`](@ref) and
[`reset_sampling_rule!`](@ref).

Implementations must be **pure**: the solver calls both queries in the same
iteration, so a rule that mutates itself in one is invoked twice per step.
[`SequentialEstimation`](@ref) caches on `st.k` for exactly this reason.
"""
abstract type SamplingRule end

"""
    grad_sample_size(rule, st) -> Int

Terms to draw for the gradient estimate. The quantity the stochastic experiments
vary. `st` carries `Δ_k`, so a rule *may* couple the sample size to the radius —
[`RadiusProportional`](@ref) does, which is what makes the sampling cost a property
of the radius mechanism rather than of the problem alone.
"""
function grad_sample_size end

"""
    obj_sample_size(rule, st) -> Int

Terms to draw for the objective estimates. Usually larger: the accuracy
requirement on `f` is `O(Δ_k²)` against `O(Δ_k)` on the gradient, so the sample
scales as `Δ_k^{-4}` rather than `Δ_k^{-2}`. On a run whose radius becomes small it
is the objective estimates, not the gradients, that dominate the bill.
"""
function obj_sample_size end

"""
    couples_to_radius(rule) -> Bool

Whether `N_k` depends on `Δ_k`. When it does, the radius mechanism and the sampling
rule stop being independent axes: for a criticality-anchored rule, `ĝ_k` sets `Δ_k`
sets `N_k` sets the accuracy of `ĝ_{k+1}`. `RDelta` has no such loop.
"""
couples_to_radius(::SamplingRule) = false

"""
    needs_scores(rule) -> Bool

Whether the rule requires per-observation scores. Checked once, at oracle
construction, against [`has_scores`](@ref) on the problem.
"""
needs_scores(::SamplingRule) = false

"""
    needs_paired(rule) -> Bool

Whether `rule` consumes the paired decrease statistic, and so whether the solver
must evaluate the trial point on the *same* batch and difference it
observation by observation. `false` by default, so no rule pays for a statistic
it does not read.
"""
needs_paired(::SamplingRule) = false

"""
    record_paired!(rule, δ, σ², N) -> nothing

Hand the rule the paired decrease statistic of the iteration just taken: the
sample mean `δ` and sample variance `σ²` of

    D_i = F(x_k, ξ_i) − F(x_k + s_k, ξ_i)

over the `N` observations of the current batch. A no-op for every rule that does
not declare [`needs_paired`](@ref).

The statistic belongs to the rule rather than to the oracle because the rule is
its only consumer, and because it is state carried across iterations in exactly
the way [`SequentialEstimation`](@ref) carries the predicted reduction.
"""
record_paired!(::SamplingRule, ::Real, ::Real, ::Integer) = nothing

"""
    paired_variance_kind(rule) -> Symbol

Which estimate of `Var(D_i)` the rule wants handed to [`record_paired!`](@ref),
where `D_i = F(x_k, ξ_i) − F(x_k + s_k, ξ_i)`.

- `:empirical` — the sample variance of the `D_i` themselves, computed by
  [`paired_decrease_stats`](@ref). Assumption-free, and it costs one pass over the
  per-observation objective values at both ends of the step.
- `:outer_product` — `s_kᵀ B̃_k s_k − (g̃_kᵀ s_k)²`, computed by
  [`paired_op_variance`](@ref) from the score matrix. This is the first-order
  Taylor approximation: `D_i ≈ −∇F(x_k, ξ_i)ᵀ s_k`, whose variance is
  `s_kᵀ Var[∇F] s_k` with `Var[∇F]` replaced by the outer-product estimate
  `B̃_k − g̃_k g̃_kᵀ`.

The two are different quantities and a rule may prefer either. The approximation
is exact only in the limit of a vanishing step, and it inherits whatever error the
outer product carries as an estimate of `Var[∇F]` — which is a *different*
requirement from the information identity, since it does not ask `B̃` to
approximate the Hessian, only the score covariance.

Default `:empirical`. Only rules that declare [`needs_paired`](@ref) are asked.
"""
paired_variance_kind(::SamplingRule) = :empirical

"""
    requires_finite_population(rule) -> Bool

Whether the rule is meaningful only on a [`FiniteSumProblem`](@ref).

`true` only for [`FullBatch`](@ref): "all of it" is not a sample size an
expectation has. Checked at oracle construction, so pairing it with an expectation
is a constructor error rather than an `N_k = typemax(Int)` allocation.
"""
requires_finite_population(::SamplingRule) = false

"""
    sample_cap(rule) -> Int

The rule's own cap resolved for arithmetic: the user's `N_max` if they set one,
`typemax(Int)` otherwise. The solver combines it with its own `budget` and with
`population(prob)`.

See [`user_cap`](@ref), which preserves the distinction between "unset" and "set to
a large number" — the one [`check_population_cap`](@ref) needs.
"""
sample_cap(r::SamplingRule) = something(user_cap(r), typemax(Int))

"""
    reset_sampling_rule!(rule) -> nothing

Restore mutable state. Default: no-op.
"""
reset_sampling_rule!(::SamplingRule) = nothing

"""
    _check_nmax(name, N_min, N_max)

Validate the optional user budget on a rule. `N_max === nothing` means "no budget
of my own"; the solver supplies one.
"""
function _check_nmax(name::Symbol, N_min::Int, N_max)
    N_min > 0 || throw(ArgumentError("$name: need N_min > 0, got $N_min"))
    N_max === nothing && return nothing
    N_max >= N_min || throw(ArgumentError(
        "$name: need N_min ≤ N_max, got N_min = $N_min, N_max = $N_max"))
    return nothing
end

# -----------------------------------------------------------------------------
# Rules
# -----------------------------------------------------------------------------

"""
    FullBatch()

`N_k = M` at every iteration: the whole finite sum.

A finite-sum run under this rule is **exactly** the deterministic run — ρ̂ = ρ, the
stopping test is the real one, no accuracy hypothesis is needed — which makes it
the reference every other sampling rule should be scored against, and a sharp
regression test: `FiniteSumTRSolver` with `FullBatch()` must reproduce
`DeterministicTRSolver` on `FullBatchNLP` iterate for iterate.

Rejected on an [`ExpectationProblem`](@ref): there is no full batch. That refusal is
the type system carrying the substantive difference between the two classes, not a
technicality — an expectation's accuracy hypotheses can never be discharged by
sampling harder, and this rule is what discharging them looks like.
"""
struct FullBatch <: SamplingRule end

requires_finite_population(::FullBatch) = true

function grad_sample_size(::FullBatch, st::SamplingState)
    st.N_pop == typemax(Int) && throw(ArgumentError(
        "FullBatch has no meaning for an expectation: the population is unbounded."))
    return st.N_pop
end
obj_sample_size(r::FullBatch, st::SamplingState) = grad_sample_size(r, st)

"""
    FixedSample(N; N_obj = N)

Sample-average approximation: the same `N` at every iteration.

The baseline every adaptive rule should be scored against, and the only rule for
which total work is proportional to the iteration count — which is what
iteration-count comparisons quietly assume.
"""
struct FixedSample <: SamplingRule
    N::Int
    N_obj::Int
    function FixedSample(N::Int; N_obj::Int = N)
        N > 0 && N_obj > 0 || throw(ArgumentError("FixedSample: need N, N_obj > 0"))
        new(N, N_obj)
    end
end
grad_sample_size(r::FixedSample, ::SamplingState) = r.N
obj_sample_size(r::FixedSample, ::SamplingState)  = r.N_obj

"""
    RadiusProportional(; κ_g = 1.0, κ_f = 1.0, N_min = 2, N_max = nothing,
                         σ_floor = 1e-12)

The STORM requirement, made operational:

```math
N_k^{\\mathrm{grad}} = \\lceil (\\sigma_g/(\\kappa_g \\Delta_k))^2 \\rceil, \\qquad
N_k^{\\mathrm{obj}}  = \\lceil (\\sigma_f/(\\kappa_f \\Delta_k^2))^2 \\rceil ,
```

so the gradient estimate is accurate to `O(Δ_k)` and the objective estimates to
`O(Δ_k²)` with fixed probability.

This is the rule that makes the mechanism pay for its own radius: `N_k` grows like
`Δ_k^{-2}`, so a criticality-anchored mechanism drives the sample size up without
bound near a solution while a criticality-blind one does not.

`N_max` is an **expectation-only** budget. On a finite sum the cap is `M` and the
solver's `budget` keyword is where a sub-population limit belongs.
"""
struct RadiusProportional <: SamplingRule
    κ_g::Float64
    κ_f::Float64
    N_min::Int
    N_max::Union{Int, Nothing}
    σ_floor::Float64
    function RadiusProportional(; κ_g::Real = 1.0, κ_f::Real = 1.0, N_min::Int = 2,
                                  N_max::Union{Int, Nothing} = nothing,
                                  σ_floor::Real = 1e-12)
        κ_g > 0 && κ_f > 0 || throw(ArgumentError("RadiusProportional: need κ_g, κ_f > 0"))
        _check_nmax(:RadiusProportional, N_min, N_max)
        new(float(κ_g), float(κ_f), N_min, N_max, float(σ_floor))
    end
end
couples_to_radius(::RadiusProportional) = true
user_cap(r::RadiusProportional) = r.N_max

function grad_sample_size(r::RadiusProportional, st::SamplingState)
    σ = sqrt(max(st.σg², r.σ_floor)); Δ = max(st.Δ, 1e-300)
    return _sample_size((σ / (r.κ_g * Δ))^2, r.N_min, sample_cap(r))
end
function obj_sample_size(r::RadiusProportional, st::SamplingState)
    σ = sqrt(max(st.σf², r.σ_floor)); Δ = max(st.Δ, 1e-300)
    return _sample_size((σ / (r.κ_f * Δ^2))^2, r.N_min, sample_cap(r))
end

"""
    NormTest(; θ = 0.5, N_min = 2, N_max = nothing, N_obj = nothing)

Byrd, Chin, Nocedal & Wu (2012): `σ_g²/N_k ≤ θ²‖ĝ_k‖²`.

Never consults `Δ_k`, so the sampling rule and the radius mechanism stay
independent — which makes it the right control for isolating what the coupling in
[`RadiusProportional`](@ref) actually does. It still drives `N_k → ∞`, but through
`‖ĝ_k‖ → 0` rather than through the radius, so the growth is a property of the
problem rather than of the mechanism.
"""
struct NormTest <: SamplingRule
    θ::Float64
    N_min::Int
    N_max::Union{Int, Nothing}
    N_obj::Union{Int, Nothing}
    function NormTest(; θ::Real = 0.5, N_min::Int = 2,
                        N_max::Union{Int, Nothing} = nothing,
                        N_obj::Union{Int, Nothing} = nothing)
        θ > 0 || throw(ArgumentError("NormTest: need θ > 0"))
        _check_nmax(:NormTest, N_min, N_max)
        new(float(θ), N_min, N_max, N_obj)
    end
end
user_cap(r::NormTest) = r.N_max

function grad_sample_size(r::NormTest, st::SamplingState)
    gn = max(st.g_norm, 1e-300)
    return _sample_size(st.σg² / (r.θ^2 * gn^2), r.N_min, sample_cap(r))
end
obj_sample_size(r::NormTest, st::SamplingState) =
    r.N_obj === nothing ? grad_sample_size(r, st) : r.N_obj

"""
    GeometricSample(; N₀ = 8, rate = 1.1, N_max = nothing)

`N_k = ⌈N₀·rate^k⌉`: a schedule fixed in advance, independent of everything the run
observes.

The other control. It reaches any accuracy eventually and cannot respond to the
radius at all, so comparing it against `RadiusProportional` at matched total sample
budget separates "spending more" from "spending it at the right iterations".

`rate^k` overflows to `Inf` around `k ≈ 7000` for `rate = 1.1`, inside the default
iteration budget, which is why the conversion goes through `_sample_size`.
"""
struct GeometricSample <: SamplingRule
    N₀::Int
    rate::Float64
    N_max::Union{Int, Nothing}
    function GeometricSample(; N₀::Int = 8, rate::Real = 1.1,
                               N_max::Union{Int, Nothing} = nothing)
        N₀ > 0    || throw(ArgumentError("GeometricSample: need N₀ > 0"))
        rate >= 1 || throw(ArgumentError("GeometricSample: need rate ≥ 1"))
        _check_nmax(:GeometricSample, N₀, N_max)
        new(N₀, float(rate), N_max)
    end
end
user_cap(r::GeometricSample) = r.N_max
grad_sample_size(r::GeometricSample, st::SamplingState) =
    _sample_size(r.N₀ * r.rate^st.k, 1, sample_cap(r))
obj_sample_size(r::GeometricSample, st::SamplingState) = grad_sample_size(r, st)

"""
    InnerProductTest(; θ = 0.9, N_min = 2, N_max = nothing, N_obj = nothing)

Bollapragada, Byrd & Nocedal (2018): control the variance of the sampled gradient
**along the estimated direction**,

```math
\\frac{1}{N_k}\\operatorname{Var}(\\nabla F_i^\\top \\hat g) \\le \\theta^2 \\|\\hat g\\|^4 .
```

This is the condition that keeps `−ĝ` a descent direction with fixed probability. It
is weaker than the [`NormTest`](@ref), which also pays for error orthogonal to `ĝ` —
error that rotates the direction without threatening descent — so it reaches the
same guarantee at a materially smaller sample.

`θ` is not on the same scale as the norm test's: values near 1 are usual here,
against 0.1–0.5 there. Requires a [`ScoredProblem`](@ref).
"""
struct InnerProductTest <: SamplingRule
    θ::Float64
    N_min::Int
    N_max::Union{Int, Nothing}
    N_obj::Union{Int, Nothing}
    function InnerProductTest(; θ::Real = 0.9, N_min::Int = 2,
                                N_max::Union{Int, Nothing} = nothing,
                                N_obj::Union{Int, Nothing} = nothing)
        θ > 0 || throw(ArgumentError("InnerProductTest: need θ > 0"))
        _check_nmax(:InnerProductTest, N_min, N_max)
        new(float(θ), N_min, N_max, N_obj)
    end
end
needs_scores(::InnerProductTest) = true
user_cap(r::InnerProductTest) = r.N_max
function grad_sample_size(r::InnerProductTest, st::SamplingState)
    gn = max(st.g_norm, 1e-300)
    return _sample_size(st.ip² / (r.θ^2 * gn^4), r.N_min, sample_cap(r))
end
obj_sample_size(r::InnerProductTest, st::SamplingState) =
    r.N_obj === nothing ? grad_sample_size(r, st) : r.N_obj

"""
    OrthogonalityTest(; ν = 2.0, N_min = 2, N_max = nothing, N_obj = nothing)

The companion test: bound the variance **orthogonal** to the estimated direction,
`E‖proj⊥∇Fᵢ‖²/N_k ≤ ν²‖ĝ‖²`.

The inner-product test keeps the sampled direction pointing downhill; this one keeps
it from rotating too far while doing so. Neither implies the other, and the norm
test implies both at a larger sample — the sense in which it is conservative rather
than wrong. Requires a [`ScoredProblem`](@ref).
"""
struct OrthogonalityTest <: SamplingRule
    ν::Float64
    N_min::Int
    N_max::Union{Int, Nothing}
    N_obj::Union{Int, Nothing}
    function OrthogonalityTest(; ν::Real = 2.0, N_min::Int = 2,
                                 N_max::Union{Int, Nothing} = nothing,
                                 N_obj::Union{Int, Nothing} = nothing)
        ν > 0 || throw(ArgumentError("OrthogonalityTest: need ν > 0"))
        _check_nmax(:OrthogonalityTest, N_min, N_max)
        new(float(ν), N_min, N_max, N_obj)
    end
end
needs_scores(::OrthogonalityTest) = true
user_cap(r::OrthogonalityTest) = r.N_max
function grad_sample_size(r::OrthogonalityTest, st::SamplingState)
    gn = max(st.g_norm, 1e-300)
    return _sample_size(st.orth² / (r.ν^2 * gn^2), r.N_min, sample_cap(r))
end
obj_sample_size(r::OrthogonalityTest, st::SamplingState) =
    r.N_obj === nothing ? grad_sample_size(r, st) : r.N_obj

"""
    AugmentedInnerProduct(; θ = 0.9, ν = 2.0, N_min = 2, N_max = nothing, N_obj = nothing)

Both conditions at once — the form Bollapragada, Byrd & Nocedal actually recommend.
Taking the maximum controls the sign and the rotation of the sampled direction
simultaneously while still costing less than bounding the total variance.
Requires a [`ScoredProblem`](@ref).
"""
struct AugmentedInnerProduct <: SamplingRule
    θ::Float64
    ν::Float64
    N_min::Int
    N_max::Union{Int, Nothing}
    N_obj::Union{Int, Nothing}
    function AugmentedInnerProduct(; θ::Real = 0.9, ν::Real = 2.0, N_min::Int = 2,
                                     N_max::Union{Int, Nothing} = nothing,
                                     N_obj::Union{Int, Nothing} = nothing)
        θ > 0 && ν > 0 || throw(ArgumentError("AugmentedInnerProduct: need θ, ν > 0"))
        _check_nmax(:AugmentedInnerProduct, N_min, N_max)
        new(float(θ), float(ν), N_min, N_max, N_obj)
    end
end
needs_scores(::AugmentedInnerProduct) = true
user_cap(r::AugmentedInnerProduct) = r.N_max
function grad_sample_size(r::AugmentedInnerProduct, st::SamplingState)
    gn = max(st.g_norm, 1e-300)
    return _sample_size(max(st.ip² / (r.θ^2 * gn^4), st.orth² / (r.ν^2 * gn^2)),
                        r.N_min, sample_cap(r))
end
obj_sample_size(r::AugmentedInnerProduct, st::SamplingState) =
    r.N_obj === nothing ? grad_sample_size(r, st) : r.N_obj

"""
    SequentialEstimation(; κ = 0.25, α = 0.05, N_min = 8, N_max = nothing,
                           monotone = true, growth = 2.0, N_start = 8)

Bastin, Cirillo & Toint (2006): make the sample large enough that the **noise in
the estimated decrease is small beside the decrease the model predicts**,

```math
N_k \\ge \\frac{2 z_{\\alpha/2}^2 \\hat\\sigma_f^2}{\\kappa^2 \\mathrm{pred}_{k-1}^2}.
```

Two things distinguish it from the gradient-based tests.

**It tracks progress, not criticality.** The accuracy demanded falls as progress
becomes harder to make, rather than as the gradient becomes small.

**It is monotone by default.** A batch that shrinks makes `f̂` jump for reasons
unrelated to the step, and ρ̂ then measures the change of estimator rather than of
objective. `growth` caps how fast it may rise in one iteration.

!!! note "Why the result is cached on `k`"
    `resample!` calls `grad_sample_size` and then `obj_sample_size`, and this is the
    only rule carrying state across iterations. Committing `N_last` inside the query
    applied the growth cap **twice per iteration** and left `N_f > N_g`, which sent
    the resample down the branch that draws *separate* batches and destroyed the
    common random numbers the layer depends on.
"""
mutable struct SequentialEstimation <: SamplingRule
    κ::Float64
    z::Float64
    N_min::Int
    N_max::Union{Int, Nothing}
    monotone::Bool
    growth::Float64
    N_start::Int
    N_last::Int
    k_cached::Int
    N_cached::Int
    function SequentialEstimation(; κ::Real = 0.25, α::Real = 0.05, N_min::Int = 8,
                                    N_max::Union{Int, Nothing} = nothing,
                                    monotone::Bool = true, growth::Real = 2.0,
                                    N_start::Int = 8)
        κ > 0 || throw(ArgumentError("SequentialEstimation: need κ > 0"))
        0 < α < 1 || throw(ArgumentError("SequentialEstimation: need 0 < α < 1"))
        growth > 1 || throw(ArgumentError("SequentialEstimation: need growth > 1"))
        N_start > 0 || throw(ArgumentError("SequentialEstimation: need N_start > 0"))
        _check_nmax(:SequentialEstimation, N_min, N_max)
        new(float(κ), _z_quantile(α), N_min, N_max, monotone, float(growth),
            N_start, 0, -1, 0)
    end
end

couples_to_radius(::SequentialEstimation) = true   # through pred, not a formula in Δ
user_cap(r::SequentialEstimation) = r.N_max

function _sequential_size!(r::SequentialEstimation, st::SamplingState)
    st.k == r.k_cached && return r.N_cached
    cap = sample_cap(r)
    N = if !isfinite(st.pred) || st.pred <= 0
        max(r.N_start, st.N_prev)
    else
        _sample_size(2 * r.z^2 * max(st.σf², 0.0) / (r.κ^2 * st.pred^2), r.N_min, cap)
    end
    if r.monotone
        base = r.N_last > 0 ? r.N_last : r.N_start
        N = max(N, r.N_last)
        N = min(N, _sample_size(r.growth * base, r.N_min, cap))
    end
    N = clamp(N, r.N_min, cap)
    r.N_last = N; r.k_cached = st.k; r.N_cached = N
    return N
end

grad_sample_size(r::SequentialEstimation, st::SamplingState) = _sequential_size!(r, st)
obj_sample_size(r::SequentialEstimation, st::SamplingState)  = _sequential_size!(r, st)
reset_sampling_rule!(r::SequentialEstimation) =
    (r.N_last = 0; r.k_cached = -1; r.N_cached = 0; nothing)

"""
    CertifiedDecrease(; p = 0.9, N_min = 2, N_max = nothing, N_start = 8,
                        monotone = false, growth = 4.0, variance = :empirical)

Sample until the achieved decrease is **certified** against the noise in the
estimate of that decrease, using paired differences under common random numbers.

With the same realisations at both ends of the step,

```math
D_i = F(x_k, \\xi_i) - F(x_k + s_k, \\xi_i),
\\qquad
N_{k+1} = \\left\\lceil z_p^2 \\hat\\sigma_N^2 / \\hat\\delta_N^2 \\right\\rceil ,
```

where `δ̂_N` and `σ̂_N²` are the sample mean and variance of the `D_i`.

# Why pairing is the whole point

Holding `ξ_i` fixed at both points makes `D_i → 0` pathwise as `s_k → 0`, so
`σ_D² = O(‖s_k‖²)`. Two independent batches give `O(1)` instead, and the rule is
then unusable near a solution. On a heavy-tailed sample problem the measured
variance ratio between the paired and unpaired estimators runs to several orders
of magnitude.

# What the rule costs, and why it does not see the radius

Both the numerator and the denominator are linear in the step:
`δ̂ ∼ ‖g_k‖‖s_k‖` and `σ̂ ∼ C‖s_k‖`, so

```math
N_k \\sim \\frac{z_p^2 C^2 \\|s_k\\|^2}{\\|g_k\\|^2 \\|s_k\\|^2}
      = \\frac{z_p^2 C^2}{\\|g_k\\|^2} ,
```

and the step length cancels. Pairing buys a large constant, not a better rate,
and the per-iteration sampling cost is insensitive to the radius rule: mechanisms
differ in how many iterations they take, not in what each costs to certify. This
is the opposite of [`RadiusProportional`](@ref), whose `N_k ∼ Δ_k^{-2}` makes the
mechanism pay for its own radius, and the two together bracket the question of
whether a sampling rule can see the radius at all.

[`couples_to_radius`](@ref) is `true` because the statistic is read off the step;
the cancellation above is a statement about the asymptotic rate, not about the
rule ignoring `s_k`.

!!! warning "The normal quantile is the weak point"
    `z_p` presumes `δ̂_N ≈ 𝒩(δ, σ_D²/N)`. The rule selects a small `N` exactly
    when the decrease is large, which is where that approximation is worst, and
    `D_i` inherits whatever tail the problem has. Read as a fixed-`N` accept
    test the calibration is poor; read as a sequential test, sampling until
    either `δ̂_N ≥ z_p σ̂_N/√N` or `δ̂_N ≤ 0`, it is restored at the price of
    conservatism. An empirical Bernstein bound is the principled replacement,
    and it pays off precisely because pairing has made `σ̂_N` small.

`monotone = false` by default, unlike [`SequentialEstimation`](@ref): the
certified size is meaningful on its own at every iteration, and forcing it upward
hides the `‖g_k‖^{-2}` growth this rule exists to exhibit.

`monotone = true` is the special case `a = 1` of [`SmoothedSize`](@ref), which is
where a two-sided bound `a N_k ≤ N_{k+1} ≤ b N_k` with `a < 1` belongs; wrap this
rule rather than adding a floor to it.

# Which variance

`variance = :empirical` (default) sizes the batch from the sample variance of the
`D_i`. `variance = :outer_product` uses `s_kᵀB̃_ks_k − (g̃_kᵀs_k)²` instead, the
first-order Taylor approximation formed from the score matrix; see
[`paired_variance_kind`](@ref) for what separates them and
[`paired_op_variance`](@ref) for the computation. The outer-product form needs
per-observation scores, so `needs_scores` becomes `true` and the rule is then
rejected on a problem that has none.

Both estimates are recorded on the trace whenever the problem supplies scores,
under `:paired_variance_trajectory` and `:paired_op_variance_trajectory`, so the
choice can be audited against the one not taken.
"""
mutable struct CertifiedDecrease <: SamplingRule
    p::Float64
    z::Float64
    N_min::Int
    N_max::Union{Int, Nothing}
    N_start::Int
    growth::Float64
    monotone::Bool
    δ::Float64
    σ²::Float64
    N_last::Int
    k_cached::Int
    N_cached::Int
    variance::Symbol
    function CertifiedDecrease(; p::Real = 0.9, N_min::Int = 2,
                                 N_max::Union{Int, Nothing} = nothing,
                                 N_start::Int = 8, monotone::Bool = false,
                                 growth::Real = 4.0,
                                 variance::Symbol = :empirical)
        0.5 < p < 1 || throw(ArgumentError("CertifiedDecrease: need 0.5 < p < 1, got $p"))
        growth > 1 || throw(ArgumentError("CertifiedDecrease: need growth > 1"))
        N_start > 0 || throw(ArgumentError("CertifiedDecrease: need N_start > 0"))
        variance in (:empirical, :outer_product) || throw(ArgumentError(
            "CertifiedDecrease: variance must be :empirical or :outer_product, " *
            "got :$variance"))
        _check_nmax(:CertifiedDecrease, N_min, N_max)
        # One-sided level p, so the two-sided quantile is taken at α = 2(1−p).
        new(float(p), _z_quantile(2 * (1 - p)), N_min, N_max, N_start,
            float(growth), monotone, NaN, NaN, 0, -1, 0, variance)
    end
end

needs_paired(::CertifiedDecrease)      = true
couples_to_radius(::CertifiedDecrease) = true
user_cap(r::CertifiedDecrease)         = r.N_max
needs_scores(r::CertifiedDecrease)     = r.variance === :outer_product
paired_variance_kind(r::CertifiedDecrease) = r.variance

function record_paired!(r::CertifiedDecrease, δ::Real, σ²::Real, ::Integer)
    r.δ = Float64(δ); r.σ² = Float64(σ²)
    return nothing
end

function _certified_size!(r::CertifiedDecrease, st::SamplingState)
    st.k == r.k_cached && return r.N_cached
    cap = sample_cap(r)
    prev = st.N_prev > 0 ? st.N_prev : r.N_start
    N = if !isfinite(r.δ) || !isfinite(r.σ²)
        max(r.N_start, prev)                      # nothing measured yet
    elseif r.δ <= 0
        # The batch did not certify a decrease. The size the formula would ask
        # for is undefined or negative, so grow: a failure to certify is
        # evidence that the batch was too small, not that it was too large.
        _sample_size(r.growth * prev, r.N_min, cap)
    else
        _sample_size(r.z^2 * max(r.σ², 0.0) / r.δ^2, r.N_min, cap)
    end
    if r.monotone
        N = max(N, r.N_last)
        N = min(N, _sample_size(r.growth * max(r.N_last, r.N_start), r.N_min, cap))
    end
    N = clamp(N, r.N_min, cap)
    r.N_last = N; r.k_cached = st.k; r.N_cached = N
    return N
end

grad_sample_size(r::CertifiedDecrease, st::SamplingState) = _certified_size!(r, st)
obj_sample_size(r::CertifiedDecrease, st::SamplingState)  = _certified_size!(r, st)
reset_sampling_rule!(r::CertifiedDecrease) =
    (r.δ = NaN; r.σ² = NaN; r.N_last = 0; r.k_cached = -1; r.N_cached = 0; nothing)

# -----------------------------------------------------------------------------
# The normal quantile
# -----------------------------------------------------------------------------

"""
    _z_quantile(α) -> Float64

The two-sided normal quantile `z_{α/2}`, by bisection on `_erf_approx`, so
no distribution package is needed for one constant per rule.
"""
function _z_quantile(α::Real)
    target = 1 - α / 2
    Φ(z) = 0.5 * (1 + _erf_approx(z / sqrt(2)))
    lo, hi = 0.0, 10.0
    for _ in 1:200
        mid = (lo + hi) / 2
        Φ(mid) < target ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

"""
    _erf_approx(x) -> Float64

Abramowitz & Stegun 7.1.26; ~1e-7 absolute, ample for a confidence multiplier.
Underscore-prefixed and unexported: the previous name, `erf` at module scope,
collided with `SpecialFunctions.erf` for anyone loading both.
"""
function _erf_approx(x::Real)
    s = sign(x); ax = abs(float(x))
    t = 1 / (1 + 0.3275911 * ax)
    y = 1 - (((((1.061405429t - 1.453152027) * t) + 1.421413741) * t -
              0.284496736) * t + 0.254829592) * t * exp(-ax * ax)
    return s * y
end

Base.show(io::IO, r::SamplingRule) =
    print(io, nameof(typeof(r)), "(",
          join(("$(f)=$(getfield(r,f))" for f in fieldnames(typeof(r))), ", "), ")")

# -----------------------------------------------------------------------------
# Sample size update functions: smoothing a rule's requirement
# -----------------------------------------------------------------------------

"""
    SmoothedSize(rule; a = 0.75, b = 2.0, N_floor = 1, N_incr = 0, N_start = 8)

Wrap a [`SamplingRule`](@ref) and bound how fast the size it asks for may move.

The inner rule proposes a candidate `N⁺_{k+1}`; this reports

```math
N_{k+1} = \\min\\bigl\\{ b N_k , \\; \\max\\{ a N_k , \\; N^{\\min}_k , \\; N^{+}_{k+1} \\} \\bigr\\} ,
\\qquad N^{\\min}_{k+1} = N^{\\min}_k + N_{incr} ,
```

with `0 < a ≤ 1` and `b > 1`. The two-sided bound is the *naive smoothing* of the
PreDoc, written there as `Naive(a, b)`; `N_floor` and `N_incr` are the growing floor
`N^{min}_k` of its sample-size update step.

# Why it exists

[`CertifiedDecrease`](@ref) already offers `monotone = true`, which is `a = 1`: the
size may never fall. That is a strong assumption, and it is not the one the PreDoc
makes. It also hides what the certified formula does, since a size that can only
rise tells you nothing about the iterations at which the formula asked for less.

`monotone = false` is the other extreme, `a = 0` and `b = ∞`, and there the raw
requirement swings over orders of magnitude between consecutive iterations. `a` and
`b` interpolate, and the interesting settings are strictly between the two.

# What it does not do

It does not change the inner rule's statistic, only the size read off it. A run under
`SmoothedSize` and a run under the bare rule see the same estimates of the same
quantities; they differ in what they do with them. In particular the smoothing is
**not** a variance reduction and should never be described as one.

# When the floor and the ceiling conflict

`N_incr > 0` makes the floor grow linearly while `b N_k` grows geometrically from
whatever `N_k` happens to be, so the floor can exceed the ceiling — most easily
right after a sharp fall. **The floor wins**, and `N_{k+1} = N^{min}_k`. It is a
requirement the caller stated outright, whereas the ceiling is a smoothing
preference; and the alternative would let a run sit below its own stated floor for
as long as it took `b` to catch up.

# Composition

Every trait is forwarded, so the wrapper is transparent to
[`check_rule_problem`](@ref), to the oracle's cap logic and to the solver:
[`needs_paired`](@ref), [`needs_scores`](@ref), [`couples_to_radius`](@ref),
[`requires_finite_population`](@ref), [`user_cap`](@ref) and
[`paired_variance_kind`](@ref) all read through to `rule`, and
[`record_paired!`](@ref) is passed straight down.

The gradient and objective channels are smoothed **separately**, each against its
own previous size, so a rule that distinguishes them — [`RadiusProportional`](@ref)
asks for `Δ^{-2}` and `Δ^{-4}` — keeps the distinction. A rule that returns one size
for both, which is every rule that pairs, is unaffected: the two channels then carry
identical state.

# Example

The PreDoc's setting, a certified size smoothed by `Naive(0.75, 2)` with a floor
starting at 10 and rising by 10 each iteration:

```julia
rule = SmoothedSize(CertifiedDecrease(p = 0.95, N_min = 10);
                    a = 0.75, b = 2.0, N_floor = 10, N_incr = 10, N_start = 100)
```
"""
mutable struct SmoothedSize{R <: SamplingRule} <: SamplingRule
    rule::R
    a::Float64
    b::Float64
    N_floor::Int
    N_incr::Int
    N_start::Int
    g_last::Int
    g_k::Int
    g_N::Int
    f_last::Int
    f_k::Int
    f_N::Int
    function SmoothedSize(rule::R; a::Real = 0.75, b::Real = 2.0,
                          N_floor::Int = 1, N_incr::Int = 0,
                          N_start::Int = 8) where {R <: SamplingRule}
        0 < a <= 1 || throw(ArgumentError("SmoothedSize: need 0 < a ≤ 1, got $a"))
        b > 1 || throw(ArgumentError("SmoothedSize: need b > 1, got $b"))
        N_floor > 0 || throw(ArgumentError("SmoothedSize: need N_floor > 0, got $N_floor"))
        N_incr >= 0 || throw(ArgumentError("SmoothedSize: need N_incr ≥ 0, got $N_incr"))
        N_start > 0 || throw(ArgumentError("SmoothedSize: need N_start > 0, got $N_start"))
        new{R}(rule, float(a), float(b), N_floor, N_incr, N_start,
               0, -1, 0, 0, -1, 0)
    end
end

needs_paired(r::SmoothedSize)               = needs_paired(r.rule)
needs_scores(r::SmoothedSize)               = needs_scores(r.rule)
couples_to_radius(r::SmoothedSize)          = couples_to_radius(r.rule)
requires_finite_population(r::SmoothedSize) = requires_finite_population(r.rule)
user_cap(r::SmoothedSize)                   = user_cap(r.rule)
paired_variance_kind(r::SmoothedSize)       = paired_variance_kind(r.rule)

record_paired!(r::SmoothedSize, δ::Real, σ²::Real, N::Integer) =
    record_paired!(r.rule, δ, σ², N)

"""
    _floor_at(r::SmoothedSize, k) -> Int

`N_floor + k·N_incr`, **saturating**.

The product overflows for a long run with a large increment, and a wrapped `Int`
would hand back a negative floor that `max` then ignores in silence.
"""
@inline function _floor_at(r::SmoothedSize, k::Int)
    (r.N_incr == 0 || k <= 0) && return r.N_floor
    inc = widemul(r.N_incr, k)
    return inc > typemax(Int) - r.N_floor ? typemax(Int) : r.N_floor + Int(inc)
end

"The smoothing itself: bound `N⁺` between `a·prev` (or the floor) and `b·prev`."
@inline function _smooth(r::SmoothedSize, k::Int, prev::Int, N⁺::Int, cap::Int)
    lo = max(_sample_size(r.a * prev, 1, cap), min(_floor_at(r, k), cap))
    hi = _sample_size(r.b * prev, 1, cap)
    return clamp(N⁺, lo, max(lo, hi))          # the floor wins a conflict
end

function grad_sample_size(r::SmoothedSize, st::SamplingState)
    st.k == r.g_k && return r.g_N
    cap  = sample_cap(r)
    prev = r.g_last > 0 ? r.g_last : (st.N_prev > 0 ? st.N_prev : r.N_start)
    N    = _smooth(r, st.k, prev, grad_sample_size(r.rule, st), cap)
    r.g_last = N; r.g_k = st.k; r.g_N = N
    return N
end

function obj_sample_size(r::SmoothedSize, st::SamplingState)
    st.k == r.f_k && return r.f_N
    cap  = sample_cap(r)
    prev = r.f_last > 0 ? r.f_last : (st.N_prev > 0 ? st.N_prev : r.N_start)
    N    = _smooth(r, st.k, prev, obj_sample_size(r.rule, st), cap)
    r.f_last = N; r.f_k = st.k; r.f_N = N
    return N
end

function reset_sampling_rule!(r::SmoothedSize)
    reset_sampling_rule!(r.rule)
    r.g_last = 0; r.g_k = -1; r.g_N = 0
    r.f_last = 0; r.f_k = -1; r.f_N = 0
    return nothing
end
