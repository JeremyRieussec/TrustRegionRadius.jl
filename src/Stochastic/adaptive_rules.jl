# =============================================================================
# src/Stochastic/adaptive_rules.jl
#
# The adaptive sampling rules that condition on second moments of the sampled
# gradient, plus the sequential-estimation rule from the variable-sample-size
# trust-region literature.
#
#   NormTest                ‖g‖-relative total variance      (Byrd et al. 2012)
#   InnerProductTest        variance along ĝ                 (Bollapragada et al. 2018)
#   OrthogonalityTest       variance orthogonal to ĝ         (Bollapragada et al. 2018)
#   AugmentedInnerProduct   both at once — the full test
#   SequentialEstimation    noise small beside the model decrease
#                                                            (Bastin et al. 2006)
#
# ---------------------------------------------------------------------------
# What separates them
#
# Decompose the per-observation gradient variance along and across the estimated
# direction ĝ:
#
#     σ_g² = ip²/‖ĝ‖² + orth²,
#     ip²  = Var(∇Fᵢᵀĝ),   orth² = E‖∇Fᵢ − (∇Fᵢᵀĝ/‖ĝ‖²)ĝ‖² .
#
# The norm test bounds the sum. But only the component *along* ĝ decides whether
# the sampled gradient still points downhill: an error orthogonal to ĝ rotates the
# direction without reversing the sign of ĝᵀ∇f. Controlling the sum is therefore
# stricter than the descent property requires, and the inner-product test is the
# same guarantee at a smaller sample size. The orthogonality test caps the
# rotation, which the inner-product test alone does not.
#
# `SequentialEstimation` is a different idea altogether: it sizes the sample
# against the *model decrease* rather than against the gradient, so the accuracy
# demanded tracks the progress available rather than the criticality. Under a
# radius mechanism that shrinks Δ this decouples the sample size from the radius,
# which `RadiusProportional` deliberately couples — see `couples_to_radius`.
# =============================================================================

# -----------------------------------------------------------------------------
# Moments from a score matrix
# -----------------------------------------------------------------------------

"""
    batch_stats(p::ScoredProblem, x, batch, g) -> SampleStats

All four second moments from one pass over the score matrix.

The parallel/orthogonal split is exact rather than estimated separately:
with `S` the `n × N` score matrix and `ĝ` its column mean,

    ip²   = Var_i( Sᵢᵀ ĝ ),
    orth² = (1/(N−1)) Σᵢ ‖Sᵢ − ĝ‖² − ip²/‖ĝ‖² ,

so the identity `σ_g² = ip²/‖ĝ‖² + orth²` holds to rounding and the two tests are
comparable on the same batch.
"""
function batch_stats(p::ScoredProblem, x, batch, g)
    S = scores(p, x, batch)
    N = size(S, 2)
    N < 2 && return SampleStats(0.0, 0.0, 0.0, 0.0)

    ḡ  = vec(sum(S; dims = 2)) ./ N
    D  = S .- ḡ                                    # centred scores
    σg² = sum(abs2, D) / (N - 1)                   # tr Var(∇Fᵢ)

    gn² = dot(ḡ, ḡ)
    if gn² <= 0
        return SampleStats(σg², obj_variance(p, x, batch), 0.0, σg²)
    end
    proj = vec(ḡ' * D)                             # Sᵢᵀĝ − ‖ĝ‖², already centred
    ip²  = sum(abs2, proj) / (N - 1)               # Var(∇Fᵢᵀĝ)
    orth² = max(σg² - ip² / gn², 0.0)              # the remainder, by the identity
    return SampleStats(σg², obj_variance(p, x, batch), ip², orth²)
end

_needs_scores(rule) = throw(ArgumentError(
    "$(nameof(typeof(rule))) needs the per-observation score matrix to estimate " *
    "Var(∇Fᵢᵀĝ); the problem must be a ScoredProblem. Use NormTest or " *
    "RadiusProportional otherwise."))

# -----------------------------------------------------------------------------
# Inner-product family
# -----------------------------------------------------------------------------

"""
    InnerProductTest(; θ = 0.9, N_min = 2, N_max = 10^6, N_obj = nothing)

The inner-product test of Bollapragada, Byrd & Nocedal (2018): choose `N_k` so
that the variance of the sampled gradient **along the estimated direction** is
controlled relative to `‖ĝ‖⁴`,

```math
\\frac{1}{N_k}\\,\\operatorname{Var}\\bigl(\\nabla F_i(x)^\\top \\hat g\\bigr)
   \\;\\le\\; \\theta^2 \\|\\hat g\\|^4 .
```

This is the condition that keeps `−ĝ` a descent direction for the true objective
with fixed probability. It is weaker than the [`NormTest`](@ref), which bounds the
total variance `σ_g²` and so also pays for error orthogonal to `ĝ` — error that
rotates the direction without threatening descent. In practice `InnerProductTest`
reaches the same guarantee at a materially smaller sample.

`θ` here plays the role `θ` plays in the norm test, but the two are not on the
same scale: the norm test's constraint is `σ_g²/N ≤ θ²‖ĝ‖²` and this one's is
`ip²/N ≤ θ²‖ĝ‖⁴`. Values near 1 are usual here, against 0.1–0.5 for the norm test.

Pair with [`OrthogonalityTest`](@ref) — or use [`AugmentedInnerProduct`](@ref),
which applies both — if the rotation matters as well as the sign.
"""
struct InnerProductTest <: SamplingRule
    θ::Float64
    N_min::Int
    N_max::Int
    N_obj::Union{Int, Nothing}
    function InnerProductTest(; θ::Real = 0.9, N_min::Int = 2, N_max::Int = 10^6,
                                N_obj::Union{Int, Nothing} = nothing)
        θ > 0 || throw(ArgumentError("InnerProductTest: need θ > 0"))
        0 < N_min <= N_max || throw(ArgumentError("InnerProductTest: need 0 < N_min ≤ N_max"))
        new(float(θ), N_min, N_max, N_obj)
    end
end

function grad_sample_size(r::InnerProductTest, st::SamplingState)
    gn = max(st.g_norm, 1e-300)
    st.ip² == 0 && st.orth² == 0 && st.σg² > 0 && _needs_scores(r)
    return clamp(ceil(Int, st.ip² / (r.θ^2 * gn^4)), r.N_min, r.N_max)
end
obj_sample_size(r::InnerProductTest, st::SamplingState) =
    r.N_obj === nothing ? grad_sample_size(r, st) : r.N_obj

"""
    OrthogonalityTest(; ν = 2.0, N_min = 2, N_max = 10^6, N_obj = nothing)

The companion test of Bollapragada, Byrd & Nocedal (2018), bounding the variance
**orthogonal** to the estimated direction:

```math
\\frac{1}{N_k}\\,
\\mathbb E\\Bigl\\|\\nabla F_i(x) - \\frac{\\nabla F_i(x)^\\top \\hat g}{\\|\\hat g\\|^2}\\hat g\\Bigr\\|^2
\\;\\le\\; \\nu^2 \\|\\hat g\\|^2 .
```

The inner-product test keeps the sampled direction pointing downhill; this one
keeps it from rotating too far while doing so. Neither implies the other, and the
norm test implies both at a larger sample size — which is the sense in which it is
conservative rather than wrong.
"""
struct OrthogonalityTest <: SamplingRule
    ν::Float64
    N_min::Int
    N_max::Int
    N_obj::Union{Int, Nothing}
    function OrthogonalityTest(; ν::Real = 2.0, N_min::Int = 2, N_max::Int = 10^6,
                                 N_obj::Union{Int, Nothing} = nothing)
        ν > 0 || throw(ArgumentError("OrthogonalityTest: need ν > 0"))
        0 < N_min <= N_max || throw(ArgumentError("OrthogonalityTest: need 0 < N_min ≤ N_max"))
        new(float(ν), N_min, N_max, N_obj)
    end
end

function grad_sample_size(r::OrthogonalityTest, st::SamplingState)
    gn = max(st.g_norm, 1e-300)
    return clamp(ceil(Int, st.orth² / (r.ν^2 * gn^2)), r.N_min, r.N_max)
end
obj_sample_size(r::OrthogonalityTest, st::SamplingState) =
    r.N_obj === nothing ? grad_sample_size(r, st) : r.N_obj

"""
    AugmentedInnerProduct(; θ = 0.9, ν = 2.0, N_min = 2, N_max = 10^6, N_obj = nothing)

Both conditions at once — the *augmented* inner-product test, and the form
Bollapragada, Byrd & Nocedal actually recommend:

```math
N_k = \\max\\Bigl\\{\\,
  \\frac{\\operatorname{Var}(\\nabla F_i^\\top\\hat g)}{\\theta^2\\|\\hat g\\|^4},\\;
  \\frac{\\mathbb E\\|\\text{proj}_{\\perp}\\nabla F_i\\|^2}{\\nu^2\\|\\hat g\\|^2}
\\,\\Bigr\\} .
```

Taking the maximum, rather than either alone, controls the sign and the rotation
of the sampled direction simultaneously while still costing less than bounding
the total variance as [`NormTest`](@ref) does.
"""
struct AugmentedInnerProduct <: SamplingRule
    θ::Float64
    ν::Float64
    N_min::Int
    N_max::Int
    N_obj::Union{Int, Nothing}
    function AugmentedInnerProduct(; θ::Real = 0.9, ν::Real = 2.0, N_min::Int = 2,
                                     N_max::Int = 10^6,
                                     N_obj::Union{Int, Nothing} = nothing)
        θ > 0 && ν > 0 || throw(ArgumentError("AugmentedInnerProduct: need θ, ν > 0"))
        0 < N_min <= N_max || throw(ArgumentError("AugmentedInnerProduct: need 0 < N_min ≤ N_max"))
        new(float(θ), float(ν), N_min, N_max, N_obj)
    end
end

function grad_sample_size(r::AugmentedInnerProduct, st::SamplingState)
    gn = max(st.g_norm, 1e-300)
    Nip = st.ip²   / (r.θ^2 * gn^4)
    Nor = st.orth² / (r.ν^2 * gn^2)
    return clamp(ceil(Int, max(Nip, Nor)), r.N_min, r.N_max)
end
obj_sample_size(r::AugmentedInnerProduct, st::SamplingState) =
    r.N_obj === nothing ? grad_sample_size(r, st) : r.N_obj

# -----------------------------------------------------------------------------
# Sequential estimation
# -----------------------------------------------------------------------------

"""
    SequentialEstimation(; κ = 0.25, α = 0.05, N_min = 8, N_max = 10^6,
                           monotone = true, growth = 2.0, N_start = 8)

The variable-sample-size rule of the sample-average-approximation literature
(Bastin, Cirillo & Toint 2006; Deng & Ferris 2009): make the sample large enough
that the **noise in the estimated decrease is small beside the decrease the model
predicts**.

With `pred_{k-1}` the predicted reduction of the previous step and `σ̂_f²` the
sample variance of the objective terms, the half-width of a `1−α` confidence
interval on the estimated decrease is `z_{α/2} σ̂_f √(2/N)` under independent
batches, and the rule asks for

```math
z_{\\alpha/2}\\,\\hat\\sigma_f \\sqrt{2/N_k} \\;\\le\\; \\kappa \\cdot \\mathrm{pred}_{k-1}
\\qquad\\Longrightarrow\\qquad
N_k \\;\\ge\\; \\frac{2 z_{\\alpha/2}^2\\,\\hat\\sigma_f^2}{\\kappa^2\\,\\mathrm{pred}_{k-1}^2}.
```

Two things distinguish it from the gradient-based tests.

**It tracks progress, not criticality.** `pred_k` is what the model claims to be
able to achieve, so the accuracy demanded falls as progress becomes harder to
make, rather than as the gradient becomes small. Near a solution where the model
still predicts healthy decrease, this is far cheaper than
[`RadiusProportional`](@ref); where the model predicts almost nothing, it is more
expensive, and correctly so — that is where noise most easily masquerades as
progress.

**It is monotone by default.** `monotone = true` forbids the sample size from
decreasing, which is standard in this literature: a batch that shrinks makes
`f̂` jump for reasons unrelated to the step, and the ratio ρ̂ then measures the
change of estimator rather than the change of objective. `growth` caps how fast it
may rise in one iteration, so a single small `pred` cannot demand the whole
population at once.

The first iteration has no `pred` to condition on and uses `N_start`.
"""
mutable struct SequentialEstimation <: SamplingRule
    κ::Float64
    z::Float64
    N_min::Int
    N_max::Int
    monotone::Bool
    growth::Float64
    N_start::Int
    N_last::Int
    function SequentialEstimation(; κ::Real = 0.25, α::Real = 0.05, N_min::Int = 8,
                                    N_max::Int = 10^6, monotone::Bool = true,
                                    growth::Real = 2.0, N_start::Int = 8)
        κ > 0 || throw(ArgumentError("SequentialEstimation: need κ > 0"))
        0 < α < 1 || throw(ArgumentError("SequentialEstimation: need 0 < α < 1"))
        growth > 1 || throw(ArgumentError("SequentialEstimation: need growth > 1"))
        0 < N_min <= N_max || throw(ArgumentError("SequentialEstimation: need 0 < N_min ≤ N_max"))
        new(float(κ), _z_quantile(α), N_min, N_max, monotone, float(growth),
            N_start, 0)
    end
end

"""
    _z_quantile(α) -> Float64

The two-sided normal quantile `z_{α/2}`, by bisection on the error function so
that no distribution package is needed for four constants.
"""
function _z_quantile(α::Real)
    target = 1 - α / 2
    Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
    lo, hi = 0.0, 10.0
    for _ in 1:200
        mid = (lo + hi) / 2
        Φ(mid) < target ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

"Abramowitz & Stegun 7.1.26; ~1e-7 absolute, ample for a confidence multiplier."
function erf(x::Real)
    s = sign(x); x = abs(float(x))
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - (((((1.061405429t - 1.453152027) * t) + 1.421413741) * t -
              0.284496736) * t + 0.254829592) * t * exp(-x * x)
    return s * y
end

function grad_sample_size(r::SequentialEstimation, st::SamplingState)
    N = if !isfinite(st.pred) || st.pred <= 0
        max(r.N_start, st.N_prev)          # no usable prediction yet
    else
        ceil(Int, 2 * r.z^2 * max(st.σf², 0.0) / (r.κ^2 * st.pred^2))
    end
    if r.monotone
        # The growth bound is measured from `N_start` on the first adaptive call,
        # not from zero: otherwise one small `pred` can demand the whole
        # population in a single step, and the cap that is supposed to make the
        # ramp gradual never binds.
        base = r.N_last > 0 ? r.N_last : r.N_start
        N = max(N, r.N_last)
        N = min(N, ceil(Int, r.growth * base))
    end
    N = clamp(N, r.N_min, r.N_max)
    r.N_last = N
    return N
end
obj_sample_size(r::SequentialEstimation, st::SamplingState) = grad_sample_size(r, st)

# `pred_k` is a property of the model and the radius together, so this rule does
# depend on Δ — but indirectly, through what the model can achieve inside the
# region, rather than through a formula in Δ. Reported as coupled, since the
# radius mechanism does influence the sample size.
couples_to_radius(::SequentialEstimation) = true

reset_sampling_rule!(r::SequentialEstimation) = (r.N_last = 0; nothing)
reset_sampling_rule!(::SamplingRule) = nothing
