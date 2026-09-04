# =============================================================================
# src/Sampling/oracles.jl
#
# The NLP oracles: what the solver actually calls.
#
#   FullBatchNLP    <: AbstractNLPModel     deterministic view of a finite sum
#   SampledNLP      abstract
#     ├── ExpectationNLP                    unbounded population
#     └── FiniteSumNLP                      population M, full batch reachable
#
# All three satisfy the ordinary NLP interface, so every radius mechanism, model
# Hessian and subproblem solver runs over them unchanged. What differs is which
# solver accepts them and what the sampling layer is allowed to do:
#
#                    solver                    cap on N_k        full batch?
#   FullBatchNLP     DeterministicTRSolver     —                 always
#   FiniteSumNLP     FiniteSumTRSolver         M (or budget)     yes
#   ExpectationNLP   ExpectationTRSolver       budget only       never
#
# `FullBatchNLP` is deterministic *and* carries the finite-sum problem underneath,
# which is what keeps `BHHHModel` legal over it: BHHH's requirement is a statement
# about the problem being a likelihood, not about randomness being present.
# =============================================================================

# -----------------------------------------------------------------------------
# The deterministic view
# -----------------------------------------------------------------------------

"""
    FullBatchNLP(prob; x0)

The exact view of a [`FiniteSumProblem`](@ref): every term evaluated at every call.

Deterministic, so it goes to [`DeterministicTRSolver`](@ref) — while
[`underlying_problem`](@ref) still reports the finite-sum problem, so
`required_problem` checks pass and `BHHHModel` over a likelihood remains legal.
That combination is the reason this is a separate type rather than a
`FiniteSumNLP` with `FullBatch()`: the two are numerically identical and
*structurally* different, and the second still carries a sampling rule, a batch, a
variance estimate and an RNG that mean nothing.

The dense Hessian is cached per iterate: `batch_hess` is a pure function of `x`
here, and on `MLPClassifier` it costs `O(n)` gradient evaluations, so recomputing it
inside every `hprod!` made `ExactHessian` unusable on the problems the BHHH
comparison needs it for.

The former name `LikelihoodNLP` remains as an alias.
"""
mutable struct FullBatchNLP{P <: FiniteSumProblem} <: AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModelMeta{Float64, Vector{Float64}}
    counters::Counters
    prob::P
    all::Vector{Int}
    H_cache::Matrix{Float64}
    H_x::Vector{Float64}
    H_ok::Bool
end

function FullBatchNLP(prob::FiniteSumProblem; x0::AbstractVector = zeros(prob.n))
    meta = NLPModelMeta(prob.n; x0 = Vector{Float64}(x0), name = "FullBatchNLP")
    return FullBatchNLP{typeof(prob)}(meta, Counters(), prob, full_batch(prob),
                                      zeros(0, 0), Float64[], false)
end

"""
    LikelihoodNLP

Alias of [`FullBatchNLP`](@ref), kept because the type was called that before the
problem classes were separated. The new name says what it is — the *full-batch*, and
therefore deterministic, view of a finite sum — rather than what it is usually used
for.
"""
const LikelihoodNLP = FullBatchNLP

underlying_problem(m::FullBatchNLP) = m.prob
problem_class(::FullBatchNLP) = :deterministic

NLPModels.obj(m::FullBatchNLP, x::AbstractVector) =
    (m.counters.neval_obj += 1; batch_obj(m.prob, x, m.all))

function NLPModels.grad!(m::FullBatchNLP, x::AbstractVector, g::AbstractVector)
    m.counters.neval_grad += 1
    batch_grad!(m.prob, x, m.all, g)
    return g
end

# -----------------------------------------------------------------------------
# The sampled oracles
# -----------------------------------------------------------------------------

"""
    SampledNLP{T, V} <: AbstractNLPModel{T, V}

Common supertype of [`ExpectationNLP`](@ref) and [`FiniteSumNLP`](@ref). Everything
that does not depend on the population — evaluation, the Hessian cache, the
counters, the variance estimates, `reset_sampling!` — is defined once here.

# Common random numbers

The batch is redrawn once per iteration, in [`resample!`](@ref), and then held fixed.
Every evaluation within the iteration — `f̂(x_k)`, `f̂(x_k + s_k)`, and the
retrospective `f̂` if the rule needs it — uses the *same* realisations.

That is not an optimisation. With independent draws the variance of
`f̂(x_k) − f̂(x_k + s_k)` does not shrink as `‖s_k‖ → 0`, so ρ̂ becomes pure noise at
small radii and every mechanism stalls for a reason that has nothing to do with the
mechanism.
"""
abstract type SampledNLP{T, V} <: AbstractNLPModel{T, V} end

"""
    ExpectationNLP(prob, rule; x0, seed = 0, N_init = 8)

Oracle for an [`ExpectationProblem`](@ref): each `resample!` draws `N_k` **fresh**
realisations.

There is no population cap of the problem's own, so the budget comes from the rule's
`N_max` and from [`ExpectationTRSolver`](@ref)'s `budget`. There is no full-batch
iteration and, unless `has_truth(prob)`, no exact gradient — so a run here can be
scored only against whatever the construction supplies.
"""
mutable struct ExpectationNLP{P <: ExpectationProblem, S <: SamplingRule} <:
               SampledNLP{Float64, Vector{Float64}}
    meta::NLPModelMeta{Float64, Vector{Float64}}
    counters::Counters
    prob::P
    rule::S
    rng::MersenneTwister
    seed::Int
    N_init::Int
    batch_g::Any
    batch_f::Any
    Ng::Int
    Nf::Int
    samples_g::Int
    samples_f::Int
    samples_confirm::Int
    Ng_hist::Vector{Int}
    Nf_hist::Vector{Int}
    σg²::Float64
    σf²::Float64
    ip²::Float64
    orth²::Float64
    last_pred::Float64
    capped::Int
    budget::Int
    H_cache::Matrix{Float64}
    H_x::Vector{Float64}
    H_ok::Bool
    scheme::Symbol
    pool::Any
end

"""
    FiniteSumNLP(prob, rule; x0, seed = 0, replace = true, N_init = 8, budget = nothing)

Oracle for a [`FiniteSumProblem`](@ref): each `resample!` subsets `1:M`.

`N_k ≤ M` always. A rule carrying a user-supplied `N_max` is rejected here — see
[`check_population_cap`](@ref) — because on a finite sum the cap is a property of the
problem; `budget` is where a deliberate sub-population limit belongs.

`:full_batch_trajectory` records the iterations at which `N_k = M`, which are exactly
the iterations that were deterministic.
"""
mutable struct FiniteSumNLP{P <: FiniteSumProblem, S <: SamplingRule} <:
               SampledNLP{Float64, Vector{Float64}}
    meta::NLPModelMeta{Float64, Vector{Float64}}
    counters::Counters
    prob::P
    rule::S
    rng::MersenneTwister
    seed::Int
    replace::Bool
    N_init::Int
    batch_g::Vector{Int}
    batch_f::Vector{Int}
    Ng::Int
    Nf::Int
    samples_g::Int
    samples_f::Int
    samples_confirm::Int
    Ng_hist::Vector{Int}
    Nf_hist::Vector{Int}
    full_hist::Vector{Bool}
    σg²::Float64
    σf²::Float64
    ip²::Float64
    orth²::Float64
    last_pred::Float64
    capped::Int
    budget::Int
    H_cache::Matrix{Float64}
    H_x::Vector{Float64}
    H_ok::Bool
    scheme::Symbol
    pool::Any
end

function ExpectationNLP(prob::ExpectationProblem, rule::SamplingRule;
                        x0::AbstractVector = zeros(prob.n), seed::Int = 0,
                        N_init::Int = 8, budget::Int = 1_000_000,
                        scheme::Symbol = :independent)
    check_rule_problem(rule, prob)
    _check_scheme(scheme, false)
    requires_finite_population(rule) && throw(ArgumentError(
        "$(nameof(typeof(rule))) needs a finite population; " *
        "$(nameof(typeof(prob))) is an expectation. Use FixedSample, " *
        "RadiusProportional, NormTest or GeometricSample."))
    budget > 0 || throw(ArgumentError("ExpectationNLP: need budget > 0"))
    meta = NLPModelMeta(prob.n; x0 = Vector{Float64}(x0), name = "ExpectationNLP")
    rng = MersenneTwister(seed)
    N0 = min(N_init, budget)
    b = draw_batch(prob, rng, N0)
    return ExpectationNLP{typeof(prob), typeof(rule)}(
        meta, Counters(), prob, rule, rng, seed, N0, b, b, N0, N0, 0, 0, 0,
        Int[], Int[], 1.0, 1.0, 0.0, 0.0, NaN, 0, budget,
        zeros(0, 0), Float64[], false, scheme, scheme === :nested ? b : nothing)
end

function FiniteSumNLP(prob::FiniteSumProblem, rule::SamplingRule;
                      x0::AbstractVector = zeros(prob.n), seed::Int = 0,
                      replace::Bool = true, N_init::Int = 8,
                      budget::Union{Int, Nothing} = nothing,
                      scheme::Symbol = :independent)
    check_rule_problem(rule, prob)
    check_population_cap(rule, prob)
    _check_scheme(scheme, true)
    M = population(prob)
    cap = budget === nothing ? M : min(budget, M)
    cap > 0 || throw(ArgumentError("FiniteSumNLP: need budget > 0"))
    meta = NLPModelMeta(prob.n; x0 = Vector{Float64}(x0), name = "FiniteSumNLP")
    rng = MersenneTwister(seed)
    N0 = min(N_init, cap)
    # `:prefix` never draws: the batch is the first N_k individuals from the start.
    b = scheme === :prefix ? collect(1:N0) : draw_batch(prob, rng, N0; replace = replace)
    return FiniteSumNLP{typeof(prob), typeof(rule)}(
        meta, Counters(), prob, rule, rng, seed, replace, N0, copy(b), copy(b),
        N0, N0, 0, 0, 0, Int[], Int[], Bool[], 1.0, 1.0, 0.0, 0.0, NaN, 0, cap,
        zeros(0, 0), Float64[], false, scheme,
        scheme === :nested ? copy(b) : nothing)
end

underlying_problem(m::SampledNLP) = m.prob
problem_class(m::ExpectationNLP)  = :expectation
problem_class(m::FiniteSumNLP)    = :finite_sum

"""
    population_cap(m) -> Int

The largest sample the oracle will ever draw: the solver budget for an expectation,
`min(budget, M)` for a finite sum. `N_k` is clamped to it after the rule has spoken,
and hitting it is counted in `m.capped` — a run spent at the cap is no longer meeting
the accuracy requirement the convergence theory assumes.
"""
population_cap(m::ExpectationNLP) = m.budget
population_cap(m::FiniteSumNLP)   = m.budget

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

NLPModels.obj(m::SampledNLP, x::AbstractVector) =
    (m.counters.neval_obj += 1; batch_obj(m.prob, x, m.batch_f))

function NLPModels.grad!(m::SampledNLP, x::AbstractVector, g::AbstractVector)
    m.counters.neval_grad += 1
    batch_grad!(m.prob, x, m.batch_g, g)
    return g
end

"""
    _cached_batch_hess(m, x) -> Matrix

The batch Hessian at `x`, computed once per `(iterate, batch)` pair.

`batch_hess` is a pure function of `(x, batch)`, so recomputing it per
Hessian-vector product was pure waste — and on the finite-difference problems it was
`O(n)` gradient evaluations of waste, per CG iteration.
"""
function _cached_batch_hess(m::Union{SampledNLP, FullBatchNLP}, x::AbstractVector)
    if m.H_ok && length(m.H_x) == length(x) && m.H_x == x
        return m.H_cache
    end
    batch = m isa FullBatchNLP ? m.all : m.batch_g
    H = Matrix{Float64}(batch_hess(m.prob, x, batch))
    m.H_cache = H; m.H_x = Vector{Float64}(x); m.H_ok = true
    return H
end

for TT in (:SampledNLP, :FullBatchNLP)
    @eval function NLPModels.hess(m::$TT, x::AbstractVector)
        m.counters.neval_hess += 1
        return Symmetric(_cached_batch_hess(m, x))
    end
    @eval function NLPModels.hprod!(m::$TT, x::AbstractVector, v::AbstractVector,
                                    Hv::AbstractVector; obj_weight = 1.0)
        m.counters.neval_hprod += 1
        mul!(Hv, _cached_batch_hess(m, x), v)
        obj_weight == 1 || (Hv .*= obj_weight)
        return Hv
    end
end

# -----------------------------------------------------------------------------
# Resampling
# -----------------------------------------------------------------------------

"""
    resample!(m::SampledNLP, k, Δ, g_norm) -> (Ng, Nf)

Choose `N_k` from the sampling rule and draw the batches for iteration `k`.

Called once per iteration by the solver, before anything is evaluated. The variance
estimates fed to the rule are those of the *previous* batch, since the new one does
not exist yet — the usual plug-in, and the reason `N_min` matters: a rule cannot
estimate a variance from one sample.

The rule states a requirement; the oracle applies the cap. For an expectation that is
the budget, for a finite sum `min(budget, M)`.
"""
function resample!(m::SampledNLP, k::Int, Δ::Real, g_norm::Real)
    cap = population_cap(m)
    st = SamplingState(k, Float64(Δ), Float64(g_norm), m.σg², m.σf²,
                       m.ip², m.orth², m.last_pred, m.Ng, _pop(m))
    Ng = min(grad_sample_size(m.rule, st), cap)
    Nf = min(obj_sample_size(m.rule, st), cap)
    (Ng >= cap || Nf >= cap) && (m.capped += 1)

    m.Ng, m.Nf = Ng, Nf
    _draw!(m, Ng, Nf)
    m.H_ok = false                       # the batch changed; the cache is stale
    m.samples_g += Ng; m.samples_f += Nf
    push!(m.Ng_hist, Ng); push!(m.Nf_hist, Nf)
    _record_full!(m, Ng, Nf)
    return Ng, Nf
end

_pop(m::ExpectationNLP) = typemax(Int)
_pop(m::FiniteSumNLP)   = population(m.prob)

"""
    _draw!(m, Ng, Nf) -> nothing

Install the batches for this iteration, under the oracle's [`sample_scheme`](@ref).

Objective and gradient share the batch when the sizes agree, which is the common
case and gives the tightest common random numbers. When they differ, the smaller is
a subset of the larger under `:nested` and `:prefix`, and independent under
`:independent` — matching what each scheme does across iterations.
"""
function _draw!(m::ExpectationNLP, Ng::Int, Nf::Int)
    if m.scheme === :nested
        _resize_pool!(m, max(Ng, Nf))
        m.batch_g = _take(m.pool, Ng)
        m.batch_f = Ng == Nf ? m.batch_g : _take(m.pool, Nf)
        return nothing
    end
    m.batch_g = draw_batch(m.prob, m.rng, Ng)
    m.batch_f = Ng == Nf ? m.batch_g : draw_batch(m.prob, m.rng, Nf)
    return nothing
end

function _draw!(m::FiniteSumNLP, Ng::Int, Nf::Int)
    if m.scheme === :prefix
        # 𝒩_k = {1, …, N_k}: the first N_k individuals, no random numbers consumed.
        M = population(m.prob)
        m.batch_g = collect(1:min(Ng, M))
        m.batch_f = Ng == Nf ? m.batch_g : collect(1:min(Nf, M))
        return nothing
    elseif m.scheme === :nested
        _resize_pool!(m, max(Ng, Nf))
        m.batch_g = _take(m.pool, Ng)
        m.batch_f = Ng == Nf ? m.batch_g : _take(m.pool, Nf)
        return nothing
    end
    m.batch_g = draw_batch(m.prob, m.rng, Ng; replace = m.replace)
    m.batch_f = Ng == Nf ? m.batch_g : draw_batch(m.prob, m.rng, Nf; replace = m.replace)
    return nothing
end

_record_full!(::ExpectationNLP, ::Int, ::Int) = nothing
_record_full!(m::FiniteSumNLP, Ng::Int, Nf::Int) =
    (push!(m.full_hist, Ng == population(m.prob) && Nf == population(m.prob)); nothing)

"""
    update_variances!(m::SampledNLP, x, g) -> (σg², σf²)

Refresh the variance estimates from the current batch, at the current iterate.

Takes the batch gradient `g` the solver has already computed instead of recomputing
it: this previously evaluated the score matrix twice — once through `batch_grad!` and
once inside `batch_stats`, which then ignored the `g` it was handed — on top of the
solver's own gradient call.
"""
function update_variances!(m::SampledNLP, x::AbstractVector, g::AbstractVector)
    stt = batch_stats(m.prob, x, m.batch_g, g)
    m.σg², m.ip², m.orth² = stt.σg², stt.ip², stt.orth²
    m.σf² = m.batch_f === m.batch_g ? stt.σf² : obj_variance(m.prob, x, m.batch_f)
    return m.σg², m.σf²
end

"""
    record_prediction!(m::SampledNLP, pred) -> pred

Store the model's predicted reduction. [`SequentialEstimation`](@ref) sizes the next
batch against it; no other rule reads it.
"""
record_prediction!(m::SampledNLP, pred::Real) = (m.last_pred = Float64(pred); pred)

"""
    paired_decrease_stats(m::SampledNLP, x, x_cand) -> (δ, σ², N)

The sample mean and variance of the paired differences

    D_i = F(x, ξ_i) − F(x_cand, ξ_i),   i ∈ batch_f,

over the objective batch of the current iteration.

Both points are evaluated on the **same** realisations, which is what makes
`σ_D² = O(‖x_cand − x‖²)`; see [`obs_objective`](@ref). Consumed by
[`CertifiedDecrease`](@ref) through [`record_paired!`](@ref).

# Cost

None beyond what the iteration already pays. The solver has already evaluated
`f(x)` and `f(x_cand)` as batch means over this batch; this recomputes the same
two quantities at per-observation granularity. It is not charged to
[`samples_used`](@ref), because no new realisation is drawn.

Returns `(NaN, NaN, N)` when the batch is too small for a sample variance.
"""
function paired_decrease_stats(m::SampledNLP, x::AbstractVector, x_cand::AbstractVector)
    b  = m.batch_f
    v0 = obs_objective(m.prob, x, b)
    v1 = obs_objective(m.prob, x_cand, b)
    N  = length(v0)
    N < 2 && return (NaN, NaN, N)
    δ = 0.0
    @inbounds for i in 1:N
        δ += v0[i] - v1[i]
    end
    δ /= N
    σ² = 0.0
    @inbounds for i in 1:N
        d = v0[i] - v1[i] - δ
        σ² += d * d
    end
    return (δ, σ² / (N - 1), N)
end

"""
    samples_used(m) -> (grad = …, obj = …, confirm = …, total = …)

Cumulative term evaluations — the cost measure for a sampled comparison. Iteration
counts are proportional to work only under [`FixedSample`](@ref).

`confirm` is what [`confirm_gradient_norm!`](@ref) spent verifying candidate stops,
and is zero unless `TRParams(true_stop = …)` asked for verification. It is separated
out because it is *not* optimisation work — comparing a mechanism that confirms
against one that does not, on `total`, would charge the first for being honest — but
it is included in `total`, because it is work that was done.
"""
samples_used(m::SampledNLP) =
    (grad = m.samples_g, obj = m.samples_f, confirm = m.samples_confirm,
     total = m.samples_g + m.samples_f + m.samples_confirm)

"""
    confirm_gradient_norm!(m, x) -> Float64

`‖∇f(x)‖` as accurately as this oracle can produce it, for confirming a candidate
stopping test. Mutates `m`: the cost is added to `m.samples_confirm`.

Two paths, and which one applies is a property of the problem, not a choice:

- [`has_truth`](@ref) — the exact gradient, via [`true_gradient`](@ref). Every finite
  sum qualifies (one pass over `M`), as does an expectation supplying a closed-form
  mean, which costs nothing per term and is charged nothing.
- otherwise — the gradient over a freshly drawn batch of [`population_cap`](@ref)
  terms, the largest the budget allows. This is still an estimate; it is simply the
  best one available, and on an expectation there is nothing better.

The finite-sum truth path draws no random numbers, so a confirmed run and an
unconfirmed one see the *same* realisations and remain comparable iterate for
iterate. The cap-batch path does consume the stream, so an expectation without truth
cannot be compared that way — confirmation there changes the run it is confirming.
"""
function confirm_gradient_norm!(m::SampledNLP, x::AbstractVector)
    if has_truth(m)
        m.samples_confirm += _truth_cost(m)
        return Float64(norm(true_gradient(m, x)))
    end
    cap = population_cap(m)
    b = _confirm_batch(m, cap)
    g = similar(Vector{Float64}, length(x))
    batch_grad!(m.prob, x, b, g)
    m.samples_confirm += cap
    return Float64(norm(g))
end

# One pass over the population for a finite sum; a closed-form mean is free.
_truth_cost(m::FiniteSumNLP)   = population(m.prob)
_truth_cost(::ExpectationNLP)  = 0

_confirm_batch(m::ExpectationNLP, cap::Int) = draw_batch(m.prob, m.rng, cap)
_confirm_batch(m::FiniteSumNLP, cap::Int) =
    draw_batch(m.prob, m.rng, cap; replace = m.replace)

"""
    grad_standard_error(m) -> Float64

The plug-in standard error of the batch gradient, from the variance estimate the
oracle already maintains and the batch size that produced it.

This is what makes the statistical stopping modes free — no extra evaluation, only
arithmetic on numbers [`update_variances!`](@ref) has already computed. It is a
*plug-in* estimate and inherits the usual caveat: `σ_g²` comes from the previous
batch, and on a batch of two it is itself a one-degree-of-freedom quantity. It is
used to decide whether a stop is *believable*, never to decide the step.

# The finite population correction is not optional here

For an expectation the population is unbounded and the answer is `σ_g/√N_k`. For a
finite sum it is not, and using `σ_g/√N_k` there makes the statistical test useless
rather than merely conservative: it would report a positive sampling error even at
`N_k = M`, where the batch **is** the population and the gradient is exact. The
practical effect is that no tolerance below `z·σ_g/√M` could ever be certified — on
a population of 20 000 with `σ_g ≈ 1` that is about `1.4e-2`, so every realistic
`tol` is unreachable and `:both` degenerates into always paying for `:full`.

So: zero at `N_k ≥ M`, since [`draw_batch`](@ref) returns the whole population there
whatever `replace` says; the textbook `√((M−N)/(M−1))` correction below that when
drawing without replacement; and no correction when drawing with replacement, where
the draws are genuinely i.i.d. and `σ_g/√N_k` is right.
"""
grad_standard_error(m::ExpectationNLP) = sqrt(max(m.σg², 0.0) / max(m.Ng, 1))

function grad_standard_error(m::FiniteSumNLP)
    M = population(m.prob)
    N = max(m.Ng, 1)
    N >= M && return 0.0                       # the batch is the population
    se = sqrt(max(m.σg², 0.0) / N)
    return m.replace ? se : se * sqrt((M - N) / (M - 1))
end

"""
    reset_sampling!(m::SampledNLP) -> m

Restore the counters, the histories, the batches and the random stream to their
initial state, so the same oracle can be reused across configurations and each run
sees the same realisations — which is what makes a comparison of mechanisms under
noise a comparison of mechanisms rather than of random seeds.
"""
function reset_sampling!(m::SampledNLP)
    m.rng = MersenneTwister(m.seed)
    N0 = min(m.N_init, population_cap(m))
    _reset_batches!(m, N0)
    m.Ng = N0; m.Nf = N0
    m.samples_g = 0; m.samples_f = 0; m.samples_confirm = 0
    empty!(m.Ng_hist); empty!(m.Nf_hist)
    m isa FiniteSumNLP && empty!(m.full_hist)
    m.σg² = 1.0; m.σf² = 1.0; m.ip² = 0.0; m.orth² = 0.0
    m.last_pred = NaN; m.capped = 0; m.H_ok = false
    reset_sampling_rule!(m.rule)
    return m
end

function _reset_batches!(m::ExpectationNLP, N0::Int)
    b = draw_batch(m.prob, m.rng, N0); m.batch_g = b; m.batch_f = b
    m.pool = m.scheme === :nested ? b : nothing
    return nothing
end
function _reset_batches!(m::FiniteSumNLP, N0::Int)
    # `:prefix` is deterministic and must not consume the stream on a reset either,
    # or two runs of the same oracle would not see the same realisations.
    b = m.scheme === :prefix ? collect(1:min(N0, population(m.prob))) :
        draw_batch(m.prob, m.rng, N0; replace = m.replace)
    m.batch_g = copy(b); m.batch_f = copy(b)
    m.pool = m.scheme === :nested ? copy(b) : nothing
    return nothing
end

# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------

has_truth(m::SampledNLP)    = has_truth(m.prob)
has_truth(m::FullBatchNLP)  = true

true_objective(m::Union{SampledNLP, FullBatchNLP}, x) = true_objective(m.prob, x)
true_gradient(m::Union{SampledNLP, FullBatchNLP}, x)  = true_gradient(m.prob, x)

Base.show(io::IO, m::FullBatchNLP) =
    print(io, "FullBatchNLP(", nameof(typeof(m.prob)), ", M=", length(m.all), ")")
Base.show(io::IO, m::ExpectationNLP) =
    print(io, "ExpectationNLP(", nameof(typeof(m.prob)), ", ", m.rule,
          ", budget=", m.budget, ")")
Base.show(io::IO, m::FiniteSumNLP) =
    print(io, "FiniteSumNLP(", nameof(typeof(m.prob)), ", M=", population(m.prob),
          ", ", m.rule, ")")

# -----------------------------------------------------------------------------
# Sample management schemes
# -----------------------------------------------------------------------------

"""
    sample_schemes() -> NTuple{3, Symbol}

The three ways a batch may be carried from one iteration to the next.

Within an iteration the batch is fixed and every evaluation shares it; that is the
common random numbers the whole design rests on and no scheme touches it. What the
scheme decides is what happens **between** iterations, and the three answers are
the ones the PreDoc states.

| scheme | there | what it does |
|:--|:--|:--|
| `:independent` | IRV, plotted as VAI | draw `N_k` afresh, independently of the last batch |
| `:nested` | I/CRV, plotted as VAI/VAC | grow by appending new draws, shrink by subsampling the current batch |
| `:prefix` | CRV, plotted as VAC | the first `N_k` of the population, `{ξ_1, …, ξ_{N_k}}` |

# `:independent`

`𝒩_k` and `𝒩_{k+1}` are independent. The estimate `f̃_k` then moves between
iterations for two reasons at once — the iterate moved, and the sample changed —
and early on, when `N_k` is small and the variance is large, the second dominates.

# `:nested`

`𝒩_k ⊆ 𝒩_{k+1}` on a growth: the previous batch is kept and `N_{k+1} − N_k` fresh
realisations are appended. On a shrink the survivors are drawn **uniformly from the
previous batch**, so `𝒩_{k+1} ⊂ 𝒩_k` and nesting holds in that direction too.
Consecutive estimates are then positively correlated and `f̃` moves mostly because
the iterate moved.

Growth without replacement draws from the complement of the current batch, which
costs `O(M)` per growth; with replacement it appends `rand(1:M, ·)` and duplicates
are allowed, as they are in the initial draw.

# `:prefix`

`𝒩_k = {1, …, N_k}`: literally the first `N_k` individuals, in population order.
Every batch is a prefix of every larger one, so this is the most correlated scheme
there is, and it is the only one that needs no random numbers at all after
construction.

It requires a finite population. On an [`ExpectationProblem`](@ref) there is no
first individual to take, and the constructor rejects it.

!!! warning "`:prefix` optimises the wrong problem"
    Under `:prefix` the run minimises `(1/N_k) Σ_{i≤N_k} F(x, ξ_i)`, and while `N_k`
    is small that function has its own minimiser, some distance from the one being
    sought. A run can converge to it, report a small gradient, then grow the batch
    and discover it is far from the solution of the full problem — the sample size
    falls again, and the cycle can repeat.

    This is not a defect of the implementation. It is the overfitting hazard the
    PreDoc describes for CRV, and reproducing it is a reason to have the scheme.
    Compare against `:independent` on the same seed before drawing any conclusion
    about iteration counts.
"""
sample_schemes() = (:independent, :nested, :prefix)

function _check_scheme(scheme::Symbol, finite::Bool)
    scheme in sample_schemes() || throw(ArgumentError(
        "scheme must be one of $(sample_schemes()), got :$scheme"))
    (scheme === :prefix && !finite) && throw(ArgumentError(
        "scheme = :prefix takes the first N_k individuals of the population, and " *
        "an ExpectationProblem has none. Use :independent or :nested, or move to " *
        "a FiniteSumProblem."))
    return nothing
end

"""
    sample_scheme(m::SampledNLP) -> Symbol

Which of [`sample_schemes`](@ref) this oracle is carrying its batch under.
"""
sample_scheme(m::SampledNLP) = m.scheme

"""
    _extend_draw(prob, rng, pool, n_new; replace) -> pool′

Append `n_new` fresh realisations to `pool`, for `:nested`.

Two methods ship: index vectors for a finite sum, and [`GaussianDraw`](@ref) for a
`PerturbedExpectation`. A problem whose `draw_batch` returns anything else has to
add one, and the fallback says so rather than silently drawing an independent batch
and calling it nested.
"""
function _extend_draw(p::FiniteSumProblem, rng, pool::Vector{Int}, n_new::Int;
                      replace::Bool = true)
    n_new <= 0 && return pool
    M = population(p)
    if replace
        return vcat(pool, rand(rng, 1:M, n_new))
    end
    # Without replacement the new draws must miss everything already held, so they
    # come from the complement. `O(M)` per growth, which is the price of nesting.
    seen = falses(M)
    @inbounds for i in pool
        seen[i] = true
    end
    avail = findall(!, seen)
    n_new >= length(avail) && return vcat(pool, avail)
    return vcat(pool, avail[randperm(rng, length(avail))[1:n_new]])
end

function _extend_draw(p::PerturbedExpectation, rng, pool::GaussianDraw, n_new::Int;
                      replace::Bool = true)
    n_new <= 0 && return pool
    fresh = draw_batch(p, rng, n_new)
    C = if pool.C === nothing && fresh.C === nothing
        nothing
    else
        vcat(something(pool.C, Matrix{Float64}[]),
             something(fresh.C, Matrix{Float64}[]))
    end
    return GaussianDraw(hcat(pool.b, fresh.b), C)
end

_extend_draw(p, ::Any, pool, ::Int; replace::Bool = true) = throw(ArgumentError(
    "scheme = :nested needs to append realisations to an existing batch, and " *
    "$(nameof(typeof(p))) draws $(typeof(pool)), for which no `_extend_draw` " *
    "method exists. Add one, or use scheme = :independent."))

"""
    _shrink_draw(rng, pool, n) -> pool′

Keep `n` of the current batch, drawn uniformly from it: the PreDoc's rule for a
decreasing sample under I/CRV, which is what makes `𝒩_{k+1} ⊂ 𝒩_k` hold on a fall
as well as on a rise.
"""
function _shrink_draw(rng, pool::Vector{Int}, n::Int)
    n >= length(pool) && return pool
    return pool[randperm(rng, length(pool))[1:n]]
end

function _shrink_draw(rng, pool::GaussianDraw, n::Int)
    N = length(pool)
    n >= N && return pool
    keep = randperm(rng, N)[1:n]
    return GaussianDraw(pool.b[:, keep], pool.C === nothing ? nothing : pool.C[keep])
end

"Resize the nested pool to `n`, growing or shrinking as the PreDoc's I/CRV asks."
function _resize_pool!(m::SampledNLP, n::Int)
    len = m.pool === nothing ? 0 : length(m.pool)
    if len == 0
        m.pool = _fresh_draw(m, n)
    elseif n > len
        m.pool = _extend_draw(m.prob, m.rng, m.pool, n - len; replace = _replace(m))
    elseif n < len
        m.pool = _shrink_draw(m.rng, m.pool, n)
    end
    return m.pool
end

_fresh_draw(m::ExpectationNLP, n::Int) = draw_batch(m.prob, m.rng, n)
_fresh_draw(m::FiniteSumNLP, n::Int)   = draw_batch(m.prob, m.rng, n; replace = m.replace)
_replace(::ExpectationNLP) = true
_replace(m::FiniteSumNLP)  = m.replace

"Take the first `n` of a pool, without consuming random numbers."
_take(pool::Vector{Int}, n::Int)   = n >= length(pool) ? pool : pool[1:n]
_take(pool::GaussianDraw, n::Int)  = n >= length(pool) ? pool :
    GaussianDraw(pool.b[:, 1:n], pool.C === nothing ? nothing : pool.C[1:n])

"""
    paired_op_variance(m::SampledNLP, x, x_cand) -> Float64

The **outer-product approximation** to `Var(D_i)`, where
`D_i = F(x, ξ_i) − F(x_cand, ξ_i)`, over the objective batch of the current
iteration.

With `s = x_cand − x`, a first-order expansion of `F(·, ξ)` about `x` gives
`D_i ≈ −∇F(x, ξ_i)ᵀ s`, so

```math
\\mathrm{Var}(D_i) \\approx s^\\top \\mathrm{Var}[\\nabla F(x, \\xi)] s
                  \\approx s^\\top \\tilde B s - (\\tilde g^\\top s)^2 ,
```

with `B̃ = (1/N) Σ ∇F_i ∇F_iᵀ` the outer-product matrix and `g̃` the batch gradient,
both over `batch_f` at `x`. The right-hand side is the sample variance of the
scalars `∇F_iᵀs`, and that is how it is computed here: one pass over the score
matrix, forming `Var({sᵀ∇F_i})` directly rather than materialising `B̃`. So the
result is **never negative**, which the difference of the two quadratic forms can
be in floating point when the two nearly cancel.

# What it costs, and what it assumes

One score matrix over `batch_f` at `x`: `O(n·N)` storage and whatever the problem
charges for `scores`. No new realisation is drawn, so like
[`paired_decrease_stats`](@ref) it is not charged to [`samples_used`](@ref) — the
comparison it enables is between two estimates of one quantity on one batch.

Two approximations, and they fail in different places. The Taylor step needs
`‖s‖` small relative to the curvature of `F(·, ξ)` along `s`, so it degrades on a
large step, which is where the radius is large and the batch is small. The outer
product is used here as an estimate of `Var[∇F]` alone: it does **not** need the
information identity, and it is legitimate on a misspecified model where BHHH as a
Hessian is not. Those are separate questions and conflating them is the standard
error.

Returns `NaN` when the batch is too small for a sample variance, matching
[`paired_decrease_stats`](@ref), and when the problem supplies no scores.
"""
function paired_op_variance(m::SampledNLP, x::AbstractVector, x_cand::AbstractVector)
    has_scores(m.prob) || return NaN
    b = m.batch_f
    S = scores(m.prob, x, b)                     # n × N
    N = size(S, 2)
    N < 2 && return NaN
    s = x_cand .- x
    # w_i = ∇F_iᵀ s. Var(D_i) ≈ Var(w_i), and the sample variance of the w_i is
    # exactly sᵀB̃s − (g̃ᵀs)² up to the 1/(N−1) against 1/N convention.
    w = transpose(S) * s
    μ = sum(w) / N
    acc = 0.0
    @inbounds for i in 1:N
        d = w[i] - μ
        acc += d * d
    end
    return acc / (N - 1)
end
