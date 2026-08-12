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
end

function ExpectationNLP(prob::ExpectationProblem, rule::SamplingRule;
                        x0::AbstractVector = zeros(prob.n), seed::Int = 0,
                        N_init::Int = 8, budget::Int = 1_000_000)
    check_rule_problem(rule, prob)
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
        zeros(0, 0), Float64[], false)
end

function FiniteSumNLP(prob::FiniteSumProblem, rule::SamplingRule;
                      x0::AbstractVector = zeros(prob.n), seed::Int = 0,
                      replace::Bool = true, N_init::Int = 8,
                      budget::Union{Int, Nothing} = nothing)
    check_rule_problem(rule, prob)
    check_population_cap(rule, prob)
    M = population(prob)
    cap = budget === nothing ? M : min(budget, M)
    cap > 0 || throw(ArgumentError("FiniteSumNLP: need budget > 0"))
    meta = NLPModelMeta(prob.n; x0 = Vector{Float64}(x0), name = "FiniteSumNLP")
    rng = MersenneTwister(seed)
    N0 = min(N_init, cap)
    b = draw_batch(prob, rng, N0; replace = replace)
    return FiniteSumNLP{typeof(prob), typeof(rule)}(
        meta, Counters(), prob, rule, rng, seed, replace, N0, copy(b), copy(b),
        N0, N0, 0, 0, 0, Int[], Int[], Bool[], 1.0, 1.0, 0.0, 0.0, NaN, 0, cap,
        zeros(0, 0), Float64[], false)
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

function _draw!(m::ExpectationNLP, Ng::Int, Nf::Int)
    m.batch_g = draw_batch(m.prob, m.rng, Ng)
    m.batch_f = Ng == Nf ? m.batch_g : draw_batch(m.prob, m.rng, Nf)
    return nothing
end

function _draw!(m::FiniteSumNLP, Ng::Int, Nf::Int)
    m.batch_g = draw_batch(m.prob, m.rng, Ng; replace = m.replace)
    # Objective and gradient share the batch when the sizes agree, which is the
    # common case and gives the tightest common random numbers.
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
    b = draw_batch(m.prob, m.rng, N0); m.batch_g = b; m.batch_f = b; return nothing
end
function _reset_batches!(m::FiniteSumNLP, N0::Int)
    b = draw_batch(m.prob, m.rng, N0; replace = m.replace)
    m.batch_g = copy(b); m.batch_f = copy(b); return nothing
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
