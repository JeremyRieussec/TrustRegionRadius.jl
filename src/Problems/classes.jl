# src/Problems/classes.jl
#
# The problem-class hierarchy.
#
#     AbstractProblem
#     ├── DeterministicProblem      f(x) evaluated exactly
#     └── SampledProblem            f(x) estimated from a sample
#         ├── ExpectationProblem    f(x) = E_ξ[F(x,ξ)];  population unbounded
#         └── FiniteSumProblem      f(x) = (1/M) Σ_{i=1}^M f_i(x);  M finite, known
#             └── ScoredProblem     per-observation gradients available
#                 └── LikelihoodProblem   f = −(1/M) Σ ln p_i;  BHHH justified here
#                     └── NLSProblem      f = (1/2M) Σ r_i²;  Gauss-Newton too
#
# ---------------------------------------------------------------------------
# Why the distinction is worth a type and not a flag
#
# EXPECTATION and FINITE SUM are not the same problem with different M.
#
#   * The population cap. An expectation has no largest sample: N_k may grow
#     without bound and `N_max` is a *budget the user chooses*. A finite sum has
#     exactly M terms, so N_k ≤ M is imposed by the problem, and a user-supplied
#     `N_max` is either redundant or a deliberate sub-population budget — two
#     different intentions that should not share a keyword. The finite-sum solver
#     therefore rejects a rule carrying a user `N_max` and takes `budget`
#     explicitly.
#
#   * The full-batch limit. At N_k = M a finite-sum iteration is *exactly*
#     deterministic: ρ̂ = ρ, the stopping test is the real one, and no accuracy
#     hypothesis is needed. An expectation has no such limit, so its convergence
#     theory can never be discharged by sampling harder. `FullBatch()` makes the
#     limit reachable and `:full_batch_trajectory` records when it was reached.
#
#   * The truth. A finite sum can always be evaluated exactly, at the cost of one
#     pass, so `true_gradient` is available on every problem in the class and the
#     stopping test can be confirmed. For an expectation it exists only when the
#     construction supplies it (`PerturbedExpectation` does; a general one does
#     not), so `has_truth` is a trait rather than an assumption.
#
# DETERMINISTIC is not "finite sum with M = 1". It has no batch, no sampling
# rule, no variance estimate and no resampling hook, and its solver should not
# carry the branches for them.
# =============================================================================

"""
    AbstractProblem

Root of the problem-class hierarchy. A concrete problem declares its regime by
subtyping [`DeterministicProblem`](@ref), [`ExpectationProblem`](@ref) or
[`FiniteSumProblem`](@ref), and the solver, the model Hessian and the sampling
rule are all checked against that declaration at construction.

Ordinary JSO models (`ADNLPModel`, `CUTEstModel`, …) are *not* subtypes: they are
deterministic by default and need no wrapper. [`underlying_problem`](@ref) returns
`nothing` for them, which is what the compatibility checks test.
"""
abstract type AbstractProblem end

"""
    DeterministicProblem <: AbstractProblem

`f(x)` is evaluated exactly. No batch, no sampling rule, no variance estimate.

Solved by [`DeterministicTRSolver`](@ref). Most users never subtype this — a plain
`AbstractNLPModel` is treated as deterministic. It exists for the *full-batch view
of a sampled problem*, [`FullBatchNLP`](@ref), which is deterministic while still
carrying the finite-sum problem underneath so that `BHHHModel` remains legal.
"""
abstract type DeterministicProblem <: AbstractProblem end

"""
    SampledProblem <: AbstractProblem

`f(x)` is estimated from a sample drawn afresh at each iteration. The common
supertype of [`ExpectationProblem`](@ref) and [`FiniteSumProblem`](@ref); dispatch
on it for anything that applies to both, on the two children for anything that
does not.
"""
abstract type SampledProblem <: AbstractProblem end

"""
    ExpectationProblem <: SampledProblem

`f(x) = E_ξ[F(x,ξ)]` with an **unbounded** population: realisations are drawn
i.i.d. from a distribution, never from a finite list.

Consequences the type carries:

- `population(p) == typemax(Int)`; `N_k` is capped only by the user's budget.
- There is no full-batch iteration and no exact evaluation, so the accuracy
  hypotheses of the stochastic convergence theory can never be discharged.
- `has_truth(p)` is `false` unless the construction happens to supply the exact
  mean — [`PerturbedExpectation`](@ref) does, a general one does not.

Solved by [`ExpectationTRSolver`](@ref).
"""
abstract type ExpectationProblem <: SampledProblem end

"""
    FiniteSumProblem <: SampledProblem

`f(x) = (1/M) Σ_{i=1}^{M} f_i(x)` with `M` finite and known: sampling means
subsetting `1:M`.

Consequences the type carries:

- `population(p) == M`, so `N_k ≤ M` is imposed by the problem. A sampling rule
  used here must **not** carry a user-supplied `N_max`; pass `budget` to
  [`FiniteSumTRSolver`](@ref) if a sub-population cap is genuinely wanted.
- `N_k = M` is an exact iteration. [`FullBatch`](@ref) reaches it deliberately, and
  `:full_batch_trajectory` records when it happened.
- [`true_objective`](@ref) and [`true_gradient`](@ref) are always available, at one
  full pass, so `has_truth` is `true` for the whole class.

Solved by [`FiniteSumTRSolver`](@ref).
"""
abstract type FiniteSumProblem <: SampledProblem end

"""
    ScoredProblem <: FiniteSumProblem

A finite-sum problem that reports the gradient of **each term separately**:

    scores(p, x, batch)     -> n × |batch| Matrix, column i = ∇f_i(x)
    loss_terms(p, x, batch) -> |batch| Vector, entry i = f_i(x)

The column mean is the gradient, so this is strictly more information than a
gradient evaluation and usually costs no more to produce. It is what the
inner-product sampling rules and the outer-product model Hessians consume.

Scores sit under `FiniteSumProblem` because every scored construction here is a
dataset. An expectation with per-realisation scores is coherent and would need a
parallel branch; `has_scores` is the trait to dispatch on if you add one.
"""
abstract type ScoredProblem <: FiniteSumProblem end

"""
    LikelihoodProblem <: ScoredProblem

`f(x) = −(1/M) Σ ln p_i(x)`, an average negative log-likelihood.

**This is the class in which BHHH is justified**, and the reason it is a type
rather than a docstring: `required_problem(::BHHHModel) === LikelihoodProblem`, so
`BHHHModel` over anything else is an `ArgumentError` at solver construction rather
than a silently meaningless matrix.

The justification is the information identity `V = −H`, which holds for a
*correctly specified* model at the *true parameters*. Belonging to this type
asserts the first half only — that `f` is a negative log-likelihood and the scores
are its per-observation gradients. Whether the model is correctly specified is a
statement about the data that no type can carry, which is what
[`information_identity_error`](@ref) is for.
"""
abstract type LikelihoodProblem <: ScoredProblem end

"""
    NLSProblem <: LikelihoodProblem

`f(x) = (1/2M) Σ r_i(x)²`, nonlinear least squares, additionally supplying

    residuals(p, x, batch) -> Vector
    jacobian(p, x, batch)  -> |batch| × n Matrix

Placed under [`LikelihoodProblem`](@ref) because it *is* one: least squares is
maximum likelihood under i.i.d. Gaussian errors of known variance, and the score
`r_i ∇r_i` is the log-likelihood score up to `σ²`. So BHHH applies here as well as
Gauss-Newton, and the two can be compared on one problem — which is worth having,
since they discard different terms and fail for different reasons.

`required_problem(::GaussNewtonModel) === NLSProblem`: the Jacobian is what it
needs and nothing above this type has one.
"""
abstract type NLSProblem <: LikelihoodProblem end

# -----------------------------------------------------------------------------
# Traits
# -----------------------------------------------------------------------------

"""
    problem_class(p) -> Symbol

`:deterministic`, `:expectation` or `:finite_sum`. Used in error messages and in
the benchmark tables, where a run's regime must be reported alongside its cost.
"""
problem_class(::DeterministicProblem) = :deterministic
problem_class(::ExpectationProblem)   = :expectation
problem_class(::FiniteSumProblem)     = :finite_sum
problem_class(::AbstractNLPModel)     = :deterministic

"""
    population(p) -> Int

The number of terms available. `M` for a finite sum, `typemax(Int)` for an
expectation — not an error, because "unbounded" is exactly what the sampling layer
needs to know: it is the value `N_k` is capped against, and capping against
`typemax(Int)` is a no-op.
"""
population(::ExpectationProblem) = typemax(Int)
function population end

"""
    n_terms(p) -> Int

Alias of [`population`](@ref), kept because it appears in the default argument of
[`information_identity_error`](@ref) and [`gauss_newton_error`](@ref).
"""
n_terms(p::AbstractProblem) = population(p)

"""
    has_scores(p) -> Bool

Whether per-observation gradients are available. `true` for the whole of
[`ScoredProblem`](@ref) and its descendants.

Checked once, at oracle construction, against [`needs_scores`](@ref) on the
sampling rule — the inner-product family needs `Var(∇F_iᵀĝ)` and cannot estimate it
from a batch gradient alone.
"""
has_scores(::AbstractProblem) = false
has_scores(::ScoredProblem)   = true

"""
    has_truth(p) -> Bool

Whether `f` and `∇f` can be evaluated exactly, so a run can be scored on the truth
rather than on its own estimates.

`true` for every [`FiniteSumProblem`](@ref) — one full pass. For an
[`ExpectationProblem`](@ref) it is `false` unless the construction supplies the
exact mean, which [`PerturbedExpectation`](@ref) does by design.

This gates `TRParams(true_stop = true)` and the `:true_grad_trajectory` entry:
requesting either on a problem without truth is an `ArgumentError`, not a silent
fallback to the batch estimate.
"""
has_truth(::AbstractProblem)    = false
has_truth(::FiniteSumProblem)   = true
has_truth(::ExpectationProblem) = false

"""
    underlying_problem(nlp) -> AbstractProblem or nothing

The [`AbstractProblem`](@ref) an NLP oracle presents, or `nothing` for a plain JSO
model. This is what the model-Hessian compatibility check dispatches on: a raw
`ADNLPModel` returns `nothing`, so a model requiring `LikelihoodProblem` is
rejected against it.
"""
underlying_problem(::AbstractNLPModel) = nothing

# -----------------------------------------------------------------------------
# Compatibility checks
# -----------------------------------------------------------------------------

"""
    required_problem(model) -> Type

The narrowest problem class a [`ModelHessian`](@ref) is defined for.
`AbstractProblem` — "anything, including a plain NLP" — by default.

The three outer-product models narrow it, which is the point of the hierarchy:

| model | requires | because |
|:--|:--|:--|
| `BHHHModel`, `BHHH2Model` | `LikelihoodProblem` | the information identity is a statement about a likelihood |
| `GaussNewtonModel` | `NLSProblem` | it needs the residual Jacobian |

Checked by [`check_model_problem`](@ref) at solver construction, so the failure is
a constructor error naming both types rather than a matrix that means nothing.
"""
required_problem(::Any) = AbstractProblem

"""
    check_model_problem(model, nlp) -> nothing

Throw an `ArgumentError` if `model` requires a narrower problem class than `nlp`
provides. Called by all three solver constructors.
"""
function check_model_problem(model, nlp)
    R = required_problem(model)
    R === AbstractProblem && return nothing
    prob = underlying_problem(nlp)
    if prob === nothing
        throw(ArgumentError(
            "$(nameof(typeof(model))) is defined only for $(nameof(R)), but the " *
            "problem is a plain $(nameof(typeof(nlp))) with no problem class. " *
            "Wrap a $(nameof(R)) in FullBatchNLP, FiniteSumNLP or ExpectationNLP, " *
            "or use ExactHessian or a quasi-Newton model."))
    end
    prob isa R || throw(ArgumentError(
        "$(nameof(typeof(model))) is defined only for $(nameof(R)), but " *
        "$(nameof(typeof(prob))) is a $(problem_class(prob)) problem." *
        (R === LikelihoodProblem ?
         " BHHH approximates ∇²f through the information identity, which is a " *
         "statement about a negative log-likelihood; over anything else the matrix " *
         "is a positive semidefinite preconditioner with no approximation " *
         "guarantee behind it." :
         R === NLSProblem ?
         " Gauss-Newton needs the residual Jacobian, which only an NLSProblem has." :
         "")))
    return nothing
end

"""
    check_rule_problem(rule, prob) -> nothing

Throw an `ArgumentError` if a sampling rule needs per-observation scores and the
problem does not supply them. Called by the two sampled oracle constructors.
"""
function check_rule_problem(rule, prob::AbstractProblem)
    needs_scores(rule) && !has_scores(prob) && throw(ArgumentError(
        "$(nameof(typeof(rule))) needs the per-observation score matrix to " *
        "estimate Var(∇Fᵢᵀĝ), but $(nameof(typeof(prob))) does not supply one. " *
        "Use NormTest or RadiusProportional, or supply a ScoredProblem."))
    return nothing
end

"""
    user_cap(rule) -> Int or nothing

The `N_max` the user supplied, or `nothing` when they supplied none.

Distinct from [`sample_cap`](@ref), which resolves to `typemax(Int)` so it can be
used in arithmetic. This one preserves the difference between "unset" and "set to
a large number", which is what [`check_population_cap`](@ref) needs.
"""
user_cap(::Any) = nothing

"""
    check_population_cap(rule, prob) -> nothing

Throw an `ArgumentError` if a rule used on a [`FiniteSumProblem`](@ref) carries a
**user-supplied** `N_max`.

For a finite sum the cap is `M`, imposed by the problem. A user `N_max` there is
either redundant (`≥ M`) or a deliberate sub-population budget (`< M`), and those
are different intentions that should not share a keyword — the first hides a
misunderstanding, the second is a legitimate experiment. Pass `budget` to
[`FiniteSumTRSolver`](@ref) for the second.

On an [`ExpectationProblem`](@ref) nothing is checked: there the population is
unbounded and the cap *is* the user's budget.
"""
function check_population_cap(rule, prob::FiniteSumProblem)
    cap = user_cap(rule)
    cap === nothing && return nothing
    throw(ArgumentError(
        "$(nameof(typeof(rule))) was given N_max = $cap, but on a finite-sum " *
        "problem the cap is imposed by the problem: N_k ≤ M = $(population(prob)). " *
        "Drop N_max. If a sub-population budget is intended, pass it to the solver " *
        "as `budget = $cap`, which records it as an experimental choice rather " *
        "than a property of the rule."))
end
check_population_cap(rule, ::ExpectationProblem) = nothing
