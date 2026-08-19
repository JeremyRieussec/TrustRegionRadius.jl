# =============================================================================
# src/Trust-region/entry.jl
#
# One name, three solvers. `tr_solve` dispatches on the problem class, so the
# regime is decided by the oracle rather than by a keyword the caller might forget.
# =============================================================================

"""
    tr_solve(nlp; rule, model, subsolver, params, x_ref, true_curvature,
             hessian_norm, kwargs...) -> TRResult

The declared keywords are `rule`, `model`, `subsolver`, `params`, `x_ref`,
`true_curvature` and `hessian_norm`. Everything else, `trace` and `callback` above
all, is forwarded through `kwargs...` to `SolverCore.solve!`, so `trace = true` is
the supported spelling rather than a keyword of this function.

Solve `nlp`, choosing the solver from the problem class:

| `nlp` | solver |
|:--|:--|
| any `AbstractNLPModel`, [`FullBatchNLP`](@ref) | [`DeterministicTRSolver`](@ref) |
| [`ExpectationNLP`](@ref) | [`ExpectationTRSolver`](@ref) |
| [`FiniteSumNLP`](@ref) | [`FiniteSumTRSolver`](@ref) |

The sampling rule is not an argument here: it belongs to the oracle, because it
decides what the oracle returns.

```julia
using TrustRegionRadius, ADNLPModels

# deterministic
nlp = ADNLPModel(x -> (1 - x[1])^2 + 100(x[2] - x[1]^2)^2, [-1.2, 1.0])
tr_solve(nlp; rule = RGrad(), model = SR1Model(mem = 5))

# finite sum: the cap is M, not a keyword
prob = LogisticRegression(K = 5, M = 2_000)
tr_solve(FiniteSumNLP(prob, RadiusProportional()); model = BHHHModel(ridge = 1e-8))

# expectation: the cap is the budget, because there is no M
exp_prob = PerturbedExpectation(base; σg = 1.0)
tr_solve(ExpectationNLP(exp_prob, RadiusProportional(); budget = 50_000))
```
"""
function tr_solve(nlp::AbstractNLPModel{T, V};
                  rule::RadiusRule            = RDelta(),
                  model::ModelHessian         = ExactHessian(),
                  subsolver::SubproblemSolver = SteihaugCG(),
                  params::TRParams            = TRParams{T}(),
                  x_ref::Union{Nothing, AbstractVector} = nothing,
                  true_curvature::Bool        = false,
                  hessian_norm::Bool          = false,
                  kwargs...) where {T, V}
    solver = DeterministicTRSolver(nlp; rule = rule, model = model,
                                   subsolver = subsolver, params = params,
                                   x_ref = x_ref, true_curvature = true_curvature,
                                   hessian_norm = hessian_norm)
    return solve!(solver, nlp; kwargs...)
end

function tr_solve(nlp::ExpectationNLP;
                  rule::RadiusRule            = RDelta(),
                  model::ModelHessian         = ExactHessian(),
                  subsolver::SubproblemSolver = SteihaugCG(),
                  params::TRParams            = TRParams{Float64}(),
                  x_ref::Union{Nothing, AbstractVector} = nothing,
                  true_curvature::Bool        = false,
                  hessian_norm::Bool          = false,
                  kwargs...)
    solver = ExpectationTRSolver(nlp; rule = rule, model = model,
                                 subsolver = subsolver, params = params,
                                 x_ref = x_ref, true_curvature = true_curvature,
                                   hessian_norm = hessian_norm)
    return solve!(solver, nlp; kwargs...)
end

function tr_solve(nlp::FiniteSumNLP;
                  rule::RadiusRule            = RDelta(),
                  model::ModelHessian         = ExactHessian(),
                  subsolver::SubproblemSolver = SteihaugCG(),
                  params::TRParams            = TRParams{Float64}(),
                  x_ref::Union{Nothing, AbstractVector} = nothing,
                  true_curvature::Bool        = false,
                  hessian_norm::Bool          = false,
                  kwargs...)
    solver = FiniteSumTRSolver(nlp; rule = rule, model = model,
                               subsolver = subsolver, params = params,
                               x_ref = x_ref, true_curvature = true_curvature,
                                   hessian_norm = hessian_norm)
    return solve!(solver, nlp; kwargs...)
end

function SolverCore.solve!(solver::AbstractTRSolver, nlp::AbstractNLPModel; kwargs...)
    stats = GenericExecutionStats(nlp)
    return solve!(solver, nlp, stats; kwargs...)
end

"""
    model_grad_evals(solver) -> Int

Gradient evaluations made from inside the *model*, which the algorithm did not
request. Non-zero only for [`SPDTarget`](@ref). Subtract before reporting
`neval_grad` as a cost.
"""
model_grad_evals(::AbstractTRSolver) = 0
