
# ============================================================
# TRRSolver.jl
#
# JSO-compatible trust-region solver struct and solve! interface.
# Implements AbstractOptimizationSolver from SolverCore.jl.
# ============================================================

import SolverCore: solve!

"""
    TRRSolver{T,V} <: AbstractOptimizationSolver

JSO-compatible trust-region solver with pre-allocated workspace.

Pre-allocates vectors for the current iterate, gradient, subproblem
step, and Hessian-vector product.  The solver is reusable across
multiple `solve!` calls — mutable rule state is reset at the start
of each call via `reset_rule!`.

# Fields
- `x`:      current iterate (pre-allocated, length = nvar)
- `g`:      current gradient (pre-allocated)
- `p`:      subproblem step (pre-allocated)
- `Hp`:     Hessian-vector product (pre-allocated)
- `rule`:   deep-copied radius update rule (state preserved between resets)
- `params`: solver hyper-parameters (`TRSolverParams`)
"""
mutable struct TRRSolver{T, V} <: AbstractOptimizationSolver
    x     ::V
    g     ::V
    p     ::V
    Hp    ::V
    rule  ::AbstractRadiusUpdate
    params::TRSolverParams
end

"""
    TRRSolver(nlp, rule, params) -> TRRSolver

Construct a `TRRSolver` for the problem `nlp`.  Pre-allocates workspace
vectors and deep-copies `rule` so that mutable rule state is owned by
the solver and can be reset independently of the caller's copy.
"""
function TRRSolver(nlp::AbstractNLPModel{T, V},
                   rule::AbstractRadiusUpdate,
                   params::TRSolverParams) where {T, V}
    n  = nlp.meta.nvar
    x  = similar(nlp.meta.x0)
    g  = similar(nlp.meta.x0)
    p  = similar(nlp.meta.x0)
    Hp = similar(nlp.meta.x0)
    TRRSolver{T, V}(x, g, p, Hp, deepcopy(rule), params)
end

# ------------------------------------------------------------
# Primary solve! — updates stats in place
# ------------------------------------------------------------

"""
    solve!(solver, nlp, stats; callback) -> GenericExecutionStats

In-place JSO solve.  The `stats` object is updated throughout the run.

`callback(nlp, solver, stats)` is called after each accepted/rejected
step (once per iteration).  The default callback is a no-op.

# Status symbols
- `:first_order`  — gradient norm ≤ tol
- `:max_iter`     — maximum iterations reached without convergence
- `:exception`    — unexpected error during iteration

# Notes
- `stats.solution` points to `solver.x` (zero-copy).
- NLPModels evaluation counters (`neval_obj`, `neval_grad`, etc.) are
  read directly from `nlp` at termination.
"""
function SolverCore.solve!(solver::TRRSolver{T, V},
                            nlp::AbstractNLPModel{T, V},
                            stats::GenericExecutionStats{T, V};
                            callback = (args...) -> nothing) where {T, V}

    params = solver.params

    # Reset mutable rule state (idempotent for immutable rules)
    reset_rule!(solver.rule)

    t_start = time()

    # ----------------------------------------------------------
    # Initialisation
    # ----------------------------------------------------------
    copyto!(solver.x, nlp.meta.x0)
    grad!(nlp, solver.x, solver.g)
    f       = obj(nlp, solver.x)
    g_norm  = norm(solver.g)
    Δ       = initial_radius(solver.rule, params.Δ₀, g_norm)

    set_iter!(stats, 0)
    set_objective!(stats, f)
    set_dual_residual!(stats, g_norm)

    # ----------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------
    k = 0
    for outer_k in 1:params.max_iterations
        k = outer_k

        # Convergence check
        if g_norm <= params.tol
            set_status!(stats, :first_order)
            set_solution!(stats, solver.x)
            set_objective!(stats, f)
            set_dual_residual!(stats, g_norm)
            set_iter!(stats, k - 1)
            set_time!(stats, time() - t_start)
            return stats
        end

        # Solve trust-region subproblem via truncated CG Steihaug
        # truncated_cg_steihaug allocates its result; copy into pre-allocated buffer
        p_new, _, _ = truncated_cg_steihaug(nlp, solver.x, solver.g, Δ)
        copyto!(solver.p, p_new)
        s_norm = norm(solver.p)

        # Candidate evaluation
        x_cand = solver.x + solver.p
        f_cand = obj(nlp, x_cand)

        # Predicted reduction  m(0) - m(p) = -gᵀp - ½pᵀHs
        hprod!(nlp, solver.x, solver.p, solver.Hp)
        predicted = -dot(solver.g, solver.p) - T(0.5) * dot(solver.p, solver.Hp)

        # Actual reduction
        actual = f - f_cand

        # Ratio ρ (guard against zero or negative predicted reduction)
        ρ = predicted > 0 ? actual / predicted : -Inf

        # Store gradient norm before acceptance (needed by R3)
        g_norm_old = g_norm

        # Accept / reject
        if ρ >= params.η₁
            copyto!(solver.x, x_cand)
            f = f_cand
            grad!(nlp, solver.x, solver.g)
            g_norm = norm(solver.g)
        end

        g_norm_new = g_norm

        # Update radius
        Δ = update_radius!(solver.rule, Δ, ρ,
                           params.η₁, params.η₂,
                           s_norm, g_norm_old, g_norm_new)

        # Update stats for callback
        set_iter!(stats, k)
        set_objective!(stats, f)
        set_dual_residual!(stats, g_norm_new)
        set_time!(stats, time() - t_start)

        # User callback
        callback(nlp, solver, stats)
    end

    # ----------------------------------------------------------
    # Maximum iterations reached
    # ----------------------------------------------------------
    @info "TRRSolver: maximum iterations ($(params.max_iterations)) reached without convergence."
    set_status!(stats, :max_iter)
    set_solution!(stats, solver.x)
    set_objective!(stats, f)
    set_dual_residual!(stats, g_norm)
    set_iter!(stats, k)
    set_time!(stats, time() - t_start)
    return stats
end

# ------------------------------------------------------------
# Convenience overload — creates stats automatically
# ------------------------------------------------------------

"""
    solve!(solver, nlp; kwargs...) -> GenericExecutionStats

Convenience method that creates a `GenericExecutionStats` and calls
the primary `solve!(solver, nlp, stats; kwargs...)`.
"""
function SolverCore.solve!(solver::TRRSolver{T, V},
                            nlp::AbstractNLPModel{T, V};
                            kwargs...) where {T, V}
    stats = GenericExecutionStats(nlp)
    solve!(solver, nlp, stats; kwargs...)
    return stats
end

# ------------------------------------------------------------
# Functional JSO interface
# ------------------------------------------------------------

"""
    trust_region_radius(nlp; rule, params, kwargs...) -> GenericExecutionStats

Functional JSO-style interface.  Creates a `TRRSolver`, solves the
problem `nlp`, and returns a `GenericExecutionStats`.

# Keyword arguments
- `rule::AbstractRadiusUpdate`   — radius update rule (default: `R1ClassicalUpdate()`)
- `params::TRSolverParams`       — solver parameters (default: `TRSolverParams()`)
- remaining kwargs forwarded to `solve!` (e.g. `callback=...`)

# Example
```julia
using ADNLPModels, TrustRegionRadius
nlp   = ADNLPModel(x -> sum(x.^2), ones(10))
stats = trust_region_radius(nlp; rule=R1ClassicalUpdate(), params=TRSolverParams(tol=1e-8))
println(stats.status)
```
"""
function trust_region_radius(nlp::AbstractNLPModel{T, V};
                              rule::AbstractRadiusUpdate   = R1ClassicalUpdate(),
                              params::TRSolverParams       = TRSolverParams(),
                              kwargs...) where {T, V}
    solver = TRRSolver(nlp, rule, params)
    return solve!(solver, nlp; kwargs...)
end
