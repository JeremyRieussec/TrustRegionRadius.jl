
# ============================================================
# TRRSolver.jl
#
# JSO-compatible trust-region solver struct and solve! interface.
# Implements AbstractOptimizationSolver from SolverCore.jl.
#
# Parametric on element type T, vector type V, radius update rule R,
# and subproblem solver S so that every dispatch is resolved at
# compile time (no dynamic dispatch in the hot loop).
# ============================================================

# ------------------------------------------------------------
# TRSolverParams -- parametric on the element type
# ------------------------------------------------------------

"""
    TRSolverParams{T}

Solver-level parameters for `TRRSolver`.  Parametric on the floating-point
type `T` so that `η₁`, `η₂`, `Δ₀`, `tol` remain in the same precision as
the NLPModel's element type (Float32, Float64, BigFloat).

# Fields
- `η₁::T`:                 lower acceptance threshold (reject if ρ < η₁); default 0.1
- `η₂::T`:                 upper threshold for "very successful" step;    default 0.9
- `Δ₀::T`:                 initial trust-region radius;                  default 1
- `max_iterations::Int`:   iteration budget;                             default 10_000
- `tol::T`:                gradient-norm convergence tolerance;          default √eps(T)

# Constructor
    TRSolverParams{T}(; η₁, η₂, Δ₀, max_iterations, tol)
    TRSolverParams(; kwargs...)   # defaults to T = Float64
"""
struct TRSolverParams{T}
    η₁::T
    η₂::T
    Δ₀::T
    max_iterations::Int
    tol::T

    function TRSolverParams{T}(; η₁::T         = T(0.1),
                                 η₂::T         = T(0.9),
                                 Δ₀::T         = T(1),
                                 max_iterations::Int = 10_000,
                                 tol::T        = sqrt(eps(T))) where {T}
        @assert 0 <= η₁ < η₂ < 1 "Need 0 ≤ η₁ < η₂ < 1"
        @assert Δ₀ > 0           "Need Δ₀ > 0"
        @assert max_iterations > 0
        @assert tol > 0
        new{T}(η₁, η₂, Δ₀, max_iterations, tol)
    end
end

TRSolverParams(; kwargs...) = TRSolverParams{Float64}(; kwargs...)

function Base.show(io::IO, p::TRSolverParams{T}) where {T}
    println(io, "TRSolverParams{$T}:")
    println(io, "  η₁: ", p.η₁, "  η₂: ", p.η₂)
    println(io, "  Δ₀: ", p.Δ₀)
    println(io, "  max_iterations: ", p.max_iterations)
    println(io, "  tol: ", p.tol)
end

# ------------------------------------------------------------
# TRRSolver struct
# ------------------------------------------------------------

"""
    TRRSolver{T, V, R, S} <: AbstractOptimizationSolver

JSO-compatible trust-region solver with pre-allocated workspace.

Parametric on:
- `T`: element type (typically Float32, Float64, BigFloat)
- `V`: vector type  (typically Vector{T} or CuArray{T})
- `R`: radius update rule type (`<: AbstractRadiusUpdate`)
- `S`: subproblem solver type  (`<: AbstractTRSubproblemSolver`)

The type parameters ensure that `update_radius!` and
`solve_subproblem!` are resolved by static dispatch.

# Fields
- `x`:         current iterate
- `g`:         current gradient
- `p`:         subproblem step (filled in place by the subsolver)
- `x_cand`:    candidate iterate x + p
- `Hp`:        Hessian-vector product H(x)·p
- `rule`:      deep-copied radius update rule
- `subsolver`: deep-copied subproblem solver
- `params`:    solver hyper-parameters (`TRSolverParams{T}`)
"""
mutable struct TRRSolver{T, V <: AbstractVector{T},
                         R <: AbstractRadiusUpdate,
                         S <: AbstractTRSubproblemSolver} <: AbstractOptimizationSolver
    x         ::V
    g         ::V
    p         ::V
    x_cand    ::V
    Hp        ::V
    rule      ::R
    subsolver ::S
    params    ::TRSolverParams{T}
end

"""
    TRRSolver(nlp; rule, subsolver, params)

Construct a `TRRSolver` for the problem `nlp`.  Pre-allocates all
workspace vectors.  The `rule` and `subsolver` are deep-copied so the
solver owns its mutable state and can be reset independently of the
caller's instance.

# Keyword arguments
- `rule::AbstractRadiusUpdate`      : default `R1ClassicalUpdate()`
- `subsolver::AbstractTRSubproblemSolver` : default `SteihaugTointCG()`
- `params::TRSolverParams{T}`       : default `TRSolverParams{T}()`
"""
function TRRSolver(nlp::AbstractNLPModel{T, V};
                   rule::AbstractRadiusUpdate              = R1ClassicalUpdate(),
                   subsolver::AbstractTRSubproblemSolver   = SteihaugTointCG(),
                   params::TRSolverParams                  = TRSolverParams{T}()) where {T, V}
    x      = similar(nlp.meta.x0)
    g      = similar(nlp.meta.x0)
    p      = similar(nlp.meta.x0)
    x_cand = similar(nlp.meta.x0)
    Hp     = similar(nlp.meta.x0)
    rule_c = deepcopy(rule)
    subc   = deepcopy(subsolver)
    return TRRSolver{T, V, typeof(rule_c), typeof(subc)}(
        x, g, p, x_cand, Hp, rule_c, subc, params
    )
end

# ------------------------------------------------------------
# SolverCore.reset! -- called by SolverBenchmark between problems
# ------------------------------------------------------------

"""
    SolverCore.reset!(solver::TRRSolver)

Restore any mutable state in the radius update rule to its construction
value.  Called automatically by `SolverBenchmark` between problems.
Returns the solver for chaining.
"""
function SolverCore.reset!(solver::TRRSolver)
    reset_rule!(solver.rule)
    return solver
end

function SolverCore.reset!(solver::TRRSolver, ::AbstractNLPModel)
    return SolverCore.reset!(solver)
end

# ------------------------------------------------------------
# Primary solve! -- updates stats in place
# ------------------------------------------------------------

"""
    solve!(solver, nlp, stats; callback) -> GenericExecutionStats

In-place JSO solve.  `stats` is mutated throughout.

`callback(nlp, solver, stats)` is called at the end of every iteration.
Set `stats.status = :user` inside the callback to stop the algorithm.

# Status symbols
- `:first_order` : gradient norm ≤ tol
- `:max_iter`    : maximum iterations reached without convergence
- `:user`        : user-requested stop via callback

# Notes
- NLPModels counters are read directly from `nlp` via `neval_*` at termination.
- The loop writes into `solver.p`, `solver.x_cand`, `solver.Hp` without
  allocating new vectors.
"""
function SolverCore.solve!(solver::TRRSolver{T, V, R, S},
                            nlp::AbstractNLPModel{T, V},
                            stats::GenericExecutionStats{T, V};
                            callback = (args...) -> nothing) where {T, V, R, S}

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
    set_solution!(stats, solver.x)
    set_status!(stats, :unknown)

    # ----------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------
    k = 0
    done = false
    while !done
        k += 1

        # Convergence check BEFORE computing anything for this iteration
        if g_norm <= params.tol
            set_status!(stats, :first_order)
            break
        end

        if k > params.max_iterations
            set_status!(stats, :max_iter)
            k -= 1
            break
        end

        # Solve trust-region subproblem in place: fills solver.p
        on_boundary = solve_subproblem!(solver.subsolver,
                                         nlp, solver.x, solver.g, Δ,
                                         solver.p, solver.Hp)
        s_norm = norm(solver.p)

        # Candidate iterate: x_cand = x + p  (in place, no allocation)
        @. solver.x_cand = solver.x + solver.p
        f_cand = obj(nlp, solver.x_cand)

        # Predicted reduction  m(0) - m(p) = -gᵀp - ½pᵀHp
        hprod!(nlp, solver.x, solver.p, solver.Hp)
        predicted = -dot(solver.g, solver.p) - T(0.5) * dot(solver.p, solver.Hp)

        # Actual reduction and ratio ρ
        actual = f - f_cand
        ρ = predicted > 0 ? actual / predicted : T(-Inf)

        # Store g-norm BEFORE acceptance (needed by R3)
        g_norm_old = g_norm

        # Accept / reject
        if ρ >= params.η₁
            copyto!(solver.x, solver.x_cand)
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
        set_solution!(stats, solver.x)
        set_time!(stats, time() - t_start)

        # User callback (may set stats.status = :user)
        callback(nlp, solver, stats)
        if stats.status == :user
            done = true
        end
    end

    # ----------------------------------------------------------
    # Finalise
    # ----------------------------------------------------------
    set_solution!(stats, solver.x)
    set_objective!(stats, f)
    set_dual_residual!(stats, g_norm)
    set_iter!(stats, k)
    set_time!(stats, time() - t_start)

    return stats
end

# ------------------------------------------------------------
# Convenience overload -- creates stats automatically
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
    trust_region_radius(nlp; rule, subsolver, params, kwargs...) -> GenericExecutionStats

Functional JSO-style interface.  Constructs a `TRRSolver`, solves the
problem `nlp`, and returns a `GenericExecutionStats`.

# Keyword arguments
- `rule::AbstractRadiusUpdate`          : default `R1ClassicalUpdate()`
- `subsolver::AbstractTRSubproblemSolver` : default `SteihaugTointCG()`
- `params::TRSolverParams`              : default `TRSolverParams{T}()`
- remaining kwargs forwarded to `solve!` (e.g. `callback=...`)

# Example
```julia
using ADNLPModels, TrustRegionRadius
nlp   = ADNLPModel(x -> sum(x.^2), ones(10))
stats = trust_region_radius(nlp;
    rule      = R1ClassicalUpdate(),
    subsolver = SteihaugTointCG(; max_iters=50),
    params    = TRSolverParams(tol=1e-8))
println(stats.status)
```
"""
function trust_region_radius(nlp::AbstractNLPModel{T, V};
                              rule::AbstractRadiusUpdate              = R1ClassicalUpdate(),
                              subsolver::AbstractTRSubproblemSolver   = SteihaugTointCG(),
                              params::TRSolverParams                  = TRSolverParams{T}(),
                              kwargs...) where {T, V}
    solver = TRRSolver(nlp; rule=rule, subsolver=subsolver, params=params)
    return solve!(solver, nlp; kwargs...)
end
