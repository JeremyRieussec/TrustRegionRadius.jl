
# ============================================================
# subproblem_interface.jl
#
# Abstract interface and concrete implementations for
# trust-region subproblem solvers.
# ============================================================

"""
    AbstractTRSubproblemSolver

Abstract supertype for trust-region subproblem solvers.
Concrete subtypes implement:

    solve_subproblem!(subsolver, nlp, x, g, Δ) -> (step::V, on_boundary::Bool)
"""
abstract type AbstractTRSubproblemSolver end

"""
    solve_subproblem!(subsolver, nlp, x, g, Δ) -> (step, on_boundary::Bool)

Solve the trust-region subproblem

    min_{‖s‖ ≤ Δ}  gᵀs + ½sᵀHs

and return the step `s` together with a boolean indicating whether the
solution lies on the trust-region boundary.

# Arguments
- `subsolver`: concrete subproblem solver
- `nlp`:       NLPModel (provides `hprod` / `hess_op`)
- `x`:         current iterate
- `g`:         current gradient
- `Δ`:         trust-region radius
"""
function solve_subproblem! end


# ------------------------------------------------------------
# SteihaugTointCG — wraps the existing truncated_cg_steihaug
# ------------------------------------------------------------

"""
    SteihaugTointCG <: AbstractTRSubproblemSolver

Steihaug–Toint truncated CG subproblem solver.

Wraps the existing `truncated_cg_steihaug` function with configurable
parameters.

# Fields
- `χ`:         forcing-function constant (default 0.1)
- `θ`:         forcing-function exponent (default 0.5)
- `max_iters`: maximum CG iterations (default 100)
"""
struct SteihaugTointCG <: AbstractTRSubproblemSolver
    χ::Float64
    θ::Float64
    max_iters::Int
    SteihaugTointCG(; χ=0.1, θ=0.5, max_iters=100) = new(χ, θ, max_iters)
end

function solve_subproblem!(solver::SteihaugTointCG,
                            nlp::AbstractNLPModel,
                            x, g, Δ)
    p, on_bnd, _ = truncated_cg_steihaug(nlp, x, g, Δ;
                                          χ=solver.χ,
                                          θ=solver.θ,
                                          max_iters=solver.max_iters)
    return p, on_bnd
end


# ------------------------------------------------------------
# KrylovCG — Krylov.cg with trust-region radius
# ------------------------------------------------------------

"""
    KrylovCG <: AbstractTRSubproblemSolver

Trust-region subproblem solver using `Krylov.cg` with a trust-region
radius constraint.

Requires the `Krylov` package to be loaded in the calling environment.

# Fields
- `atol`:   absolute tolerance (default 1e-6)
- `rtol`:   relative tolerance (default 1e-6)
- `itmax`:  maximum CG iterations (0 = use Krylov default)
"""
struct KrylovCG <: AbstractTRSubproblemSolver
    atol::Float64
    rtol::Float64
    itmax::Int
    KrylovCG(; atol=1e-6, rtol=1e-6, itmax=0) = new(atol, rtol, itmax)
end

function solve_subproblem!(solver::KrylovCG,
                            nlp::AbstractNLPModel,
                            x::V, g::V, Δ::T) where {T, V}
    op = hess_op(nlp, x)
    p, stats = Krylov.cg(op, -g;
                          radius=T(Δ),
                          atol=solver.atol,
                          rtol=solver.rtol,
                          itmax=solver.itmax)
    on_bnd = stats.on_boundary
    return V(p), on_bnd
end


# ------------------------------------------------------------
# KrylovCGLanczos — Krylov.cg_lanczos with trust-region radius
# ------------------------------------------------------------

"""
    KrylovCGLanczos <: AbstractTRSubproblemSolver

Trust-region subproblem solver using `Krylov.cg_lanczos` with a
trust-region radius constraint.

Requires the `Krylov` package to be loaded in the calling environment.

# Fields
- `atol`:   absolute tolerance (default 1e-6)
- `rtol`:   relative tolerance (default 1e-6)
- `itmax`:  maximum iterations (0 = use Krylov default)
"""
struct KrylovCGLanczos <: AbstractTRSubproblemSolver
    atol::Float64
    rtol::Float64
    itmax::Int
    KrylovCGLanczos(; atol=1e-6, rtol=1e-6, itmax=0) = new(atol, rtol, itmax)
end

function solve_subproblem!(solver::KrylovCGLanczos,
                            nlp::AbstractNLPModel,
                            x::V, g::V, Δ::T) where {T, V}
    op = hess_op(nlp, x)
    p, stats = Krylov.cg_lanczos(op, -g;
                                   radius=T(Δ),
                                   atol=solver.atol,
                                   rtol=solver.rtol,
                                   itmax=solver.itmax)
    on_bnd = stats.on_boundary
    return V(p), on_bnd
end
