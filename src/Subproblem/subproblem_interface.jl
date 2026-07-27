
# ============================================================
# subproblem_interface.jl
#
# Abstract interface and concrete implementations for
# trust-region subproblem solvers.
#
# All implementations write the step into a caller-provided buffer
# `p::V` so the outer solver loop never allocates.
# ============================================================

"""
    AbstractTRSubproblemSolver

Abstract supertype for trust-region subproblem solvers.
Concrete subtypes implement:

    solve_subproblem!(subsolver, nlp, x, g, Δ, p, Hbuf) -> on_boundary::Bool

which solves

    min_{‖s‖ ≤ Δ}  gᵀs + ½ sᵀH(x)s

writing the optimal step into the caller-provided buffer `p`.  The
buffer `Hbuf` may be used as scratch for a Hessian-vector product.

The function returns a single boolean indicating whether the returned
step lies on the trust-region boundary.
"""
abstract type AbstractTRSubproblemSolver end

"""
    solve_subproblem!(subsolver, nlp, x, g, Δ, p, Hbuf) -> on_boundary::Bool

Solve the trust-region subproblem and write the step into `p`.

# Arguments
- `subsolver`: concrete subproblem solver
- `nlp`:       NLPModel (provides `hprod!` / `hess_op`)
- `x`:         current iterate
- `g`:         current gradient
- `Δ`:         trust-region radius
- `p`:         output buffer for the step (filled in place, same type as `x`)
- `Hbuf`:     scratch buffer for Hessian-vector products (same type as `x`)

# Returns
`on_boundary::Bool` -- `true` if ‖p‖ ≈ Δ, `false` otherwise.
"""
function solve_subproblem! end


# ============================================================
# SteihaugTointCG -- hand-rolled Steihaug-Toint truncated CG
# ============================================================

"""
    SteihaugTointCG(; χ=0.1, θ=0.5, max_iters=100)

Steihaug-Toint truncated CG subproblem solver.

# Fields
- `χ`:         forcing-function constant
- `θ`:         forcing-function exponent
- `max_iters`: maximum CG iterations
"""
struct SteihaugTointCG <: AbstractTRSubproblemSolver
    χ::Float64
    θ::Float64
    max_iters::Int
    SteihaugTointCG(; χ=0.1, θ=0.5, max_iters=100) = new(χ, θ, max_iters)
end

function solve_subproblem!(sub::SteihaugTointCG,
                            nlp::AbstractNLPModel{T, V},
                            x::V, g::V, Δ::T,
                            p::V, Hbuf::V) where {T, V}
    n     = length(g)
    normg = norm(g)

    # Zero-gradient guard
    if normg == 0
        fill!(p, zero(T))
        return false
    end

    fill!(p, zero(T))                 # step
    r  = similar(g);  @. r = -g       # residual    (negative gradient)
    d  = similar(g);  @. d =  r       # direction
    cand = similar(g)                 # scratch for the trial step p + α d
    rs_old = dot(r, r)

    on_boundary = false
    threshold = min(T(sub.χ), normg^T(sub.θ)) * normg

    for _ in 1:sub.max_iters
        hprod!(nlp, x, d, Hbuf)         # Hbuf = H(x) * d   (in place)
        dHd = dot(d, Hbuf)

        if dHd <= 0
            τ = _find_tr_boundary(p, d, Δ)
            @. p += τ * d
            return true
        end

        α = rs_old / dHd

        # Step would exceed the trust region?  Truncate to the boundary.
        # Use a dedicated scratch buffer so Hbuf (= H·d) is preserved.
        @. cand = p + α * d
        if norm(cand) > Δ
            τ = _find_tr_boundary(p, d, Δ)
            @. p += τ * d
            return true
        end
        @. p = cand                     # accept the full CG step

        # r_new = r - α * Hd   (Hbuf still holds H·d -- no recompute needed)
        @. r -= α * Hbuf
        rs_new = dot(r, r)

        if sqrt(rs_new) < threshold
            return on_boundary
        end

        β = rs_new / rs_old
        @. d = r + β * d
        rs_old = rs_new
    end

    return on_boundary
end

# Find τ > 0 such that ‖p + τ d‖ = Δ
@inline function _find_tr_boundary(p::AbstractVector{T}, d::AbstractVector{T}, Δ::T) where {T}
    a = dot(d, d)
    b = 2 * dot(p, d)
    c = dot(p, p) - Δ^2
    disc = b^2 - 4 * a * c
    disc = disc < 0 ? zero(T) : disc
    return (-b + sqrt(disc)) / (2 * a)
end


# ============================================================
# KrylovCG -- Krylov.cg with trust-region radius
# ============================================================

"""
    KrylovCG(; atol=1e-6, rtol=1e-6, itmax=0)

Trust-region subproblem solver wrapping `Krylov.cg` with the
`radius` keyword (Steihaug-style truncation at the TR boundary).

# Fields
- `atol`:  absolute tolerance
- `rtol`:  relative tolerance
- `itmax`: maximum CG iterations (0 = Krylov default)
"""
struct KrylovCG <: AbstractTRSubproblemSolver
    atol::Float64
    rtol::Float64
    itmax::Int
    KrylovCG(; atol=1e-6, rtol=1e-6, itmax=0) = new(atol, rtol, itmax)
end

function solve_subproblem!(sub::KrylovCG,
                            nlp::AbstractNLPModel{T, V},
                            x::V, g::V, Δ::T,
                            p::V, Hbuf::V) where {T, V}
    op = hess_op(nlp, x)
    rhs = similar(g)
    @. rhs = -g
    sol, stats = Krylov.cg(op, rhs;
                            radius = Δ,
                            atol   = T(sub.atol),
                            rtol   = T(sub.rtol),
                            itmax  = sub.itmax)
    copyto!(p, sol)
    return _get_on_boundary(stats, p, Δ)
end


# ============================================================
# KrylovCGLanczos -- Krylov.cg_lanczos with trust-region radius
# ============================================================

"""
    KrylovCGLanczos(; atol=1e-6, rtol=1e-6, itmax=0)

Trust-region subproblem solver wrapping `Krylov.cg_lanczos` with
`radius`.  Handles indefinite Hessians gracefully.
"""
struct KrylovCGLanczos <: AbstractTRSubproblemSolver
    atol::Float64
    rtol::Float64
    itmax::Int
    KrylovCGLanczos(; atol=1e-6, rtol=1e-6, itmax=0) = new(atol, rtol, itmax)
end

function solve_subproblem!(sub::KrylovCGLanczos,
                            nlp::AbstractNLPModel{T, V},
                            x::V, g::V, Δ::T,
                            p::V, Hbuf::V) where {T, V}
    op = hess_op(nlp, x)
    rhs = similar(g)
    @. rhs = -g
    sol, stats = Krylov.cg_lanczos(op, rhs;
                                    radius = Δ,
                                    atol   = T(sub.atol),
                                    rtol   = T(sub.rtol),
                                    itmax  = sub.itmax)
    copyto!(p, sol)
    return _get_on_boundary(stats, p, Δ)
end


# ============================================================
# Helper: extract on_boundary flag from Krylov stats
#
# The `on_boundary` field has moved between Krylov.jl versions.
# Inspect `propertynames(stats)` at runtime and fall back to a
# norm comparison if neither field is present.
# ============================================================

@inline function _get_on_boundary(stats, p, Δ)
    if hasproperty(stats, :on_boundary)
        return stats.on_boundary
    elseif hasproperty(stats, :solved) && hasproperty(stats, :niter)
        # No explicit flag: decide by step norm
        return norm(p) >= (1 - 1e-8) * Δ
    else
        return norm(p) >= (1 - 1e-8) * Δ
    end
end
