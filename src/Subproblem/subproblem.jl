# =============================================================================
# src/Subproblem/subproblem.jl
#
# Trust-region subproblem solvers.
#
# One abstract type (`SubproblemSolver`) and one entry point:
#
#     solve_subproblem!(sub, model, nlp, x, g, Δ, s, Hbuf) -> active::Bool
#
# The model Hessian is an argument rather than being taken from `nlp`, so the
# same solvers serve the exact Hessian and every quasi-Newton model. All write
# the step into the caller's buffer and allocate nothing per call beyond their
# own scratch.
#
# `active` is `true` when the returned step lies on the trust-region boundary.
# It is worth propagating: the fraction of iterations on which the constraint
# binds is the observable that separates radius mechanisms whose first-order
# behaviour is identical.
# =============================================================================

"""
    SubproblemSolver

Abstract supertype for trust-region subproblem solvers. A concrete subtype
implements

    solve_subproblem!(sub, model, nlp, x, g, Δ, s, Hbuf) -> active::Bool

solving `min_{‖s‖ ≤ Δ} gᵀs + ½ sᵀHs` and writing the step into `s`.
"""
abstract type SubproblemSolver end

"""
    solve_subproblem!(sub, model, nlp, x, g, Δ, s, Hbuf) -> active::Bool

Solve the trust-region subproblem, writing the step into `s`.

- `sub`:   the solver
- `model`: [`ModelHessian`](@ref) supplying the curvature
- `nlp`:   the problem, for `hessian_op`
- `x, g`:  current iterate and gradient
- `Δ`:     radius
- `s`:     output buffer for the step
- `Hbuf`:  scratch for Hessian-vector products

Returns `true` if `‖s‖ = Δ`.
"""
function solve_subproblem! end

# =============================================================================
# SteihaugCG
# =============================================================================

"""
    SteihaugCG(; χ = 0.1, θ = 0.5, max_iters = 100)

Steihaug–Toint truncated conjugate gradient. Stops on negative curvature, on
reaching the boundary, or when the residual satisfies the forcing condition
`‖r‖ ≤ min(χ, ‖g‖^θ)‖g‖`.

Needs only `B * v`, so it scales to large problems with any operator-backed
model.

!!! note "The Cauchy point"
    The first CG direction is `-g`. A step that leaves the region on CG
    iteration 1 is therefore exactly the Cauchy point, and the model Hessian
    has had no influence on its direction. When the radius is small — a low
    `μ_max` in [`RGradCapped`](@ref), a low `ζ` in [`RDFO`](@ref) — this
    happens at every iteration and the method is gradient descent in disguise.
    `cg_iters` in the return of [`cg_step_info`](@ref) reveals it.
"""
struct SteihaugCG <: SubproblemSolver
    χ::Float64
    θ::Float64
    max_iters::Int
    SteihaugCG(; χ::Float64 = 0.1, θ::Float64 = 0.5, max_iters::Int = 100) =
        new(χ, θ, max_iters)
end

function solve_subproblem!(sub::SteihaugCG, model::ModelHessian,
                           nlp::AbstractNLPModel{T, V},
                           x::V, g::V, Δ::T, s::V, Hbuf::V) where {T, V}
    B = hessian_op(model, nlp, x)
    return _steihaug!(sub, B, g, Δ, s, Hbuf)
end

function _steihaug!(sub::SteihaugCG, B, g::V, Δ::T, s::V, Hbuf::V) where {T, V}
    gn = norm(g)
    if gn == 0
        fill!(s, zero(T)); return false
    end

    fill!(s, zero(T))
    r = similar(g); @. r = -g
    d = similar(g); @. d = r
    cand = similar(g)
    rs = dot(r, r)
    threshold = min(T(sub.χ), gn^T(sub.θ)) * gn

    for _ in 1:sub.max_iters
        _apply!(Hbuf, B, d)
        dBd = dot(d, Hbuf)
        if dBd <= 0                                   # negative curvature
            temp = _to_boundary(s, d, Δ)
            @. s += temp * d
            return true
        end
        α = rs / dBd
        @. cand = s + α * d
        if norm(cand) >= Δ                            # boundary
            temp = _to_boundary(s, d, Δ)
            @. s += temp * d
            return true
        end
        @. s = cand
        @. r -= α * Hbuf
        rs_new = dot(r, r)
        sqrt(rs_new) < threshold && return false
        @. d = r + (rs_new / rs) * d
        rs = rs_new
    end
    return false
end

"Positive root τ of ‖s + τd‖ = Δ."
@inline function _to_boundary(s::AbstractVector{T}, d::AbstractVector{T}, Δ::T) where {T}
    a = dot(d, d)
    a <= 0 && return zero(T)
    b = 2 * dot(s, d)
    c = dot(s, s) - Δ^2
    disc = max(b^2 - 4a * c, zero(T))
    return (-b + sqrt(disc)) / (2a)
end

"""
    _apply!(y, B, v)

`y ← B*v`, using `mul!` where available and falling back to `B*v` otherwise.
The fallback exists for `UniformScaling` and for operators that define `*` but
not a three-argument `mul!` for every element type.
"""
@inline function _apply!(y, B, v)
    try
        mul!(y, B, v)
    catch
        copyto!(y, B * v)
    end
    return y
end

# =============================================================================
# ExactMS
# =============================================================================

"""
    ExactMS(; nmax = 200, tol = 1e-12, maxit = 100)

Exact step by the Moré–Sorensen characterisation: `s*` is optimal iff there is
`λ ≥ 0` with

    (B + λI)s* = −g,   B + λI ⪰ 0,   λ(Δ − ‖s*‖) = 0.

`‖s(λ)‖` decreases strictly in `λ`, so the root is bracketed and bisected.

Handles **indefinite** `B`, including the hard case (`g` orthogonal to the
eigenspace of `λ_min(B)`, where no `λ` attains `‖s(λ)‖ = Δ`), by adding the
right multiple of that eigenvector. That matters with the exact Hessian or SR1,
where indefiniteness is exactly the information one wants to exploit.

Requires a dense eigendecomposition, hence `n ≤ nmax`; larger problems raise
rather than silently allocating an `n × n` array.
"""
struct ExactMS <: SubproblemSolver
    nmax::Int
    tol::Float64
    maxit::Int
    ExactMS(; nmax::Int = 200, tol::Float64 = 1e-12, maxit::Int = 100) =
        new(nmax, tol, maxit)
end

function solve_subproblem!(sub::ExactMS, model::ModelHessian,
                           nlp::AbstractNLPModel{T, V},
                           x::V, g::V, Δ::T, s::V, Hbuf::V) where {T, V}
    n = length(g)
    n <= sub.nmax || throw(ArgumentError(
        "ExactMS: n = $n exceeds nmax = $(sub.nmax); use SteihaugCG or raise nmax"))
    if norm(g) == 0
        fill!(s, zero(T)); return false
    end
    B = dense_hessian(model, nlp, x)
    step, active = _exact_ms(B, Vector{T}(g), Δ, sub)
    copyto!(s, step)
    return active
end

"Dense Moré–Sorensen kernel; separated so it can be tested on a bare matrix."
function _exact_ms(B::AbstractMatrix{T}, g::Vector{T}, Δ::T, sub::ExactMS) where {T}
    n = length(g)
    F = eigen(Symmetric(Matrix(B)))
    w, Q = F.values, F.vectors
    λmin = w[1]
    gq = Q' * g

    if λmin > 0                                       # interior Newton point?
        sN = -(Symmetric(Matrix(B)) \ g)
        norm(sN) <= Δ && return sN, false
    end

    snorm(λ) = sqrt(sum(abs2, gq ./ (w .+ λ)))

    idx = abs.(w .- λmin) .< 1e-12                    # hard case
    if λmin <= 0 && all(abs.(gq[idx]) .< 1e-12)
        safe = .!idx
        sbar = zeros(T, n)
        any(safe) && (sbar[safe] = -gq[safe] ./ (w[safe] .- λmin))
        ns = norm(sbar)
        if ns <= Δ
            τ = sqrt(max(Δ^2 - ns^2, zero(T)))
            return Q * (sbar .+ τ .* T.(idx)), true
        end
    end

    lo = max(zero(T), -λmin) + T(1e-14)
    hi = max(lo + one(T), norm(g) / max(Δ, T(1e-300)))  # ‖s(λ)‖ ≈ ‖g‖/λ for large λ
    it = 0
    while snorm(hi) > Δ && it < 200
        hi = lo + 2(hi - lo); it += 1
        hi > T(1e300) && break
    end
    for _ in 1:sub.maxit
        mid = (lo + hi) / 2
        snorm(mid) > Δ ? (lo = mid) : (hi = mid)
        hi - lo <= sub.tol * max(one(T), hi) && break
    end
    return -Q * (gq ./ (w .+ hi)), true
end

# =============================================================================
# Krylov wrappers
# =============================================================================

"""
    KrylovCG(; atol = 1e-6, rtol = 1e-6, itmax = 0)

`Krylov.cg` with the `radius` keyword (Steihaug-style truncation).
Assumes positive definite `B`; use [`KrylovCGLanczos`](@ref) otherwise.
"""
struct KrylovCG <: SubproblemSolver
    atol::Float64
    rtol::Float64
    itmax::Int
    KrylovCG(; atol::Float64 = 1e-6, rtol::Float64 = 1e-6, itmax::Int = 0) =
        new(atol, rtol, itmax)
end

"""
    KrylovCGLanczos(; atol = 1e-6, rtol = 1e-6, itmax = 0)

`Krylov.cg_lanczos` with `radius`; handles indefinite `B`.
"""
struct KrylovCGLanczos <: SubproblemSolver
    atol::Float64
    rtol::Float64
    itmax::Int
    KrylovCGLanczos(; atol::Float64 = 1e-6, rtol::Float64 = 1e-6, itmax::Int = 0) =
        new(atol, rtol, itmax)
end

for (Tsub, fn) in ((:KrylovCG, :(Krylov.cg)), (:KrylovCGLanczos, :(Krylov.cg_lanczos)))
    @eval function solve_subproblem!(sub::$Tsub, model::ModelHessian,
                                     nlp::AbstractNLPModel{T, V},
                                     x::V, g::V, Δ::T, s::V, Hbuf::V) where {T, V}
        B = hessian_op(model, nlp, x)
        rhs = similar(g); @. rhs = -g
        sol, st = $fn(B, rhs; radius = Δ, atol = T(sub.atol),
                      rtol = T(sub.rtol), itmax = sub.itmax)
        copyto!(s, sol)
        return _on_boundary(st, s, Δ)
    end
end

"""
    _on_boundary(stats, s, Δ) -> Bool

Extract the boundary flag from a Krylov stats object, falling back to a norm
comparison. The field has moved between Krylov.jl versions, so both paths are
kept.
"""
@inline function _on_boundary(stats, s, Δ)
    hasproperty(stats, :on_boundary) && return stats.on_boundary
    return norm(s) >= (1 - 1e-8) * Δ
end

# =============================================================================
# Diagnostics
# =============================================================================

"""
    cg_step_info(sub, model, nlp, x, g, Δ) -> (; cg_iters, active, cos_cauchy)

Run [`SteihaugCG`](@ref) and report how the step was produced:

- `cg_iters`:   CG iterations taken
- `active`:     whether the step ended on the boundary
- `cos_cauchy`: `cos(s, −g)`, which is `1` exactly when the step is the Cauchy
                point

`cg_iters == 1 && active` and `cos_cauchy ≈ 1` together certify that the model
Hessian played no part in choosing the direction. Allocating, and intended for
diagnostics rather than the solver loop.
"""
function cg_step_info(sub::SteihaugCG, model::ModelHessian,
                      nlp::AbstractNLPModel{T, V}, x::V, g::V, Δ::T) where {T, V}
    B = hessian_op(model, nlp, x)
    s = similar(g); Hbuf = similar(g)
    gn = norm(g)
    gn == 0 && return (cg_iters = 0, active = false, cos_cauchy = T(NaN))

    fill!(s, zero(T))
    r = similar(g); @. r = -g
    d = similar(g); @. d = r
    cand = similar(g)
    rs = dot(r, r)
    threshold = min(T(sub.χ), gn^T(sub.θ)) * gn

    for j in 1:sub.max_iters
        _apply!(Hbuf, B, d)
        dBd = dot(d, Hbuf)
        if dBd <= 0
            temp =  _to_boundary(s, d, Δ)
            @. s += temp * d
            return (cg_iters = j, active = true, cos_cauchy = dot(s, -g)/(norm(s)*gn))
        end
        α = rs / dBd
        @. cand = s + α * d
        if norm(cand) >= Δ
            temp =  _to_boundary(s, d, Δ)
            @. s += temp * d
            return (cg_iters = j, active = true, cos_cauchy = dot(s, -g)/(norm(s)*gn))
        end
        @. s = cand
        @. r -= α * Hbuf
        rs_new = dot(r, r)
        if sqrt(rs_new) < threshold
            return (cg_iters = j, active = false, cos_cauchy = dot(s, -g)/(norm(s)*gn))
        end
        @. d = r + (rs_new / rs) * d
        rs = rs_new
    end
    return (cg_iters = sub.max_iters, active = false,
            cos_cauchy = dot(s, -g)/(norm(s)*gn))
end

Base.show(io::IO, s::SubproblemSolver) =
    print(io, nameof(typeof(s)), "(",
          join(("$(f)=$(getfield(s,f))" for f in fieldnames(typeof(s))), ", "), ")")
