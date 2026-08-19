# =============================================================================
# src/Subproblem/subproblem.jl
#
# Trust-region subproblem solvers.
#
# One abstract type (`SubproblemSolver`) and one entry point:
#
#     solve_subproblem!(sub, model, nlp, x, g, Δ, s, Hs, ws; curv) -> active::Bool
#
# The model Hessian is an argument rather than being taken from `nlp`, so the
# same solvers serve the exact Hessian and every quasi-Newton model.
#
# THREE CHANGES FROM THE PREVIOUS INTERFACE, all of them to stop the subsolver
# distorting the benchmark it is supposed to be measured by.
#
#   `ws`   A caller-owned, *typed* workspace. `_steihaug!` previously allocated
#          three n-vectors on every call, despite this header claiming the
#          solvers "allocate nothing per call beyond their own scratch". The
#          workspace lives on `TRSolver` and is parametric in the vector type,
#          so nothing is allocated in the loop and nothing is type-unstable.
#
#   `Hs`   Now an *output*: on return it holds `B·s` for any subsolver declaring
#          `returns_hprod`. CG already forms this incrementally, so recomputing
#          it in the solver to get the predicted reduction was one wasted
#          Hessian-vector product per iteration.
#
#   `curv` An optional `(λ_min, v_min)` supplied by the solver when it has
#          already computed it for a τ-anchored rule or for `tol_H`. `EigenPoint`
#          previously recomputed the eigendecomposition the solver had just
#          done — two dense `eigen` calls per iteration on the dense branch.
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

    solve_subproblem!(sub, model, nlp, x, g, Δ, s, Hs, ws; curv = nothing) -> Bool

solving `min_{‖s‖ ≤ Δ} gᵀs + ½ sᵀHs` and writing the step into `s`.
"""
abstract type SubproblemSolver end

"""
    SubWorkspace(x)

Scratch vectors for the subproblem solvers, allocated once per `TRSolver` and
parametric in the problem's vector type.

Four buffers: the CG residual `r`, the direction `d`, the trial iterate `cand`,
and `Hd = B·d`. `EigenPoint` reuses `cand` and `Hd` after the inner solver has
finished with them, and the trust-region loop reuses `cand` after
`solve_subproblem!` has returned, to form the two residual diagnostics without
allocating.

It also carries `iters`, the inner-iteration count of the last solve. That count
is what distinguishes a step the model shaped from the Cauchy point: truncated CG
returning after one iteration has taken the step along `-g` and the model Hessian
has influenced nothing. `0` for solvers with no meaningful iteration count.
"""
mutable struct SubWorkspace{V <: AbstractVector}
    r::V
    d::V
    cand::V
    Hd::V
    iters::Int
end

SubWorkspace(x::AbstractVector) =
    SubWorkspace(similar(x), similar(x), similar(x), similar(x), 0)

# Positional four-buffer form, kept so that existing call sites and tests that
# build a workspace from their own vectors continue to work.
SubWorkspace(r::V, d::V, cand::V, Hd::V) where {V <: AbstractVector} =
    SubWorkspace{V}(r, d, cand, Hd, 0)

"""
    solve_subproblem!(sub, model, nlp, x, g, Δ, s, Hs, ws; curv = nothing) -> Bool

Solve the trust-region subproblem, writing the step into `s`.

- `sub`:   the solver
- `model`: [`ModelHessian`](@ref) supplying the curvature
- `nlp`:   the problem, for `hessian_op`
- `x, g`:  current iterate and gradient
- `Δ`:     radius
- `s`:     output buffer for the step
- `Hs`:    output buffer; holds `B·s` on return iff `returns_hprod(sub)`
- `ws`:    [`SubWorkspace`](@ref) scratch
- `curv`:  optional `(λ_min, v_min)` already computed by the caller

Returns `true` if `‖s‖ = Δ`.
"""
function solve_subproblem! end

"""
    solve_subproblem!(sub, model, nlp, x, g, Δ, s, Hs; curv = nothing) -> Bool

Convenience method that allocates its own [`SubWorkspace`](@ref).

For interactive use, tests, and one-off diagnostics. The solver loop calls the
nine-argument form with a workspace it owns, so nothing is allocated per
iteration; this method exists so that a call site with a step buffer and a
Hessian buffer to hand does not have to construct one.
"""
function solve_subproblem!(sub::SubproblemSolver, model::ModelHessian,
                           nlp::AbstractNLPModel{T, V},
                           x::V, g::V, Δ::T, s::V, Hs::V;
                           curv = nothing) where {T, V}
    return solve_subproblem!(sub, model, nlp, x, g, Δ, s, Hs,
                             SubWorkspace(s); curv = curv)
end

"""
    returns_hprod(sub) -> Bool

Whether `solve_subproblem!` leaves `B·s` in the `Hs` buffer. When `true` the
solver skips its own `model_hprod!` for the predicted reduction, saving one
Hessian-vector product per iteration.
"""
returns_hprod(::SubproblemSolver) = false

"""
    needs_eigenvector(sub) -> Bool

Whether the subsolver wants the leftmost eigenvector in `curv`. The solver then
requests `lambda_min_estimate(...; vector = true)` when it is computing the
curvature anyway, so the estimate is paid for once rather than twice.
"""
needs_eigenvector(::SubproblemSolver) = false

# =============================================================================
# SteihaugCG
# =============================================================================

"""
    SteihaugCG(; χ = 0.1, θ = 0.5, max_iters = 100)

Steihaug–Toint truncated conjugate gradient. Stops on negative curvature, on
reaching the boundary, or when the residual satisfies the forcing condition
`‖r‖ ≤ min(χ, ‖g‖^θ)‖g‖`.

Needs only `B * v`, so it scales to large problems with any operator-backed
model. Accumulates `B·s` alongside `s`, so the predicted reduction costs no
extra product.

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
    function SteihaugCG(; χ::Real = 0.1, θ::Real = 0.5, max_iters::Int = 100)
        χ > 0 || throw(ArgumentError("SteihaugCG: need χ > 0, got $χ"))
        max_iters > 0 || throw(ArgumentError("SteihaugCG: need max_iters > 0"))
        new(float(χ), float(θ), max_iters)
    end
end

returns_hprod(::SteihaugCG) = true

function solve_subproblem!(sub::SteihaugCG, model::ModelHessian,
                           nlp::AbstractNLPModel{T, V},
                           x::V, g::V, Δ::T, s::V, Hs::V,
                           ws::SubWorkspace{V}; curv = nothing) where {T, V}
    B = hessian_op(model, nlp, x)
    active, iters = _steihaug!(sub, B, g, Δ, s, Hs, ws)
    ws.iters = iters
    return active
end

"""
    _steihaug!(sub, B, g, Δ, s, Hs, ws) -> (active, iters)

The Steihaug–Toint recurrence. Single implementation, shared with
[`cg_step_info`](@ref) — the two were previously separate copies of the same
loop, which is exactly the kind of duplication that drifts.

Writes the step into `s` and `B·s` into `Hs`, allocating nothing.
"""
function _steihaug!(sub::SteihaugCG, B, g::V, Δ::T, s::V, Hs::V,
                    ws::SubWorkspace{V}) where {T, V}
    fill!(s, zero(T))
    fill!(Hs, zero(T))

    gn = norm(g)
    (gn == 0 || Δ <= 0) && return (false, 0)

    r, d, cand, Hd = ws.r, ws.d, ws.cand, ws.Hd
    @. r = -g
    @. d = r
    rs = dot(r, r)
    threshold = min(T(sub.χ), gn^T(sub.θ)) * gn

    for j in 1:sub.max_iters
        _apply_op!(Hd, B, d)
        dBd = dot(d, Hd)

        if dBd <= 0                                   # negative curvature
            τ = _to_boundary(s, d, Δ)
            @. s += τ * d
            @. Hs += τ * Hd
            return (true, j)
        end

        α = rs / dBd
        @. cand = s + α * d
        if norm(cand) >= Δ                            # boundary
            τ = _to_boundary(s, d, Δ)
            @. s += τ * d
            @. Hs += τ * Hd
            return (true, j)
        end

        @. s = cand
        @. Hs += α * Hd
        @. r -= α * Hd
        rs_new = dot(r, r)
        sqrt(rs_new) < threshold && return (false, j)
        @. d = r + (rs_new / rs) * d
        rs = rs_new
    end
    return (false, sub.max_iters)
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
right multiple of a *single* eigenvector from that eigenspace. The previous
implementation added `τ` to every index within tolerance of `λ_min`, so with a
degenerate leftmost eigenvalue of multiplicity `m` it returned a step of norm
`√(‖s̄‖² + mτ²) ≠ Δ` — and degenerate `λ_min` is common on symmetric test
problems, which is precisely where the hard case arises.

Requires a dense eigendecomposition, hence `n ≤ nmax`; larger problems raise
rather than silently allocating an `n × n` array.
"""
struct ExactMS <: SubproblemSolver
    nmax::Int
    tol::Float64
    maxit::Int
    ExactMS(; nmax::Int = 200, tol::Real = 1e-12, maxit::Int = 100) =
        new(nmax, float(tol), maxit)
end

function solve_subproblem!(sub::ExactMS, model::ModelHessian,
                           nlp::AbstractNLPModel{T, V},
                           x::V, g::V, Δ::T, s::V, Hs::V,
                           ws::SubWorkspace{V}; curv = nothing) where {T, V}
    n = length(g)
    n <= sub.nmax || throw(ArgumentError(
        "ExactMS: n = $n exceeds nmax = $(sub.nmax); use SteihaugCG or raise nmax"))
    ws.iters = 0        # no meaningful inner-iteration count for a direct solve
    if norm(g) == 0 || Δ <= 0
        fill!(s, zero(T)); return false
    end
    B = dense_hessian(model, nlp, x)
    step, active = _exact_ms(Matrix{T}(B), Vector{T}(g), Δ, sub)
    copyto!(s, step)
    return active
end

"Dense Moré–Sorensen kernel; separated so it can be tested on a bare matrix."
function _exact_ms(B::AbstractMatrix{T}, g::Vector{T}, Δ::T, sub::ExactMS) where {T}
    n = length(g)
    Bs = Symmetric(Matrix(B))
    F = eigen(Bs)
    w, Q = F.values, F.vectors
    λmin = w[1]
    gq = Q' * g

    if λmin > 0                                       # interior Newton point?
        # Reuse the factorisation already computed rather than a second solve.
        sN = -Q * (gq ./ w)
        norm(sN) <= Δ && return sN, false
    end

    snorm(λ) = sqrt(sum(abs2, gq ./ (w .+ λ)))

    # --- hard case -----------------------------------------------------------
    tolw = 1e-12 * max(one(T), abs(λmin))
    idx = abs.(w .- λmin) .< tolw
    if λmin <= 0 && all(abs.(gq[idx]) .< 1e-12)
        safe = .!idx
        sbar = zeros(T, n)
        any(safe) && (sbar[safe] .= -gq[safe] ./ (w[safe] .- λmin))
        ns = norm(sbar)
        if ns <= Δ
            τ = sqrt(max(Δ^2 - ns^2, zero(T)))
            # ONE eigenvector from the degenerate eigenspace, not all of them:
            # adding τ to every index gives ‖s‖ = √(‖sbar‖² + mτ²) ≠ Δ.
            e = zeros(T, n)
            e[findfirst(idx)] = one(T)
            return Q * (sbar .+ τ .* e), true
        end
    end

    # --- bracket and bisect on λ --------------------------------------------
    lo = max(zero(T), -λmin) + T(1e-14)
    hi = max(lo + one(T), norm(g) / max(Δ, T(1e-300)))  # ‖s(λ)‖ ≈ ‖g‖/λ for large λ
    it = 0
    while snorm(hi) > Δ
        it += 1
        if it > 200 || hi > T(1e300)
            # The bracket never closed. Returning the bisection result here
            # would give a step *outside* the region while reporting
            # `active = true`; refuse instead.
            throw(ErrorException(
                "ExactMS: failed to bracket the Moré–Sorensen multiplier " *
                "(Δ = $Δ, ‖g‖ = $(norm(g)), λ_min = $λmin). " *
                "The model Hessian is probably not finite."))
        end
        hi = lo + 2(hi - lo)
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
Assumes positive definite `B`; use [`KrylovCR`](@ref) otherwise.
"""
struct KrylovCG <: SubproblemSolver
    atol::Float64
    rtol::Float64
    itmax::Int
    KrylovCG(; atol::Real = 1e-6, rtol::Real = 1e-6, itmax::Int = 0) =
        new(float(atol), float(rtol), itmax)
end

"""
    KrylovCR(; atol = 1e-6, rtol = 1e-6, itmax = 0)

`Krylov.cr` with the `radius` keyword; handles indefinite `B`, stepping to the
boundary on negative curvature exactly as [`SteihaugCG`](@ref) does.

!!! note "This was `KrylovCGLanczos`, and that wrapper never ran"
    The type was previously generated from `Krylov.cg_lanczos`, which **has no
    `radius` keyword** — `Base.kwarg_decl` on it lists `check_curvature` and no
    `radius` in any released version — so every call raised a `MethodError`
    before doing any work. The docstring's claim that it solved the constrained
    subproblem was therefore never exercised.

    `Krylov.cr` is the routine in the same package that does accept `radius`
    *and* handles indefinite `B`, so it is what the type always meant. The
    alternative, keeping `cg_lanczos` and rescaling its unconstrained iterate
    down to `‖s‖ = Δ`, was rejected: a scaled Newton-ish direction is not the
    trust-region step, and on indefinite `B` the iterate being rescaled can
    already have diverged.

    `KrylovCGLanczos` remains as a deprecated alias.
"""
struct KrylovCR <: SubproblemSolver
    atol::Float64
    rtol::Float64
    itmax::Int
    KrylovCR(; atol::Real = 1e-6, rtol::Real = 1e-6, itmax::Int = 0) =
        new(float(atol), float(rtol), itmax)
end

Base.@deprecate_binding KrylovCGLanczos KrylovCR true

for (Tsub, fn) in ((:KrylovCG, :(Krylov.cg)), (:KrylovCR, :(Krylov.cr)))
    @eval function solve_subproblem!(sub::$Tsub, model::ModelHessian,
                                     nlp::AbstractNLPModel{T, V},
                                     x::V, g::V, Δ::T, s::V, Hs::V,
                                     ws::SubWorkspace{V}; curv = nothing) where {T, V}
        B = hessian_op(model, nlp, x)
        @. ws.r = -g                                  # rhs, from the workspace
        sol, st = $fn(B, ws.r; radius = Δ, atol = T(sub.atol),
                      rtol = T(sub.rtol), itmax = sub.itmax)
        copyto!(s, sol)
        ws.iters = hasproperty(st, :niter) ? Int(getproperty(st, :niter)) : 0
        return _on_boundary(st, s, Δ)
    end
end

"""
    _on_boundary(stats, s, Δ) -> Bool

Extract the boundary flag from a Krylov stats object, falling back to a norm
comparison.

Krylov's `SimpleStats` does not carry an `on_boundary` field in current
versions, so in practice this takes the norm path; the previous comment claimed
"both paths are kept" for version robustness, which was aspirational rather than
tested. Pin the Krylov compat bound in `Project.toml` and assert the flag
explicitly in the tests if the distinction matters.
"""
@inline function _on_boundary(stats, s, Δ)
    hasproperty(stats, :on_boundary) && return getproperty(stats, :on_boundary)
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
Hessian played no part in choosing the direction.

Now a thin wrapper over `_steihaug!` rather than a second copy of the
recurrence. Allocating, and intended for diagnostics rather than the solver loop.
"""
function cg_step_info(sub::SteihaugCG, model::ModelHessian,
                      nlp::AbstractNLPModel{T, V}, x::V, g::V, Δ::T) where {T, V}
    gn = norm(g)
    gn == 0 && return (cg_iters = 0, active = false, cos_cauchy = T(NaN))

    B  = hessian_op(model, nlp, x)
    s  = similar(g)
    Hs = similar(g)
    ws = SubWorkspace(g)
    active, iters = _steihaug!(sub, B, g, Δ, s, Hs, ws)

    ns = norm(s)
    cosc = ns == 0 ? T(NaN) : dot(s, -g) / (ns * gn)
    return (cg_iters = iters, active = active, cos_cauchy = cosc)
end

Base.show(io::IO, s::SubproblemSolver) =
    print(io, nameof(typeof(s)), "(",
          join(("$(f)=$(getfield(s,f))" for f in fieldnames(typeof(s))), ", "), ")")
