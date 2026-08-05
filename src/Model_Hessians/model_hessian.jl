# =============================================================================
# src/Model_Hessians/model_hessian.jl
#
# The model-Hessian axis.
#
# The solver never forms a matrix: it needs only the operator action H·v, which
# is what `hessian_op` returns and what every Krylov subsolver consumes. This
# keeps large CUTEst problems feasible and lets quasi-Newton models be supplied
# as LinearOperators without changing a line of the solver loop.
#
# Interface contract
# ------------------
# A concrete `M <: ModelHessian` implements
#
#     reset_model!(m, n)            -- clear state for dimension n
#     hessian_op(m, nlp, x)         -- object supporting `*` and `mul!`
#     update_model!(m, s, y)        -- absorb the pair (s_k, y_k = g_{k+1} - g_k)
#
# and optionally
#
#     dense_hessian(m, nlp, x)               -- dense Matrix, small n only
#     reports_negative_curvature(m)          -- can λ_min(B) ever be < 0?
#     model_eltype(m)                        -- element type, or nothing
#     required_problem(m)                    -- narrowest problem class it is
#                                               defined for; see Problems/classes.jl
#
# `update_model!` is a no-op for models that carry no state.
# =============================================================================

"""
    ModelHessian

Abstract supertype for the model Hessian `H_k` used in the trust-region model

    m_k(s) = f(x_k) + g_kᵀ s + ½ sᵀ H_k s.

See [`ExactHessian`](@ref), [`LBFGSModel`](@ref), [`SR1Model`](@ref),
[`ScaledIdentity`](@ref), [`SPDTarget`](@ref).
"""
abstract type ModelHessian end

"""
    reset_model!(model, n)

Clear any mutable state, for a problem of dimension `n`. Called at the start of
every solve. Default: no-op.
"""
reset_model!(::ModelHessian, ::Int) = nothing

"""
    update_model!(model, s, y)

Absorb the secant pair `(s_k, y_k)` with `y_k = g_{k+1} - g_k`, on accepted
steps only. Default: no-op.
"""
update_model!(::ModelHessian, s, y) = nothing

"""
    hessian_op(model, nlp, x)

Return an object `B` supporting `B * v` and `mul!(y, B, v)`, representing `H_k`
at `x`.
"""
function hessian_op end

"""
    dense_hessian(model, nlp, x) -> Matrix

Dense representation, for diagnostics on small problems only.
"""
function dense_hessian end

"""
    reports_negative_curvature(model) -> Bool

Whether the model is *capable* of reporting `λ_min(B) < 0`.

`false` for every model that is positive (semi)definite by construction:
[`LBFGSModel`](@ref), [`ScaledIdentity`](@ref), [`SPDTarget`](@ref), and the
outer-product models `BHHHModel`, `BHHH2Model`, `GaussNewtonModel`.

This trait exists because four separate docstrings previously asserted the
consequence — that `SecondOrder` over such a model gives `τ ≡ ‖g‖` and a
`:second_order` status certifying nothing — and one of them claimed the solver
warned about it, while nothing in the solver did. `TRSolver` now consults this
and warns once at construction. The failure mode is silent and the diagnosis is
a one-line trait, so it should not be prose.
"""
reports_negative_curvature(::ModelHessian) = true

"""
    model_eltype(model) -> Type or nothing

The element type a stateful model was constructed for, or `nothing` when the
model carries no typed state. `TRSolver` checks it against the problem's element
type and raises rather than letting a `Float64` quasi-Newton operator be applied
to `Float32` vectors deep inside the CG recurrence.
"""
model_eltype(::ModelHessian) = nothing

"""
    _apply_op!(y, B, v) -> y

`y ← B*v`, by dispatch rather than by `try`/`catch`.

The previous implementation wrapped `mul!` in a bare `catch` and fell back to
`B * v`. That swallowed *every* exception type — including the `DomainError`
that `solve!` catches to report `:exception`, which survived only because the
fallback threw it again — and it prevented the fallback from being resolved
statically inside the CG inner loop. `UniformScaling` is the only case the old
comment actually named, and it gets its own method here.
"""
@inline _apply_op!(y, B, v) = (mul!(y, B, v); y)
@inline _apply_op!(y, B::UniformScaling, v) = (@. y = B.λ * v; y)
@inline _apply_op!(y, B::AbstractMatrix, v) = (mul!(y, B, v); y)

# -----------------------------------------------------------------------------
# ExactHessian
# -----------------------------------------------------------------------------

"""
    ExactHessian()

The true Hessian `∇²f(x_k)`, as an operator via `NLPModels.hess_op`.

Note that the exact Hessian does *not* satisfy the secant equation
`H_{k+1} s_k = y_k`, so retrospective rules relying on that condition
(the Fan–Pan–Song route to `ρ̃ → 1`) do not apply to it; the Bastin route,
which needs asymptotic second-order coherence instead, does.
"""
struct ExactHessian <: ModelHessian end

hessian_op(::ExactHessian, nlp, x) = hess_op(nlp, x)

"""
    dense_hessian(::ExactHessian, nlp, x) -> Matrix

`NLPModels.hess` is documented to return the **lower triangle**, but several
model types (ADNLPModels among them) return a `Symmetric` wrapper, for which
`Matrix` already gives the full matrix. Wrapping unconditionally in
`Symmetric(·, :L)` is correct in both cases: on a `Symmetric{...,:L}` it is the
identity, and on a bare lower triangle it mirrors.

The convention is asserted here rather than assumed, because the previous code
made the opposite assumption in `batch_hess(::PerturbedSum, ...)` and doubled
every off-diagonal entry there. `_full_hessian` is now the single place that
knows the answer, and both call sites use it.
"""
dense_hessian(::ExactHessian, nlp, x) = _full_hessian(hess(nlp, x))

"""
    _full_hessian(H) -> Matrix

Materialise whatever `NLPModels.hess` returned as a full dense symmetric matrix.

Handles the three shapes actually observed in the ecosystem: a `Symmetric`
wrapper (already full), a dense matrix that is already symmetric, and a bare
lower triangle. This is the fix for the off-diagonal doubling that previously
corrupted every `PerturbedSum` run.
"""
function _full_hessian(H::Symmetric)
    return Matrix(H)
end
function _full_hessian(H::AbstractMatrix)
    A = Matrix(H)
    issymmetric(A) && return A
    # a bare triangle: mirror it, without doubling the diagonal
    return A .+ transpose(A) .- Diagonal(diag(A))
end

reports_negative_curvature(::ExactHessian) = true
Base.show(io::IO, ::ExactHessian) = print(io, "exact ∇²f")

# -----------------------------------------------------------------------------
# LBFGSModel
# -----------------------------------------------------------------------------

"""
    LBFGSModel(; mem = 5, T = Float64)

Limited-memory BFGS model Hessian, backed by `LinearOperators.LBFGSOperator`.

The operator is stored in a *typed* field, `Union{Nothing, LBFGSOperator{T}}`,
so `hessian_op` is type-stable and every `mul!` in the CG recurrence resolves at
compile time. The previous `op::Any` silently defeated the `M <: ModelHessian`
type parameter on `TRSolver`, whose whole purpose is that "every dispatch in the
loop is resolved at compile time"; the comment justifying `Any` cited avoiding a
hard dependency on LinearOperators, which the module already `using`s.

Pass `T` to match the problem's element type; `TRSolver` checks and raises on a
mismatch.

The operator enforces positive definiteness, so the model never reports
negative curvature — see [`reports_negative_curvature`](@ref). On a problem
whose true Hessian is indefinite along the trajectory this is invisible to every
first-order diagnostic: ρ stays healthy, ‖g‖ decreases, and the limit can still
fail second-order optimality.
"""
mutable struct LBFGSModel{T} <: ModelHessian
    mem::Int
    op::Union{Nothing, LBFGSOperator{T}}
end

LBFGSModel(; mem::Int = 5, T::Type = Float64) = LBFGSModel{T}(mem, nothing)

function reset_model!(m::LBFGSModel{T}, n::Int) where {T}
    m.op = LBFGSOperator(T, n, mem = m.mem)
    return nothing
end

hessian_op(m::LBFGSModel, nlp, x) = m.op

function update_model!(m::LBFGSModel, s, y)
    m.op === nothing && return nothing
    push!(m.op, s, y)          # LBFGSOperator applies its own curvature safeguard
    return nothing
end

dense_hessian(m::LBFGSModel, nlp, x) = Matrix(m.op)
reports_negative_curvature(::LBFGSModel) = false
model_eltype(::LBFGSModel{T}) where {T} = T
Base.show(io::IO, m::LBFGSModel{T}) where {T} = print(io, "L-BFGS(mem=", m.mem, ", T=", T, ")")

# -----------------------------------------------------------------------------
# SR1Model
# -----------------------------------------------------------------------------

"""
    SR1Model(; mem = 5, T = Float64)

Symmetric rank-one model Hessian, backed by `LinearOperators.LSR1Operator`.

Unlike L-BFGS, SR1 may be indefinite — which is the point: it can represent
negative curvature, so a subsolver that exploits it (truncated CG detecting
`dᵀHd ≤ 0`, or an exact solver handling the hard case) can escape a saddle
that a positive-definite model would converge to.
"""
mutable struct SR1Model{T} <: ModelHessian
    mem::Int
    op::Union{Nothing, LSR1Operator{T}}
end

SR1Model(; mem::Int = 5, T::Type = Float64) = SR1Model{T}(mem, nothing)

function reset_model!(m::SR1Model{T}, n::Int) where {T}
    m.op = LSR1Operator(T, n, mem = m.mem)
    return nothing
end

hessian_op(m::SR1Model, nlp, x) = m.op

function update_model!(m::SR1Model, s, y)
    m.op === nothing && return nothing
    push!(m.op, s, y)
    return nothing
end

dense_hessian(m::SR1Model, nlp, x) = Matrix(m.op)
reports_negative_curvature(::SR1Model) = true
model_eltype(::SR1Model{T}) where {T} = T
Base.show(io::IO, m::SR1Model{T}) where {T} = print(io, "SR1(mem=", m.mem, ", T=", T, ")")

# -----------------------------------------------------------------------------
# ScaledIdentity
# -----------------------------------------------------------------------------

"""
    ScaledIdentity(; c = 1.0)

`H_k = c·I`. Carries no curvature information whatsoever.

Diagnostic instrument rather than a practical model: with this model the
trust-region step is `-g/c` truncated to the region, so the method is exactly
gradient descent and any difference between radius rules is attributable to the
radius rule alone. Useful for isolating the Cauchy-point degeneration described
in the survey.
"""
struct ScaledIdentity <: ModelHessian
    c::Float64
    function ScaledIdentity(; c::Real = 1.0)
        # ArgumentError, not @assert: `check_factors` documents why, and this is
        # the same kind of check.
        c > 0 || throw(ArgumentError("ScaledIdentity: need c > 0, got $c"))
        new(float(c))
    end
end

hessian_op(m::ScaledIdentity, nlp, x) = m.c * I     # UniformScaling; `_apply_op!` has a method
dense_hessian(m::ScaledIdentity, nlp, x) =
    Matrix(m.c * I, nlp.meta.nvar, nlp.meta.nvar)
reports_negative_curvature(::ScaledIdentity) = false
Base.show(io::IO, m::ScaledIdentity) = print(io, m.c, "·I")

# -----------------------------------------------------------------------------
# SPDTarget
# -----------------------------------------------------------------------------

"""
    SPDTarget(; target, λ⊥ = 1.0)

Two-dimensional construction whose unconstrained model minimiser is pinned to
`target`, while `H_k ≻ 0`.

With `d = target - x`, `u = d/‖d‖`, `v = u^⊥`:

    H = a·uuᵀ + b·(uvᵀ + vuᵀ) + c·vvᵀ,
    a = -gᵀu/‖d‖,   b = -gᵀv/‖d‖,   c = b²/a + λ⊥.

This has minimiser `target` for *every* `c`, and is positive definite exactly
when `a > 0`, equivalently when

    φ(x) = gᵀ(target - x) < 0,

i.e. the target lies downhill. That condition is necessary for any SPD model
with that minimiser, so a `DomainError` here is not an artefact of the
construction: no such model exists at `x`.

Point a `SPDTarget` at a saddle to obtain a run in which every hypothesis of
the first-order theory holds, every ρ is successful, ‖g‖ → 0, and the limit is
nonetheless a saddle — the failure being the model, which no radius rule can
repair.

!!! note "This model evaluates the gradient"
    `dense_hessian` calls `grad(nlp, x)`, so `neval_grad` counts gradients the
    *algorithm* never asked for. Since evaluation counts are the benchmark's
    cost measure, subtract `model_grad_evals(solver)` before reporting, or run
    this model only in the diagnostic experiments it exists for.
"""
struct SPDTarget <: ModelHessian
    target::Vector{Float64}
    λ⊥::Float64
    function SPDTarget(; target::AbstractVector, λ⊥::Real = 1.0)
        length(target) == 2 || throw(ArgumentError(
            "SPDTarget is a two-dimensional construction, got length $(length(target))"))
        λ⊥ > 0 || throw(ArgumentError("SPDTarget: need λ⊥ > 0, got $λ⊥"))
        new(Vector{Float64}(target), float(λ⊥))
    end
end

"""
    phi_target(model, nlp, x) -> Float64

The descent functional `φ(x) = ∇f(x)ᵀ(target − x)`. The `SPDTarget`
construction exists at `x` if and only if `φ(x) < 0`.
"""
phi_target(m::SPDTarget, nlp, x) = dot(grad(nlp, x), m.target .- x)

function dense_hessian(m::SPDTarget, nlp, x)
    d  = m.target .- x
    nd = norm(d)
    nd < 1e-14 && return Matrix{Float64}(I, 2, 2)
    u = d ./ nd
    v = [-u[2], u[1]]
    g = grad(nlp, x)
    a = -dot(g, u) / nd
    b = -dot(g, v) / nd
    a <= 0 && throw(DomainError(a,
        "no SPD model has its minimiser at the target here: φ(x) ≥ 0"))
    c = b * b / a + m.λ⊥
    return a * (u * u') + b * (u * v' + v * u') + c * (v * v')
end

hessian_op(m::SPDTarget, nlp, x) = dense_hessian(m, nlp, x)
reports_negative_curvature(::SPDTarget) = false
Base.show(io::IO, m::SPDTarget) = print(io, "SPD→", m.target)

# -----------------------------------------------------------------------------
# model_hprod!
# -----------------------------------------------------------------------------

"""
    model_hprod!(model, nlp, x, v, Hv)

Write `H_k · v` into `Hv`, where `H_k` is the **model** Hessian.

Used for the predicted reduction `-gᵀs - ½ sᵀH_k s`. Using `∇²f` here instead
would make `ρ` measure agreement between the true function and a model the
algorithm never minimised, so `ρ` would stop being the quantity the acceptance
test and the radius rules are stated in terms of.

Subsolvers that already form `B·s` while computing the step declare
`returns_hprod`, and the solver then skips this call entirely — see
`Subproblem/subproblem.jl`.
"""
model_hprod!(::ExactHessian, nlp, x, v, Hv) = hprod!(nlp, x, v, Hv)

function model_hprod!(model::ModelHessian, nlp, x, v, Hv)
    B = hessian_op(model, nlp, x)
    return _apply_op!(Hv, B, v)
end
