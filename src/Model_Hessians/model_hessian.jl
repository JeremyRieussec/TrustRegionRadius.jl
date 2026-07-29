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
#     dense_hessian(m, nlp, x)        -- dense Matrix, small n only (diagnostics)
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

Return an object `B` supporting `B * v`, representing `H_k` at `x`.
"""
function hessian_op end

"""
    dense_hessian(model, nlp, x) -> Matrix

Dense representation, for diagnostics on small problems only.
"""
function dense_hessian end

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
dense_hessian(::ExactHessian, nlp, x) = Matrix(Symmetric(hess(nlp, x), :L))
Base.show(io::IO, ::ExactHessian) = print(io, "exact ∇²f")

# -----------------------------------------------------------------------------
# LBFGSModel
# -----------------------------------------------------------------------------

"""
    LBFGSModel(; mem = 5)

Limited-memory BFGS model Hessian, backed by `LinearOperators.LBFGSOperator`.

The operator enforces positive definiteness, so the model never reports
negative curvature. On a problem whose true Hessian is indefinite along the
trajectory this is invisible to every first-order diagnostic: ρ stays healthy,
‖g‖ decreases, and the limit can still fail second-order optimality.
"""
mutable struct LBFGSModel <: ModelHessian
    mem::Int
    op::Any          # LBFGSOperator{T}; Any avoids a hard type dependency here
    LBFGSModel(; mem::Int = 5) = new(mem, nothing)
end

function reset_model!(m::LBFGSModel, n::Int)
    m.op = LBFGSOperator(Float64, n, mem = m.mem)
    return nothing
end

hessian_op(m::LBFGSModel, nlp, x) = m.op

function update_model!(m::LBFGSModel, s, y)
    m.op === nothing && return nothing
    push!(m.op, s, y)          # LBFGSOperator applies its own curvature safeguard
    return nothing
end

dense_hessian(m::LBFGSModel, nlp, x) = Matrix(m.op)
Base.show(io::IO, m::LBFGSModel) = print(io, "L-BFGS(mem=", m.mem, ")")

# -----------------------------------------------------------------------------
# SR1Model
# -----------------------------------------------------------------------------

"""
    SR1Model(; mem = 5)

Symmetric rank-one model Hessian, backed by `LinearOperators.LSR1Operator`.

Unlike L-BFGS, SR1 may be indefinite — which is the point: it can represent
negative curvature, so a subsolver that exploits it (truncated CG detecting
`dᵀHd ≤ 0`, or an exact solver handling the hard case) can escape a saddle
that a positive-definite model would converge to.
"""
mutable struct SR1Model <: ModelHessian
    mem::Int
    op::Any
    SR1Model(; mem::Int = 5) = new(mem, nothing)
end

function reset_model!(m::SR1Model, n::Int)
    m.op = LSR1Operator(Float64, n, mem = m.mem)
    return nothing
end

hessian_op(m::SR1Model, nlp, x) = m.op

function update_model!(m::SR1Model, s, y)
    m.op === nothing && return nothing
    push!(m.op, s, y)
    return nothing
end

dense_hessian(m::SR1Model, nlp, x) = Matrix(m.op)
Base.show(io::IO, m::SR1Model) = print(io, "SR1(mem=", m.mem, ")")

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
    ScaledIdentity(; c::Float64 = 1.0) = (@assert c > 0 "ScaledIdentity: need c > 0"; new(c))
end

hessian_op(m::ScaledIdentity, nlp, x) = m.c * I     # UniformScaling: `B * v` works
dense_hessian(m::ScaledIdentity, nlp, x) =
    Matrix(m.c * I, nlp.meta.nvar, nlp.meta.nvar)
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
"""
struct SPDTarget <: ModelHessian
    target::Vector{Float64}
    λ⊥::Float64
    function SPDTarget(; target::Vector{Float64}, λ⊥::Float64 = 1.0)
        @assert length(target) == 2 "SPDTarget is a two-dimensional construction"
        @assert λ⊥ > 0              "SPDTarget: need λ⊥ > 0"
        new(copy(target), λ⊥)
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
"""
model_hprod!(::ExactHessian, nlp, x, v, Hv) = hprod!(nlp, x, v, Hv)

function model_hprod!(model::ModelHessian, nlp, x, v, Hv)
    B = hessian_op(model, nlp, x)
    # `mul!` is not defined for every operator/eltype pair (a `UniformScaling`
    # supports `*` but not always the three-argument form), so fall back.
    try
        mul!(Hv, B, v)
    catch
        copyto!(Hv, B * v)
    end
    return Hv
end
