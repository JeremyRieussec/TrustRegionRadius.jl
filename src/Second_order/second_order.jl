# =============================================================================
# src/Second_order/second_order.jl
#
# The second-order layer: one criticality measure, one rule wrapper, one
# curvature estimate, one subsolver wrapper.
#
#   τ(x) = max{ ‖g(x)‖, −λ_min(B(x)) }
#
# τ vanishes exactly at a second-order critical point of the model, whereas ‖g‖
# vanishes at every critical point including saddles and maxima. Replacing ‖g‖
# by τ in a criticality-anchored rule is the whole of the second-order upgrade
# of Part II §4: the update *logic* does not change at all, only the quantity the
# radius is measured against.
#
# That is why `SecondOrder` is a wrapper rather than a family of new rules. It
# forwards `update_radius!` to the rule it wraps, unchanged, and overrides only
# `criticality`. The solver asks the rule which measure it wants, computes it,
# and passes it in the slots that carried ‖g_k‖ and ‖g_{k+1}‖ before — so no
# rule signature changes and every criticality-anchored mechanism acquires a
# second-order variant for free.
#
# Contents
#   criticality, needs_curvature      the interface the solver dispatches on
#   SecondOrder, RGradTau, RDFOTau…   the wrapper and its aliases
#   lambda_min_estimate               dense eigendecomposition or Lanczos
#   curvature_estimate                the type-stable form the solver calls
#   EigenPoint                        guarantees negative curvature is exploited
# =============================================================================

# -----------------------------------------------------------------------------
# The criticality interface
# -----------------------------------------------------------------------------

"""
    tau_criticality(g_norm, λmin) -> Float64

The second-order criticality measure

```math
τ = \\max\\{\\,\\|g\\|,\\; -\\lambda_{\\min}(B)\\,\\} = \\max\\{\\|g\\|, -\\lambda^-\\},
\\qquad \\lambda^- = \\min\\{\\lambda_{\\min}(B),\\,0\\}.
```

`τ = 0` exactly when `g = 0` and `B ⪰ 0`, i.e. at a second-order critical point of
the model. Positive curvature never enlarges it: `λmin > 0` contributes `-λ^- = 0`
and the measure falls back to `‖g‖`.
"""
@inline tau_criticality(g_norm::Real, λmin::Real) =
    max(float(g_norm), -min(float(λmin), 0.0))

"""
    criticality(rule, g_norm, λmin) -> Float64

The quantity `rule` measures its radius against. `‖g‖` by default; `τ` for a rule
wrapped in [`SecondOrder`](@ref).

The solver evaluates this once per iteration and passes the result to
[`initial_radius`](@ref) and [`update_radius!`](@ref) in place of the gradient
norm. A rule that ignores criticality altogether — `RDelta`, `RStep` — never sees
the difference.
"""
criticality(::RadiusRule, g_norm::Real, ::Real) = float(g_norm)

"""
    needs_curvature(rule) -> Bool

Whether `rule` needs `λ_min(B_k)`, and so whether the solver should pay for a
curvature estimate at every iteration.

`false` by default. The estimate costs a dense symmetric eigendecomposition on
small problems and a short Lanczos run otherwise, so it is not levied unless a
rule, the second-order stopping test, or [`EigenPoint`](@ref) asks for it.
"""
needs_curvature(::RadiusRule) = false

# -----------------------------------------------------------------------------
# The wrapper
# -----------------------------------------------------------------------------

"""
    SecondOrder(rule)

The second-order variant of a criticality-anchored `rule`: identical update logic,
with the radius anchored to `τ = max{‖g‖, −λ_min(B)}` instead of `‖g‖`.

```julia
SecondOrder(RGrad(μ = 1.0))        # Δ_k = μ_k τ_k, μ updated exactly as before
SecondOrder(RDFO(ζ = 1.0))         # expands while Δ_k ≤ ζ τ_k
```

Only rules that actually read the criticality measure may be wrapped; wrapping
`RDelta` or `RStep` would change nothing, since they never consult it, so it
raises `ArgumentError` rather than producing a silent no-op.

# What the upgrade buys

At a first-order critical point that is not a minimiser, `‖g‖ = 0` while
`λ_min < 0`. A rule anchored to `‖g‖` therefore reports the radius it would use
at a solution, and the two anchored families fail in different ways:

- `RGrad(‖g‖)` sets `Δ_k = μ_k‖g_k‖ = 0` and **halts outright** — every subsequent
  step is zero, ρ is `NaN`, and no descent direction can be taken however much
  negative curvature the model reports;
- `RDFO(‖g‖)` finds `Δ_k > ζ‖g_k‖ = 0` at every iteration and **contracts
  geometrically**, so the radius reaches the stagnation floor in `O(log(1/eps))`
  iterations.

Under τ both are repaired: `τ_k = -λ_min > 0` near such a point, the radius stays
proportional to the curvature available, and a subsolver that exploits negative
curvature escapes. This is the sharpest experiment in the second-order half of the
survey, and it is two lines to run — see the docstring of [`EigenPoint`](@ref) for
the other half of what makes it work.

!!! warning "τ ≡ ‖g‖ for a positive definite model"
    `LBFGSModel` enforces `B ≻ 0`, so `λ_min > 0` always, so `τ = ‖g‖` identically
    and `SecondOrder` is an expensive no-op. The same holds for `ScaledIdentity`,
    `SPDTarget` and the outer-product models. A second-order variant is only
    meaningful over a model that can report negative curvature — `ExactHessian`
    or `SR1Model`.

    This is now enforced rather than described: `reports_negative_curvature` is a
    trait on `ModelHessian`, and `TRSolver` warns once at construction when a
    τ-anchored rule or `tol_H > 0` meets a model that cannot supply the
    information. The previous docstring claimed the solver warned; nothing did.

# Fields

The wrapped rule's fields are forwarded, so `SecondOrder(RGrad(μ = 2.0)).μ` works
and `r.inner` reaches the rule itself.
"""
struct SecondOrder{R <: RadiusRule} <: RadiusRule
    inner::R
    function SecondOrder(inner::R) where {R <: RadiusRule}
        is_criticality_anchored(inner) || throw(ArgumentError(
            "SecondOrder($(nameof(R))): only a criticality-anchored rule can be " *
            "upgraded, since no other rule reads the criticality measure. " *
            "$(nameof(R)) is $(asymptotic_regime(inner)); wrapping it would " *
            "change nothing."))
        new{R}(inner)
    end
end

criticality(::SecondOrder, g_norm::Real, λmin::Real) = tau_criticality(g_norm, λmin)
needs_curvature(::SecondOrder) = true

# Everything else is the wrapped rule's, verbatim. `update_radius!` in particular
# is *not* redefined: the second-order variant differs from the first-order one
# only in the value the solver puts in the criticality slots.
initial_radius(r::SecondOrder, Δ0::Float64, crit::Float64) =
    initial_radius(r.inner, Δ0, crit)

update_radius!(r::SecondOrder, Δ::Float64, ρ::Float64, accepted::Bool,
               η1::Float64, η2::Float64, s_norm::Float64,
               crit_old::Float64, crit_new::Float64) =
    update_radius!(r.inner, Δ, ρ, accepted, η1, η2, s_norm, crit_old, crit_new)

reset_rule!(r::SecondOrder)             = reset_rule!(r.inner)
needs_retrospective(r::SecondOrder)     = needs_retrospective(r.inner)
is_criticality_anchored(::SecondOrder)  = true
asymptotic_regime(r::SecondOrder)       = asymptotic_regime(r.inner)
validate_thresholds(r::SecondOrder, η::Real, η1::Real, η2::Real) =
    validate_thresholds(r.inner, η, η1, η2)

# Field forwarding, so a wrapped rule behaves like the rule it wraps.
function Base.getproperty(r::SecondOrder, s::Symbol)
    s === :inner && return getfield(r, :inner)
    return getproperty(getfield(r, :inner), s)
end
function Base.setproperty!(r::SecondOrder, s::Symbol, v)
    s === :inner && throw(ArgumentError("SecondOrder: `inner` is immutable"))
    return setproperty!(getfield(r, :inner), s, v)
end
Base.propertynames(r::SecondOrder) =
    (:inner, propertynames(getfield(r, :inner))...)

Base.show(io::IO, r::SecondOrder) = print(io, "SecondOrder(", getfield(r, :inner), ")")
function Base.show(io::IO, ::MIME"text/plain", r::SecondOrder)
    println(io, "SecondOrder — radius anchored to τ = max{‖g‖, −λ_min(B)}")
    show(io, MIME"text/plain"(), getfield(r, :inner))
end

# -----------------------------------------------------------------------------
# Aliases
# -----------------------------------------------------------------------------

"""
    RGradTau(; μ = 1.0, γ1, γ2, γ3, half_test)

`Δ_k = μ_k τ_k` with `μ` updated exactly as in [`RGrad`](@ref): the four branches
of Update (R-grad), the `‖s_k‖ > ½Δ_k` guard on expansion, no cap.

Shorthand for `SecondOrder(RGrad(...))`. Since `μ_k` is now the ratio
`Δ_k/τ_k` rather than `Δ_k/‖g_k‖`, the geometric search it performs is a search
for the *second-order* inactivity threshold.
"""
RGradTau(; kwargs...) = SecondOrder(RGrad(; kwargs...))

"""
    RGradCappedTau(; μ, μ_max, γ1, γ2, γ3, half_test)

[`RGradCapped`](@ref) anchored to τ. The cap carries the same warning as in the
first-order case, now against the second-order threshold: `μ_max` below it leaves
the trust region binding for ever.
"""
RGradCappedTau(; kwargs...) = SecondOrder(RGradCapped(; kwargs...))

"""
    RDFOTau(; ζ = 1.0, γ1, γ2, γ3, Δmin, Δmax)

[`RDFO`](@ref) anchored to τ: contract on an unsuccessful iteration, otherwise
shrink when `Δ_k > ζ τ_k` and expand when `Δ_k ≤ ζ τ_k`.
"""
RDFOTau(; kwargs...) = SecondOrder(RDFO(; kwargs...))

# The remaining criticality-anchored mechanisms under τ, for completeness of the
# rule × measure grid. The two that carry the survey's second-order results are
# `RGradTau` and `RDFOTau`.

"""
    RAdaptiveGradTau(; kwargs...)

[`RAdaptiveGrad`](@ref) anchored to τ.

The multiplier **accumulates** (`μ_{k+1} = μ_k R(ρ_k)`), so it is unbounded above
and inherits `RGrad`'s unconditional eventual inactivity. 
"""
RAdaptiveGradTau(; kwargs...) = SecondOrder(RAdaptiveGrad(; kwargs...))


"""
    RRTRGradTau(; kwargs...)

[`RRTRGrad`](@ref) anchored to τ: the retrospective ratio ρ̃ drives the multiplier,
and τ sets the scale it multiplies.
"""
RRTRGradTau(; kwargs...) = SecondOrder(RRTRGrad(; kwargs...))

# -----------------------------------------------------------------------------
# Curvature estimation
# -----------------------------------------------------------------------------

"""
    lambda_min_estimate(model, nlp, x; nmax = 200, lanczos_k = 40, vector = false)

Smallest eigenvalue of the model Hessian `B(x)`, returned as a `Float64`, or the
pair `(λ, v)` when `vector = true`.

Two regimes:

- `n ≤ nmax`: a dense symmetric eigendecomposition of `dense_hessian`. Exact.
- `n > nmax`: `lanczos_k` steps of Lanczos with full reorthogonalisation against
  `hessian_op`, returning the smallest Ritz value and its Ritz vector.

Prefer [`curvature_estimate`](@ref) inside the solver: it has a single return
type, whereas this returns either a scalar or a tuple depending on a keyword.

!!! note "The Lanczos value is an upper bound"
    A Ritz value satisfies `λ_Ritz ≥ λ_min`, so an under-resolved run *understates*
    the available negative curvature. In `τ = max{‖g‖, −λ_min}` that biases the
    measure downward and makes the second-order test optimistic — it can report
    second-order criticality at a saddle whose negative direction Lanczos has not
    yet found. Raise `lanczos_k` when the second-order status matters; the
    estimate is exact in the dense branch.
"""
function lambda_min_estimate(model::ModelHessian, nlp, x;
                             nmax::Int = 200, lanczos_k::Int = 40,
                             vector::Bool = false)
    n = length(x)
    if n <= nmax
        B = Symmetric(Matrix(dense_hessian(model, nlp, x)))
        if vector
            F = eigen(B)
            return Float64(F.values[1]), Vector{Float64}(F.vectors[:, 1])
        end
        return Float64(eigvals(B)[1])
    end
    return _lanczos_min(hessian_op(model, nlp, x), n; k = lanczos_k, vector = vector)
end

"""
    curvature_estimate(model, nlp, x, want_vector; nmax, lanczos_k)
        -> (λ::Float64, v::Vector{Float64})

Type-stable wrapper around [`lambda_min_estimate`](@ref): always returns a pair,
with an empty `v` when the eigenvector was not requested.

This is what the solver calls, once per iteration, and what it hands to
`solve_subproblem!` through the `curv` keyword. `EigenPoint` previously computed
its own estimate on top of the solver's, so a τ-anchored run with
`EigenPoint(SteihaugCG())` paid for two dense eigendecompositions per iteration.
"""
function curvature_estimate(model::ModelHessian, nlp, x, want_vector::Bool;
                            nmax::Int = 200, lanczos_k::Int = 40)
    if want_vector
        λ, v = lambda_min_estimate(model, nlp, x; nmax = nmax,
                                   lanczos_k = lanczos_k, vector = true)
        return Float64(λ), Vector{Float64}(v)
    end
    λ = lambda_min_estimate(model, nlp, x; nmax = nmax,
                            lanczos_k = lanczos_k, vector = false)
    return Float64(λ), Float64[]
end

"""
    _lanczos_min(B, n; k, vector) -> λ  or  (λ, v)

Smallest Ritz value of `B` from `k` Lanczos steps with full reorthogonalisation.

The starting vector is deterministic, so repeated runs of an experiment agree.
Full reorthogonalisation costs `O(k²n)` and is worth it here: `k` is small, and a
loss of orthogonality would show up directly as a spurious `τ`.
"""
function _lanczos_min(B, n::Int; k::Int = 40, vector::Bool = false)
    m = min(k, n)
    Q = zeros(Float64, n, m)
    α = zeros(Float64, m)
    β = zeros(Float64, max(m - 1, 0))

    q = [sin(1.7 * i) for i in 1:n]          # deterministic, generically not
    q ./= norm(q)                            # orthogonal to any eigenvector
    q_prev = zeros(Float64, n)
    j_last = m

    for j in 1:m
        @views Q[:, j] .= q
        z = Vector{Float64}(B * q)
        α[j] = dot(q, z)
        @. z -= α[j] * q
        j > 1 && (@. z -= β[j - 1] * q_prev)
        for _ in 1:2                          # full reorthogonalisation, twice
            for i in 1:j
                @views z .-= dot(Q[:, i], z) .* Q[:, i]
            end
        end
        nz = norm(z)
        if j == m || nz <= 1e-12 * max(1.0, abs(α[j]))
            j_last = j; break
        end
        β[j] = nz
        q_prev = q
        q = z ./ nz
    end

    Tm = SymTridiagonal(α[1:j_last], β[1:max(j_last - 1, 0)])
    if vector
        F = eigen(Tm)
        return F.values[1], Q[:, 1:j_last] * F.vectors[:, 1]
    end
    return eigvals(Tm)[1]
end

# -----------------------------------------------------------------------------
# The eigenpoint
# -----------------------------------------------------------------------------

"""
    EigenPoint(inner = SteihaugCG(); nmax = 200, lanczos_k = 40)

Wraps a subproblem solver so that the returned step always achieves at least the
model decrease available along the leftmost eigenvector.

When `λ_min(B) < 0` this computes the **eigenpoint**

```math
d = \\pm\\Delta\\, v_{\\min}, \\qquad
m(0) - m(d) \\;\\ge\\; \\tfrac12 |\\lambda_{\\min}|\\,\\Delta^2,
```

with the sign chosen so that `gᵀd ≤ 0`, and returns whichever of `d` and the inner
solver's step decreases the model more. When `λ_min ≥ 0` the inner step is
returned untouched and nothing is spent beyond the curvature estimate.

The estimate is taken from the `curv` argument when the solver has already
computed it (which it has whenever the rule is τ-anchored or `tol_H > 0`), and
computed here only otherwise. `needs_eigenvector` tells the solver to ask for the
eigenvector as well, so the shared estimate is the one with `v`.

# Why this is needed

τ-anchoring keeps the radius *positive* near a saddle; it does not make the step
*go anywhere*. The two halves are independent, and both are required:

- [`SteihaugCG`](@ref) terminates on the first direction of negative curvature and
  runs to the boundary along it. That is a genuine decrease, but the direction is
  whichever the CG recurrence happened to reach, and the guaranteed decrease
  carries no `|λ_min|Δ²` term — the fraction of the eigenpoint decrease it
  achieves can be arbitrarily small.
- [`ExactMS`](@ref) already solves the subproblem exactly, hard case included, so
  it satisfies the condition on its own and needs no wrapping. Wrapping it is
  harmless and now costs nothing extra when the solver is already estimating the
  curvature.

So `EigenPoint(SteihaugCG())` is the combination to use for second-order runs on
anything but the smallest problems, and it is what makes the `O(max{ε_g^{-2},
ε_H^{-3}})` complexity bound of Part II attainable in practice rather than only in
principle.
"""
struct EigenPoint{S <: SubproblemSolver} <: SubproblemSolver
    inner::S
    nmax::Int
    lanczos_k::Int
end

EigenPoint(inner::SubproblemSolver = SteihaugCG();
           nmax::Int = 200, lanczos_k::Int = 40) =
    EigenPoint(inner, nmax, lanczos_k)

needs_eigenvector(::EigenPoint) = true
# Always true: even when the inner solver does not supply B·s, this wrapper needs
# it to compare the two candidate decreases, so it can always leave it behind.
returns_hprod(::EigenPoint) = true

function solve_subproblem!(sub::EigenPoint, model::ModelHessian,
                           nlp::AbstractNLPModel{T, V},
                           x::V, g::V, Δ::T, s::V, Hs::V,
                           ws::SubWorkspace{V}; curv = nothing) where {T, V}
    active = solve_subproblem!(sub.inner, model, nlp, x, g, Δ, s, Hs, ws; curv = curv)

    B = hessian_op(model, nlp, x)
    # The inner solver may or may not have left B·s behind; make sure it is there.
    returns_hprod(sub.inner) || _apply_op!(Hs, B, s)

    # Reuse the solver's estimate when it supplied one with an eigenvector.
    λ, v = if curv !== nothing && !isempty(curv[2])
        curv
    else
        curvature_estimate(model, nlp, x, true; nmax = sub.nmax,
                           lanczos_k = sub.lanczos_k)
    end
    λ >= 0 && return active                       # no negative curvature to exploit

    # Model decrease of the inner step, m(0) − m(s) = −gᵀs − ½ sᵀBs.
    dec_inner = -dot(g, s) - T(0.5) * dot(s, Hs)

    # The eigenpoint, signed so that gᵀd ≤ 0 and scaled to the boundary.
    d, Hd = ws.cand, ws.Hd            # the inner solver is done with both
    copyto!(d, v)
    nv = norm(d)
    nv == 0 && return active
    @. d *= Δ / nv
    dot(g, d) > 0 && (@. d = -d)
    _apply_op!(Hd, B, d)
    dec_eigen = -dot(g, d) - T(0.5) * dot(d, Hd)

    if dec_eigen > dec_inner
        copyto!(s, d)
        copyto!(Hs, Hd)                           # keep the B·s contract
        return true                               # the eigenpoint is on the boundary
    end
    return active
end

# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

"""
    second_order_status(g_norm, λmin, tol_g, tol_H) -> Symbol

`:second_order` when `‖g‖ ≤ tol_g` and `λ_min ≥ −tol_H`, `:first_order` when only
the gradient test passes, `:unknown` otherwise.

Kept separate from the solver loop so that a trace can be re-classified after the
fact under a different `tol_H` without re-running anything.
"""
function second_order_status(g_norm::Real, λmin::Real, tol_g::Real, tol_H::Real)
    g_norm <= tol_g || return :unknown
    return λmin >= -tol_H ? :second_order : :first_order
end
