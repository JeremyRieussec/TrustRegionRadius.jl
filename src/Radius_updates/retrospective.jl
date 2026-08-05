# =============================================================================
# src/Radius_updates/retrospective.jl
#
# The retrospective ratio ρ̃.
#
# The classical ratio measures how well m_k predicted f at x_{k+1}:
#
#     ρ_k  = [f(x_k) − f(x_k + s_k)] / [m_k(x_k) − m_k(x_k + s_k)]
#
# The retrospective ratio measures how well the NEW model m_{k+1} would have
# predicted the same change, i.e. it judges the model that will actually be
# minimised inside the next trust region:
#
#     ρ̃_{k+1} = [f(x_k) − f(x_{k+1})] / [m_{k+1}(x_k) − m_{k+1}(x_{k+1})]
#
# Expanding m_{k+1} about x_{k+1} and using first-order coherence,
#
#     m_{k+1}(x_k) − m_{k+1}(x_{k+1}) = −g_{k+1}ᵀs_k + ½ s_kᵀ H_{k+1} s_k
#
# so one extra Hessian-vector product with the *new* model is needed. It is
# computed only when the active rule declares `needs_retrospective`, so rules
# that do not use ρ̃ pay nothing.
# =============================================================================

"""
    retrospective_ratio(actual, s, g_new, Hnew_s) -> Float64

Retrospective ratio ρ̃, from

- `actual`: `f(x_k) − f(x_{k+1})`, the achieved reduction (same numerator as ρ);
- `s`:      the step `s_k`;
- `g_new`:  `g_{k+1}`, the gradient at the new iterate;
- `Hnew_s`: `H_{k+1} · s_k`, the *new* model Hessian applied to the step.

Returns `-Inf` when the retrospective predicted reduction is non-positive,
which sends the rule to its most conservative branch — the same convention the
solver uses for ρ.

The denominator is `−g_{k+1}ᵀs_k + ½ s_kᵀH_{k+1}s_k`. Note the sign of the
first term: it is a reduction measured *backwards* along the step. Near
convergence `g_{k+1}ᵀs_k → 0` — the new gradient is asymptotically orthogonal to
the step that produced it — so it is the curvature term `½s_kᵀH_{k+1}s_k` that
carries the denominator, and positivity of ρ̃ in the limit is a statement about
the model Hessian along `s_k`, not about the gradient. That is why `ρ̃ → 1`
requires either the secant condition or asymptotic second-order coherence, and
is not implied by first-order coherence alone.
"""
@inline function retrospective_ratio(actual::T, s::V, g_new::V, Hnew_s::V) where {T <: Real, V}
    predicted = -dot(g_new, s) + T(0.5) * dot(s, Hnew_s)
    return (isfinite(actual) && predicted > 0) ? actual / predicted : T(-Inf)
end

"""
    retrospective_requires_new_model(rule) -> Bool

Whether the solver must evaluate the model Hessian at the *new* iterate before
updating the radius. Identical to [`needs_retrospective`](@ref); provided as a
separate name because the cost it implies is a solver concern rather than a
property of the rule.

The cost is one Hessian-vector product per accepted iteration. With a
quasi-Newton model that is essentially free (the operator is already updated);
with the exact Hessian it is a genuine extra `hprod`.
"""
retrospective_requires_new_model(rule::RadiusRule) = needs_retrospective(rule)
