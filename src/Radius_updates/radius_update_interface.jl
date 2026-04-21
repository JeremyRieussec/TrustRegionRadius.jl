
# ============================================================
# radius_update_interface.jl
#
# reset_rule! interface for AbstractRadiusUpdate.
# Mutable rules (R4RelativeGradUpdate, HeiFanYuanUpdate) store
# μ₀ so their state can be restored between solve! calls.
# ============================================================

"""
    reset_rule!(rule::AbstractRadiusUpdate)

Reset any mutable state in `rule` to its initial value.

The default implementation is a no-op (for immutable rules such as
R1ClassicalUpdate, R2StepSizeUpdate, R3DFOLikeUpdate, HeiUpdate,
HeiGradUpdate).  Mutable rules (R4RelativeGradUpdate, HeiFanYuanUpdate)
override this to restore μ ← μ₀.
"""
reset_rule!(::AbstractRadiusUpdate) = nothing

"""
    reset_rule!(rule::R4RelativeGradUpdate)

Restore the multiplier μ to its construction value μ₀.
"""
function reset_rule!(rule::R4RelativeGradUpdate)
    rule.μ = rule.μ₀
    return nothing
end

"""
    reset_rule!(rule::HeiFanYuanUpdate)

Restore the multiplier μ to its construction value μ₀.
"""
function reset_rule!(rule::HeiFanYuanUpdate)
    rule.μ = rule.μ₀
    return nothing
end
