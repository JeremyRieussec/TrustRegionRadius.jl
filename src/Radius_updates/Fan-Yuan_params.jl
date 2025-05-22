


"""
    YuanFanParameters <: AbstractTrustRegionParameters

Parameters for the Yuan-Fan trust region update strategy.

# Fields
- `η₁::Float64`: Acceptance threshold for step acceptance (default: 0.1).
- `η₂::Float64`: Threshold for considering a step as "good" (default: 0.75).
- `γ₁::Float64`: Factor by which the trust region radius is decreased after a poor step (default: 0.5).
- `γ₂::Float64`: Factor by which the trust region radius is increased after a good step (default: 2.0).
- `μ::Float64`: Multiplier for the initial trust region radius (default: 0.1).

# Constructor
    YuanFanParameters([η₁, η₂, γ₁, γ₂, μ])

Creates a `YuanFanParameters` instance with optional custom parameter values.
"""
mutable struct YuanFanParameters <: AbstractTrustRegionParameters
    η₁::Float64  # Acceptance threshold
    η₂::Float64  # Good step threshold
    γ₁::Float64  # Radius decrease factor
    γ₂::Float64  # Radius increase factor
    μ::Float64   # Initial radius multiplier

    function YuanFanParameters(η₁::Float64 = 0.1, η₂::Float64 = 0.75, 
                               γ₁::Float64 = 0.5, γ₂::Float64 = 2.0, 
                               μ::Float64 = 0.1)
        new(η₁, η₂, γ₁, γ₂, μ)
    end
end


function trust_region_update!(state::TrustRegionState, params::YuanFanParameters, algoInfo::AlgorithmInfoTR, nlp::AbstractNLPModel)

    # Update step
    accept_or_reject_step!(state, params, algoInfo, nlp)

    if state.ρ > params.η₂
        params.μ *= params.γ₂
    elseif state.ρ < params.η₁
        params.μ *= params.γ₁
    end
    state.Δ = params.μ * norm(state.gradient)
    push!(algoInfo.mus, params.μ)
    push!(algoInfo.radii, state.Δ)
end


function Base.show(io::IO, params::YuanFanParameters)
    println(io, "Yuan-Fan Parameters:")
    println(io, "  η₁: ", params.η₁)
    println(io, "  η₂: ", params.η₂)
    println(io, "  γ₁: ", params.γ₁)
    println(io, "  γ₂: ", params.γ₂)
    println(io, "  μ: ", params.μ)
end