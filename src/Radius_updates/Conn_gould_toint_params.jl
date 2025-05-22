
"""
    SimpleTointGouldTointParameters

Parameters for the simple Toint-Gould trust-region update strategy.

# Fields
- `η₁::Float64`: Acceptance threshold for step quality (default: 0.1).
- `γ₁::Float64`: Factor to decrease the trust-region radius after a poor step (default: 0.5).
- `γ₂::Float64`: Factor to increase the trust-region radius after a good step (default: 2.0).

- `Δ::Float64`: Initial trust-region radius (default: 0.1).
# Description
This struct encapsulates the parameters used in the simple Toint-Gould trust-region update rules, which control how the trust-region radius is adjusted based on the quality of optimization steps.
"""
struct SimpleTointGouldTointParameters <: AbstractTrustRegionParameters
    η₁::Float64  # Acceptance threshold
    
    γ₁::Float64  # Radius decrease factor
    γ₂::Float64  # Radius increase factor

    Δ::Float64  # Initial trust-region radius

    function SimpleTointGouldTointParameters(η₁::Float64 = 0.1,  
                                             γ₁::Float64 = 0.5, γ₂::Float64 = 2.0, 
                                             Δ::Float64 = 0.1)
        new(η₁, γ₁, γ₂, Δ)
    end
end


function Base.show(io::IO, params::SimpleTointGouldTointParameters)
    println(io, "Simple Toint-Gould-Toint Parameters:")
    println(io, "  η₁: ", params.η₁)
    println(io, "  γ₁: ", params.γ₁)
    println(io, "  γ₂: ", params.γ₂)
    println(io, "  Δ: ", params.Δ)
end

function trust_region_update!(state::TrustRegionState, params::SimpleTointGouldTointParameters, algoInfo::AlgorithmInfoTR, nlp::AbstractNLPModel)

    # Update step
    accept_or_reject_step!(state, params, algoInfo, nlp)
    
    # Update trust-region radius based on the acceptance ratio ρ
    if state.ρ > params.η₁
        state.Δ *= params.γ₂ 
    else
        state.Δ *= params.γ₁
    end
    push!(algoInfo.radii, state.Δ)
end

"""
    TointGouldTointParameters

Parameters for the Toint-Gould trust-region update strategy.

# Fields
- `η₁::Float64`: Acceptance threshold for step quality (default: 0.1).
- `η₂::Float64`: Threshold for considering a step as "good" (default: 0.75).
- `γ₁::Float64`: Factor to decrease the trust-region radius after a poor step (default: 0.5).
- `γ₂::Float64`: Factor to increase the trust-region radius after a good step (default: 0.9).
- `γ₃::Float64`: Additional adjustment factor for the trust-region radius (default: 2.0).
- `Δ::Float64`: Initial trust-region radius (default: 0.1).

# Description
This struct encapsulates the parameters used in the Toint-Gould trust-region update rules, which control how the trust-region radius is adjusted based on the quality of optimization steps.
"""
struct TointGouldTointParameters <: AbstractTrustRegionParameters
    η₁::Float64  # Acceptance threshold
    η₂::Float64  # Good step threshold
    
    γ₁::Float64  # Radius decrease factor
    γ₂::Float64  # Radius increase factor
    γ₃::Float64  # Additional radius adjustment factor

    Δ::Float64  # Initial trust-region radius

    function TointGouldTointParameters(η₁::Float64 = 0.1, η₂::Float64 = 0.75, 
                                       γ₁::Float64 = 0.5, γ₂::Float64 = 0.9, γ₃::Float64 = 2.0, 
                                       Δ::Float64 = 0.1)
        new(η₁, η₂, γ₁, γ₂, γ₃, Δ)
    end
end

function trust_region_update!(state::TrustRegionState, params::TointGouldTointParameters, algoInfo::AlgorithmInfoTR, nlp::AbstractNLPModel)

    # Update step
    accept_or_reject_step!(state, params, algoInfo, nlp)
    
    # Update trust-region radius based on the acceptance ratio ρ
    if state.ρ > params.η₂
        state.Δ *= params.γ₃ 
    elseif state.ρ < params.η₁
        state.Δ *= params.γ₁
    else
        state.Δ *= params.γ₂
    end
    push!(algoInfo.radii, state.Δ)
end

function Base.show(io::IO, params::TointGouldTointParameters)
    println(io, "Toint-Gould-Toint Parameters:")
    println(io, "  η₁: ", params.η₁)
    println(io, "  η₂: ", params.η₂)
    println(io, "  γ₁: ", params.γ₁)
    println(io, "  γ₂: ", params.γ₂)
    println(io, "  γ₃: ", params.γ₃)
    println(io, "  Δ: ", params.Δ)
end