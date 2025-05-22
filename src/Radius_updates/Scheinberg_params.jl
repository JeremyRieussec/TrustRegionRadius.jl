
"""
    ScheinbergParameters

Parameters for the Scheinberg trust-region update strategy.
# Fields
- `η₁::Float64`: Acceptance threshold for step quality (default: 0.1).
- `ζ::Float64`: Gradient norm threshold for radius update (default: 0.75).
- `γ₁::Float64`: Factor to decrease the trust-region radius after a poor step (default: 0.5).
- `γ₂::Float64`: Factor to increase the trust-region radius after a good step (default: 2.0).
- `Δ::Float64`: Initial trust-region radius (default: 0.1).
# Description
This struct encapsulates the parameters used in the Scheinberg trust-region update rules, which control how the trust-region radius is adjusted based on the quality of optimization steps. The step is acccepted if the ratio of the actual reduction in the objective function to the predicted reduction is greater than or equal to the acceptance threshold `η₁` and the gradient norm.
"""
struct ScheinbergParameters <: AbstractTrustRegionParameters
    η₁::Float64  # Acceptance threshold
    ζ::Float64   # Gradient norm threshold for radius update
    γ₁::Float64  # Radius decrease factor
    γ₂::Float64  # Radius increase factor
    Δ::Float64 # Initial trust-region radius

    function ScheinbergParameters(η₁::Float64 = 0.1, ζ::Float64 = 0.75, 
                                   γ₁::Float64 = 0.5, γ₂::Float64 = 2.0, 
                                   Δ::Float64 = 0.1)
        new(η₁, ζ, γ₁, γ₂, Δ)
    end
end

function trust_region_update!(state::TrustRegionState, params::ScheinbergParameters, algoInfo::AlgorithmInfoTR, nlp::AbstractNLPModel)

    if state.ρ >= params.η₁ && norm(state.gradient) >= params.ζ * state.Δ
        # Accept step
        state.x = copy(state.x_candidate)
        state.objective = state.f_candidate
        state.gradient = grad(nlp, state.x)
        state.accepted = true

        # Increase trust-region radius
        state.Δ *= params.γ₂
        push!(algoInfo.path, copy(state.x))
        push!(algoInfo.objectives, state.objective)
        push!(algoInfo.gradient_norms, norm(state.gradient))
        push!(algoInfo.accepted_steps, state.iteration)
    else
        # Decrease trust-region radius
        state.Δ *= params.γ₁
    end
    push!(algoInfo.radii, state.Δ)
end

"""
    SimpleScheinbergParameters

Parameters for the simple Scheinberg trust-region update strategy.

# Fields
- `η₁::Float64`: Acceptance threshold for step quality (default: 0.1).
- `ζ::Float64`: Gradient norm threshold for radius update (default: 0.75).
- `γ₁::Float64`: Factor to decrease the trust-region radius after a poor step (default: 0.5).
- `γ₂::Float64`: Factor to increase the trust-region radius after a good step (default: 2.0).
- `Δ::Float64`: Initial trust-region radius (default: 0.1).

# Description
This struct encapsulates the parameters used in the simple Scheinberg trust-region update rules, which control how the trust-region radius is adjusted based on the quality of optimization steps. The simple Scheinberg method is a variant of the Scheinberg method that uses a simpler acceptance criterion for the step. 
    
    - The step is accepted if the ratio of the actual reduction in the objective function to the predicted reduction is greater than or equal to the acceptance threshold `η₁`. 
    
    - The trust-region radius is increased or decreased based  and the gradient norm.
"""
struct SimpleScheinbergParameters <: AbstractTrustRegionParameters
    η₁::Float64  # Acceptance threshold
    ζ::Float64   # Gradient norm threshold for radius update
    γ₁::Float64  # Radius decrease factor
    γ₂::Float64  # Radius increase factor
    Δ::Float64 # Initial trust-region radius

    function SimpleScheinbergParameters(η₁::Float64 = 0.1, ζ::Float64 = 0.75, 
                                   γ₁::Float64 = 0.5, γ₂::Float64 = 2.0, 
                                   Δ::Float64 = 0.1)
        new(η₁, ζ, γ₁, γ₂, Δ)
    end
end


function trust_region_update!(state::TrustRegionState, params::SimpleScheinbergParameters, algoInfo::AlgorithmInfoTR, nlp::AbstractNLPModel)

    # Update step
    accept_or_reject_step!(state, params, algoInfo, nlp)

    # Update trust-region radius based on the acceptance ratio ρ and gradient norm
    if state.ρ >= params.η₁ && norm(state.gradient) >= params.ζ * state.Δ
        state.Δ *= params.γ₂
        push!(algoInfo.path, copy(state.x))
        push!(algoInfo.objectives, state.objective)
        push!(algoInfo.gradient_norms, norm(state.gradient))
        push!(algoInfo.accepted_steps, state.iteration)
    else
        # Decrease trust-region radius
        state.Δ *= params.γ₁
    end
    push!(algoInfo.radii, state.Δ)
end

# Show function for ScheinbergParameters and SimpleScheinbergParameters
function Base.show(io::IO, params::Union{SimpleScheinbergParameters, ScheinbergParameters})
    println(io, "Scheinberg Parameters:")
    println(io, "  η₁: ", params.η₁)
    println(io, "  ζ: ", params.ζ)
    println(io, "  γ₁: ", params.γ₁)
    println(io, "  γ₂: ", params.γ₂)
    println(io, "  Δ: ", params.Δ)
end