
# Function to compute the trust-region radius based on the acceptance ratio ρ
# and the parameters defined in the Yuan-Fan method.
"""
    R_exp(t, η, β, γ₁, γ₂, M, λ1, λ2)

Computes the trust-region radius based on the acceptance ratio `t` and the parameters defined in the Yuan-Fan method.
- `t::Float64`: Acceptance ratio.
- `η::Float64`: Threshold for the transition between exponential growth and decay.
- `β::Float64`: Lower bound for the radius when `t < η`.
- `γ₁::Float64`: Adjustment factor for the radius when `t < η`.
- `γ₂::Float64`: Adjustment factor for the radius when `t ≥ η`.
- `M::Float64`: Upper bound for the radius when `t ≥ η`.
- `λ1::Float64`: Exponential growth rate for `t < η`.
- `λ2::Float64`: Exponential decay rate for `t ≥ η`.
"""
function R_exp(t, η, β, γ₁, γ₂, M, λ1, λ2)
    if t < η
        return β + (1 - γ₁ - β) * exp(λ1* (t - η))  # Exponential increase for t < η
    else
        return 1 + γ₂ + (M - (1 + γ₂)) * (1- exp(-λ2 * (t - η)))  # Exponential decay for t ≥ η
    end
end


"""
    HeiParameters <: AbstractTrustRegionParameters

Parameters for the trust-region radius update strategy based on the method by Hei et al.

# Fields
- `Δ::Float64`: Initial trust-region radius.
- `η₁::Float64`: Threshold for the ratio `t` that determines the update regime.
- `β::Float64`: Lower bound for the exponential update when `t < η₁`.
- `γ₁::Float64`: Adjustment factor for the exponential update when `t < η₁`.
- `γ₂::Float64`: Adjustment factor for the exponential update when `t ≥ η₁`.
- `M::Float64`: Upper bound for the exponential update when `t ≥ η₁`.
- `λ1::Float64`: Exponential growth rate for `t < η₁`.
- `λ2::Float64`: Exponential decay rate for `t ≥ η₁`.

# Constructor
    HeiParameters(Δ=0.1, η₁=0.1, β=0.1, γ₁=0.1, γ₂=0.5, M=10.0, λ1=0.5, λ2=0.1)

Creates a `HeiParameters` instance with the specified or default values. 

The trust region radius update is of the form 
``
\\Delta = R_{μ}(t) \\cdot \\|s_k\\|
``
where ``R_{exp}`` is a function that computes the coefficient multiplier based on the acceptance ratio ``t``.
"""
struct HeiParameters <: AbstractTrustRegionParameters
    Δ::Float64 # Initial trust-region radius
    η₁::Float64  # Threshold for t
    β::Float64  # Lower bound for R_exp(t) when t < η
    γ₁::Float64 # Adjustment factor for R_exp(t) when t < η
    γ₂::Float64 # Adjustment factor for R_exp(t) when t ≥ η
    M::Float64  # Upper bound for R_exp(t) when t ≥ η
    λ1::Float64 # Exponential growth rate for t < η
    λ2::Float64 # Exponential decay rate for t ≥ η

    function HeiParameters(Δ::Float64 = 0.1, η₁::Float64 = 0.1, β::Float64 = 0.1, γ₁::Float64 = 0.1, γ₂::Float64 = 0.5, 
                           M::Float64 = 10.0, λ1::Float64 = 0.5, λ2::Float64 = 0.1)
        new(Δ, η₁, β, γ₁, γ₂, M, λ1, λ2)
    end
end


function trust_region_update!(state::TrustRegionState, params::HeiParameters, algoInfo::AlgorithmInfoTR, nlp::AbstractNLPModel)

    # Update step
    accept_or_reject_step!(state, params, algoInfo, nlp)

    # Update trust-region radius based on the acceptance ratio ρ and gradient norm
    state.Δ = R_exp(state.ρ, params.η₁, params.β, params.γ₁, params.γ₂, params.M, params.λ1, params.λ2)*norm(state.step)
    
    push!(algoInfo.radii, state.Δ)
end



"""
    HeiGradParameters <: AbstractTrustRegionParameters

Parameters for the trust-region radius update strategy based on gradient information.

# Fields
- `Δ::Float64`: Initial trust-region radius.
- `η₁::Float64`: Threshold for the trust-region update parameter `t`.
- `β::Float64`: Lower bound for the radius update function `R_exp(t)` when `t < η₁`.
- `γ₁::Float64`: Adjustment factor for `R_exp(t)` when `t < η₁`.
- `γ₂::Float64`: Adjustment factor for `R_exp(t)` when `t ≥ η₁`.
- `M::Float64`: Upper bound for `R_exp(t)` when `t ≥ η₁`.
- `λ1::Float64`: Exponential growth rate for `t < η₁`.
- `λ2::Float64`: Exponential decay rate for `t ≥ η₁`.

# Constructor
    HeiGradParameters(Δ=0.1, η₁=0.1, β=0.1, γ₁=0.1, γ₂=0.5, M=10.0, λ1=0.5, λ2=0.1)

Creates a `HeiGradParameters` instance with the specified or default parameter values. The step size is adjusted based on the acceptance ratio `ρ` and the gradient norm.
"""
struct HeiGradParameters <: AbstractTrustRegionParameters
    Δ::Float64 # Initial trust-region radius
    η₁::Float64  # Threshold for t
    β::Float64  # Lower bound for R_exp(t) when t < η
    γ₁::Float64 # Adjustment factor for R_exp(t) when t < η
    γ₂::Float64 # Adjustment factor for R_exp(t) when t ≥ η
    M::Float64  # Upper bound for R_exp(t) when t ≥ η
    λ1::Float64 # Exponential growth rate for t < η
    λ2::Float64 # Exponential decay rate for t ≥ η

    function HeiGradParameters(Δ::Float64 = 0.1, η₁::Float64 = 0.1, β::Float64 = 0.1, γ₁::Float64 = 0.1, γ₂::Float64 = 0.5, 
                           M::Float64 = 10.0, λ1::Float64 = 0.5, λ2::Float64 = 0.1)
        new(Δ, η₁, β, γ₁, γ₂, M, λ1, λ2)
    end
end

function trust_region_update!(state::TrustRegionState, params::HeiGradParameters, algoInfo::AlgorithmInfoTR, nlp::AbstractNLPModel)

    # Update step
    accept_or_reject_step!(state, params, algoInfo, nlp)

    # Update trust-region radius based on the acceptance ratio ρ and gradient norm
    state.Δ = R_exp(state.ρ, params.η₁, params.β, params.γ₁, params.γ₂, params.M, params.λ1, params.λ2)*norm(state.gradient)

    push!(algoInfo.radii, state.Δ)
end


function Base.show(io::IO, params::HeiParameters)
    println(io, "Hei Parameters:")
    println(io, "  Δ: ", params.Δ)
    println(io, "  η₁: ", params.η₁)
    println(io, "  β: ", params.β)
    println(io, "  γ₁: ", params.γ₁)
    println(io, "  γ₂: ", params.γ₂)
    println(io, "  M: ", params.M)
    println(io, "  λ1: ", params.λ1)
    println(io, "  λ2: ", params.λ2)
end

function Base.show(io::IO, params::HeiGradParameters)
    println(io, "Hei Gradient Parameters:")
    println(io, "  Δ: ", params.Δ)
    println(io, "  η₁: ", params.η₁)
    println(io, "  β: ", params.β)
    println(io, "  γ₁: ", params.γ₁)
    println(io, "  γ₂: ", params.γ₂)
    println(io, "  M: ", params.M)
    println(io, "  λ1: ", params.λ1)
    println(io, "  λ2: ", params.λ2)
end

