"""
    HeiFanYuanParameters <: AbstractTrustRegionParameters

Parameters for the modified Hei gradient-based trust region radius update strategy.

# Fields
- `μ::Float64`: Initial radius multiplier.
- `η₁::Float64`: Threshold for the trust region ratio `t`.
- `β::Float64`: Lower bound for the expected radius `R_exp(t)` when `t < η₁`.
- `γ₁::Float64`: Adjustment factor for `R_exp(t)` when `t < η₁`.
- `γ₂::Float64`: Adjustment factor for `R_exp(t)` when `t ≥ η₁`.
- `M::Float64`: Upper bound for `R_exp(t)` when `t ≥ η₁`.
- `λ1::Float64`: Exponential growth rate for `t < η₁`.
- `λ2::Float64`: Exponential decay rate for `t ≥ η₁`.

# Constructor
    HeiFanYuanParameters(
        μ::Float64 = 0.1,
        η₁::Float64 = 0.1,
        β::Float64 = 0.1,
        γ₁::Float64 = 0.1,
        γ₂::Float64 = 0.5,
        M::Float64 = 10.0,
        λ1::Float64 = 0.5,
        λ2::Float64 = 0.1
    )

Creates a new instance of `HeiFanYuanParameters` with the specified or default parameter values.
"""
mutable struct HeiFanYuanParameters <: AbstractTrustRegionParameters
    μ::Float64   # Initial radius multiplier
    η₁::Float64  # Threshold for t
    β::Float64  # Lower bound for R_exp(t) when t < η
    γ₁::Float64 # Adjustment factor for R_exp(t) when t < η
    γ₂::Float64 # Adjustment factor for R_exp(t) when t ≥ η
    M::Float64  # Upper bound for R_exp(t) when t ≥ η
    λ1::Float64 # Exponential growth rate for t < η
    λ2::Float64 # Exponential decay rate for t ≥ η

    function HeiFanYuanParameters(μ::Float64 = 0.1, η₁::Float64 = 0.1, β::Float64 = 0.1, γ₁::Float64 = 0.1, γ₂::Float64 = 0.5, 
                           M::Float64 = 10.0, λ1::Float64 = 0.5, λ2::Float64 = 0.1)
        new(μ, η₁, β, γ₁, γ₂, M, λ1, λ2)
    end
end


function trust_region_update!(state::TrustRegionState, params::HeiFanYuanParameters, algoInfo::AlgorithmInfoTR, nlp::AbstractNLPModel)

    # Update step
    accept_or_reject_step!(state, params, algoInfo, nlp)

    params.μ *= R_exp(state.ρ, params.η₁, params.β, params.γ₁, params.γ₂, params.M, params.λ1, params.λ2)

    state.Δ = params.μ * norm(state.gradient)
    
    push!(algoInfo.mus, params.μ)
    push!(algoInfo.radii, state.Δ)
end

function Base.show(io::IO, params::HeiFanYuanParameters)
    println(io, "Hei Gradient Parameters (Modified):")
    println(io, "  μ: ", params.μ)
    println(io, "  η₁: ", params.η₁)
    println(io, "  β: ", params.β)
    println(io, "  γ₁: ", params.γ₁)
    println(io, "  γ₂: ", params.γ₂)
    println(io, "  M: ", params.M)
    println(io, "  λ1: ", params.λ1)
    println(io, "  λ2: ", params.λ2)
end
