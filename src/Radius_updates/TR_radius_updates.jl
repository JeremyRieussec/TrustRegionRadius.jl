struct SimpleTointGouldTointParameters <: AbstractTrustRegionParameters
    η₁::Float64  # Acceptance threshold
    η₂::Float64  # Good step threshold
    
    γ₁::Float64  # Radius decrease factor
    γ₂::Float64  # Radius increase factor

    Δ::Float64  # Initial trust-region radius

    function SimpleTointGouldTointParameters(η₁::Float64 = 0.1, η₂::Float64 = 0.75, 
                                             γ₁::Float64 = 0.5, γ₂::Float64 = 2.0, 
                                             Δ::Float64 = 0.1)
        new(η₁, η₂, γ₁, γ₂, Δ)
    end
end

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


function R_exp(t, η, β, γ₁, γ₂, M, λ1, λ2)
    if t < η
        return β + (1 - γ₁ - β) * exp(λ1* (t - η))  # Exponential increase for t < η
    else
        return 1 + γ₂ + (M - (1 + γ₂)) * (1- exp(-λ2 * (t - η)))  # Exponential decay for t ≥ η
    end
end


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

struct HeiParametersModified <: AbstractTrustRegionParameters
    Δ::Float64 # Initial trust-region radius
    η₁::Float64  # Threshold for t
    β::Float64  # Lower bound for R_exp(t) when t < η
    γ₁::Float64 # Adjustment factor for R_exp(t) when t < η
    γ₂::Float64 # Adjustment factor for R_exp(t) when t ≥ η
    M::Float64  # Upper bound for R_exp(t) when t ≥ η
    λ1::Float64 # Exponential growth rate for t < η
    λ2::Float64 # Exponential decay rate for t ≥ η

    function HeiParametersModified(Δ::Float64 = 0.1, η₁::Float64 = 0.1, β::Float64 = 0.1, γ₁::Float64 = 0.1, γ₂::Float64 = 0.5, 
                           M::Float64 = 10.0, λ1::Float64 = 0.5, λ2::Float64 = 0.1)
        new(Δ, η₁, β, γ₁, γ₂, M, λ1, λ2)
    end
end

mutable struct HeiGradParametersModified <: AbstractTrustRegionParameters
    μ::Float64   # Initial radius multiplier
    η₁::Float64  # Threshold for t
    β::Float64  # Lower bound for R_exp(t) when t < η
    γ₁::Float64 # Adjustment factor for R_exp(t) when t < η
    γ₂::Float64 # Adjustment factor for R_exp(t) when t ≥ η
    M::Float64  # Upper bound for R_exp(t) when t ≥ η
    λ1::Float64 # Exponential growth rate for t < η
    λ2::Float64 # Exponential decay rate for t ≥ η

    function HeiGradParametersModified(μ::Float64 = 0.1, η₁::Float64 = 0.1, β::Float64 = 0.1, γ₁::Float64 = 0.1, γ₂::Float64 = 0.5, 
                           M::Float64 = 10.0, λ1::Float64 = 0.5, λ2::Float64 = 0.1)
        new(μ, η₁, β, γ₁, γ₂, M, λ1, λ2)
    end
end


function Base.show(io::IO, params::SimpleTointGouldTointParameters)
    println(io, "Simple Toint-Gould-Toint Parameters:")
    println(io, "  η₁: ", params.η₁)
    println(io, "  η₂: ", params.η₂)
    println(io, "  γ₁: ", params.γ₁)
    println(io, "  γ₂: ", params.γ₂)
    println(io, "  Δ: ", params.Δ)
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

function Base.show(io::IO, params::Union{SimpleScheinbergParameters, ScheinbergParameters})
    println(io, "Scheinberg Parameters:")
    println(io, "  η₁: ", params.η₁)
    println(io, "  ζ: ", params.ζ)
    println(io, "  γ₁: ", params.γ₁)
    println(io, "  γ₂: ", params.γ₂)
    println(io, "  Δ: ", params.Δ)
end

function Base.show(io::IO, params::YuanFanParameters)
    println(io, "Yuan-Fan Parameters:")
    println(io, "  η₁: ", params.η₁)
    println(io, "  η₂: ", params.η₂)
    println(io, "  γ₁: ", params.γ₁)
    println(io, "  γ₂: ", params.γ₂)
    println(io, "  μ: ", params.μ)
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

function Base.show(io::IO, params::HeiParametersModified)
    println(io, "Hei Parameters (Modified):")
    println(io, "  Δ: ", params.Δ)
    println(io, "  η₁: ", params.η₁)
    println(io, "  β: ", params.β)
    println(io, "  γ₁: ", params.γ₁)
    println(io, "  γ₂: ", params.γ₂)
    println(io, "  M: ", params.M)
    println(io, "  λ1: ", params.λ1)
    println(io, "  λ2: ", params.λ2)
end

function Base.show(io::IO, params::HeiGradParametersModified)
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