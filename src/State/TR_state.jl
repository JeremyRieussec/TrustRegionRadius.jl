mutable struct TrustRegionState <: AbstractTrustRegionState
    x::Vector{Float64}       # Current point
    Δ::Float64               # Trust-region radius

    objective::Float64       # Objective value at the current point
    gradient::Vector{Float64} # Gradient at the current point
    
    iteration::Int           # Current iteration number

    step::Vector{Float64}     # Step taken
    x_candidate::Vector{Float64} # Candidate step
    f_candidate::Float64 # Objective value at the candidate point
    ρ::Float64               # Ratio of actual to predicted reduction

    accepted::Bool           # Whether the step was accepted

    on_boundary::Bool        # Whether the step is on the boundary of the trust region
    number_cg_iterations::Int # Number of CG iterations performed
end

function Base.show(io::IO, state::TrustRegionState)
    println(io, "------------ TrustRegionState --------------")
    println(io, "  Current point (x): ", state.x)
    println(io, "  Trust-region radius (Δ): ", state.Δ)
    println(io, "  Objective value: ", state.objective)
    println(io, "  Gradient: ", state.gradient)
    println(io, "  Iteration: ", state.iteration)
    println(io, "  Candidate point (x_candidate): ", state.x_candidate)
    println(io, "  Candidate objective value (f_candidate): ", state.f_candidate)
    println(io, "  Ratio of actual to predicted reduction (ρ): ", state.ρ)
    println(io, "  Step accepted: ", state.accepted)
    println(io, "  On boundary: ", state.on_boundary)
end