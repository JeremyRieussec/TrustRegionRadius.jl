
# ============================================================
# TROutput — lean output struct for benchmarking
#
# Returned by trust_region_solver (canonical R1–R4 interface).
# Contains only the fields needed for Dolan-Moré profiles,
# data profiles, and trajectory plots.
# ============================================================

"""
    TROutput

Output of `trust_region_solver`. Stores the final status and all
trajectory data needed for the companion COAP benchmark.

# Fields
- `status::Symbol`:                  `:solved`, `:max_iter`, or `:failure`
- `iterations::Int`:                 number of iterations performed
- `f_evals::Int`:                    number of objective function evaluations
- `final_grad_norm::Float64`:        ‖∇f(x*)‖ at termination
- `final_delta::Float64`:            trust-region radius Δ at termination
- `delta_trajectory::Vector{Float64}`:     Δ_k for k = 0, 1, …
- `grad_norm_trajectory::Vector{Float64}`: ‖g_k‖ for k = 0, 1, …
- `obj_trajectory::Vector{Float64}`:       f(x_k) for k = 0, 1, …
- `solve_time::Float64`:             wall-clock time in seconds
"""
mutable struct TROutput <: AbstractAlgorithmInfo
    status::Symbol
    iterations::Int
    f_evals::Int
    g_evals::Int 
    h_evals::Int
    h_prod_evals::Int
    final_grad_norm::Float64
    final_delta::Float64
    delta_trajectory::Vector{Float64}
    grad_norm_trajectory::Vector{Float64}
    obj_trajectory::Vector{Float64}
    solve_time::Float64
end

function Base.show(io::IO, out::TROutput)
    println(io, "--------- TROutput ---------")
    println(io, "  Status:             ", out.status)
    println(io, "  Iterations:         ", out.iterations)
    println(io, "  f evaluations:      ", out.f_evals)
    println(io, "  g evaluations:      ", out.g_evals)
    println(io, "  h evaluations:      ", out.h_evals)
    println(io, "  h_prod evaluations: ", out.h_prod_evals)
    println(io, "  Final ‖g‖:          ", out.final_grad_norm)
    println(io, "  Final Δ:            ", out.final_delta)
    println(io, "  Solve time (s):     ", out.solve_time)
end

# ============================================================
# TRSolverParams — common parameters for trust_region_solver
# ============================================================

"""
    TRSolverParams

Solver-level parameters shared across all canonical radius update rules.
The update-specific factors (γ₁, γ₂, γ₃, ζ, μ) live inside the
`AbstractRadiusUpdate` rule struct.

# Fields
- `η₁::Float64`:           lower acceptance threshold (reject if ρ < η₁); default 0.1
- `η₂::Float64`:           upper threshold for "very successful" step; default 0.9
- `Δ₀::Float64`:           initial trust-region radius (overridden by R4's μ·‖g₀‖); default 1.0
- `max_iterations::Int`:   iteration budget; default 10_000
- `tol::Float64`:          gradient-norm convergence tolerance; default 1e-5

# Constructor
    TRSolverParams(; η₁=0.1, η₂=0.9, Δ₀=1.0, max_iterations=10_000, tol=1e-5)
"""
struct TRSolverParams
    η₁::Float64
    η₂::Float64
    Δ₀::Float64
    max_iterations::Int
    tol::Float64

    function TRSolverParams(; η₁::Float64 = 0.1, η₂::Float64 = 0.9,
                              Δ₀::Float64 = 1.0,
                              max_iterations::Int = 10_000,
                              tol::Float64 = 1e-5)
        @assert 0 <= η₁ < η₂ < 1  "Need 0 ≤ η₁ < η₂ < 1"
        @assert Δ₀ > 0            "Need Δ₀ > 0"
        @assert max_iterations > 0
        @assert tol > 0
        new(η₁, η₂, Δ₀, max_iterations, tol)
    end
end

function Base.show(io::IO, p::TRSolverParams)
    println(io, "TRSolverParams:")
    println(io, "  η₁: ", p.η₁, "  η₂: ", p.η₂)
    println(io, "  Δ₀: ", p.Δ₀)
    println(io, "  max_iterations: ", p.max_iterations)
    println(io, "  tol: ", p.tol)
end
