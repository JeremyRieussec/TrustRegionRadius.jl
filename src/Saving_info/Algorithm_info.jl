mutable struct AlgorithmInfoTR <: AbstractAlgorithmInfo
    problem::AbstractNLPModel # Optimization problem being solved
    stopping_criteria::AbstractStoppingCriteria # Stopping criteria for the algorithm
    parameters::AbstractTrustRegionParameters # Parameters used in the algorithm

    iterations::Int                # Total number of iterations
    
    accepted_steps::Array          # Number of accepted steps
    boundary_steps::Array          # Iteration of steps on trust-region boundary
    number_cg_iterations::Array     # Number of CG iterations

    path::Array{Vector{Float64}}   # Path of optimization
    objectives::Array{Float64}     # Objective values at each step
    gradient_norms::Array{Float64}  # Norm of the gradient at each step
    radii::Array{Float64}          # Trust-region radii at each step
    rhos::Array{Float64}           # Ratios of actual to predicted reductions
    mus::Array{Float64}            # Trust-region multipliers at each step

    final_x::Vector{Float64}       # Final point
    final_objective::Float64       # Final objective value
    final_gradient_norm::Float64   # Norm of the final gradient
    final_trust_region_radius::Float64 # Final trust-region radius

    converged::String                # Convergence status

    function AlgorithmInfoTR(problem::AbstractNLPModel, stopping_criteria::AbstractStoppingCriteria,    
                            parameters::AbstractTrustRegionParameters,
                            iterations::Int, accepted_steps::Array, boundary_steps::Array, number_cg_iterations::Array , 
                            path::Array{Vector{Float64}}, objectives::Array{Float64}, gradient_norms::Array{Float64},
                            radii::Array{Float64}, rhos::Array{Float64}, mus::Array{Float64},
                            final_x::Vector{Float64}, final_objective::Float64, 
                            final_gradient_norm::Float64, final_trust_region_radius::Float64,
                            converged::String)
        new(problem, stopping_criteria, parameters, iterations, accepted_steps, boundary_steps, number_cg_iterations,path, objectives, gradient_norms, radii, rhos, mus, final_x, final_objective, final_gradient_norm,  final_trust_region_radius, converged)
    end
end

function Base.show(io::IO, algo_info::AlgorithmInfoTR)
    
    println(io, "--------- Algorithm Information ---------")
    println(io, algo_info.problem)
    println(io, "Starting point: ", algo_info.problem.meta.x0)
    println(io, algo_info.stopping_criteria)
    println(io, algo_info.parameters)
    println(io, "--------- Algorithm History ---------")
    println(io, "  Convergence status: ", algo_info.converged)
    println(io, "  Total iterations: ", algo_info.iterations)
    println(io, "  Number Accepted steps: ", length(algo_info.accepted_steps))
    println(io, "  Nb Accepted steps: ", algo_info.accepted_steps)

    println(io, "  Nb Boundary steps: ", length(algo_info.boundary_steps))
    println(io, "  Boundary steps: ", algo_info.boundary_steps)

    println(io, "  Objectives: ", algo_info.objectives)
    println(io, "  Gradient norms: ", algo_info.gradient_norms)

    println(io, "  Trust-region radii: ", algo_info.radii)
    println(io, "  Ratios: ", algo_info.rhos)
    println(io, "  Trust-region multipliers: ", algo_info.mus)

    println(io, "--------- Final State ---------")
    println(io, "  Final point: ", algo_info.final_x)
    println(io, "  Final objective value: ", algo_info.final_objective)
    println(io, "  Final gradient norm: ", algo_info.final_gradient_norm)
    println(io, "  Final trust-region radius: ", algo_info.final_trust_region_radius)
end


mutable struct AlgorithmInfoGD <: AbstractAlgorithmInfo
    problem::AbstractNLPModel
    stopping_criteria::AbstractStoppingCriteria

    iterations::Int

    step_sizes::Vector{Float64}

    objectives::Vector{Float64}
    gradient_norms::Vector{Float64}
    path::Vector{Vector{Float64}}

    final_x::Vector{Float64}
    final_objective::Float64
    final_gradient_norm::Float64

    converged::String

    function AlgorithmInfoGD(
        problem::AbstractNLPModel,
        stopping_criteria::AbstractStoppingCriteria,
        iterations::Int,
        step_sizes::Vector{Float64},
        objectives::Vector{Float64},
        gradient_norms::Vector{Float64},
        path::Vector{Vector{Float64}},
        final_x::Vector{Float64},
        final_objective::Float64,
        final_gradient_norm::Float64,
        converged::String
    )
        new(
            problem, stopping_criteria, iterations,
            step_sizes, objectives, gradient_norms, path,
            final_x, final_objective, final_gradient_norm, converged
        )
    end
end

# Fucntion to add an iteration to the algorithm information
"""
    add_iteration!(algo_info::AlgorithmInfoGD, step_size, objective, gradient_norm, x)  
    
Adds an iteration to the algorithm information.
"""
function add_iteration!(algo_info::AlgorithmInfoGD, step_size, objective, gradient_norm, x)
    push!(algo_info.step_sizes, step_size)
    push!(algo_info.objectives, objective)
    push!(algo_info.gradient_norms, gradient_norm)
    push!(algo_info.path, x)
    algo_info.iterations += 1
end

function Base.show(io::IO, algo_info::AlgorithmInfoGD)
    println(io, "--------- Algorithm Information (Gradient Descent) ---------")
    println(io, algo_info.problem)
    println(io, "Starting point: ", algo_info.problem.meta.x0)
    println(io, algo_info.stopping_criteria)
    println(io, "--------- Algorithm History ---------")
    println(io, "  Convergence status: ", algo_info.converged)
    println(io, "  Total iterations: ", algo_info.iterations)
    println(io, "  Step sizes: ", algo_info.step_sizes)
    println(io, "  Objectives: ", algo_info.objectives)
    println(io, "  Gradient norms: ", algo_info.gradient_norms)
    println(io, "--------- Final State ---------")
    println(io, "  Final point: ", algo_info.final_x)
    println(io, "  Final objective value: ", algo_info.final_objective)
    println(io, "  Final gradient norm: ", algo_info.final_gradient_norm)
end