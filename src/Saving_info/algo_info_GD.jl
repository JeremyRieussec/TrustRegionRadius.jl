# This file contains the algorithm information structure for storing the results of the optimization process
# for gradient based methods.

"""
    AlgorithmInfoGD <: AbstractAlgorithmInfo

Holds information about the execution of a Gradient Descent (GD) algorithm on a nonlinear programming problem.

# Fields
- `problem::AbstractNLPModel`: The nonlinear programming model being solved.
- `stopping_criteria::AbstractStoppingCriteria`: The stopping criteria used to terminate the algorithm.
- `iterations::Int`: The total number of iterations performed.
- `step_sizes::Vector{Float64}`: The sequence of step sizes used at each iteration.
- `objectives::Vector{Float64}`: The objective function values at each iteration.
- `gradient_norms::Vector{Float64}`: The norms of the gradient at each iteration.
- `path::Vector{Vector{Float64}}`: The sequence of iterates (solution points) visited by the algorithm.
- `final_x::Vector{Float64}`: The final solution found by the algorithm.
- `final_objective::Float64`: The objective function value at the final solution.
- `final_gradient_norm::Float64`: The gradient norm at the final solution.
- `converged::String`: A string indicating the reason for convergence or termination.

# Constructor
    AlgorithmInfoGD(
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

Creates a new `AlgorithmInfoGD` instance with the specified fields.
"""
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