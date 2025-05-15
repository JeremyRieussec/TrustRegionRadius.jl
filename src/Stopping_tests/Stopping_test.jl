abstract type AbstractStoppingCriteria end

mutable struct StoppingCriteriaGradient <: AbstractStoppingCriteria
    max_iterations::Int       # Maximum number of iterations
    gradient_norm_tolerance::Float64        # Tolerance for convergence

    function StoppingCriteriaGradient(max_iterations::Int = 100, 
                               gradient_norm_tolerance::Float64 = 1e-5)
        new(max_iterations, gradient_norm_tolerance )
    end
end

function Base.show(io::IO, criteria::StoppingCriteriaGradient)
    println(io, "Stopping Criteria:")
    println(io, "  Max Iterations: ", criteria.max_iterations)
    println(io, "  Gradient Norm Tolerance: ", criteria.gradient_norm_tolerance)
end
