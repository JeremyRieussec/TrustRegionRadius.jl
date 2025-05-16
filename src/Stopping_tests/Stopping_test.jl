
# Stopping criteria for optimization algorithms
"""
    StoppingCriteriaGradient

A structure representing the stopping criteria for gradient-based optimization algorithms. 
This type typically encapsulates parameters such as tolerance thresholds, maximum number of iterations, 
and other conditions under which the optimization process should terminate.

# Fields
- `tol::Float64`: The tolerance for the gradient norm below which the algorithm will stop.
- `max_iter::Int`: The maximum number of iterations allowed.
- `other_criteria`: Any additional stopping criteria relevant to the optimization process.

# Usage
Create an instance of `StoppingCriteriaGradient` to control when an optimization routine should terminate 
based on the gradient's properties and other user-defined conditions.
"""
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

function check_convergence(state::TrustRegionState, stopping_criteria::StoppingCriteriaGradient, algoInfo::AlgorithmInfoTR)
    gradient_norm = norm(state.gradient)
    if gradient_norm < stopping_criteria.gradient_norm_tolerance
        @info "CONVERGED: Gradient norm ($gradient_norm) is below the tolerance ($(stopping_criteria.gradient_norm_tolerance))."
        algoInfo.converged = GRADIENT_TOLERANCE_SYMBOL
        return true
    end
    return false
end

function check_convergence(state::GradientDescentState, stopping_criteria::StoppingCriteriaGradient, algoInfo::AlgorithmInfoGD)
    gradient_norm = norm(state.gradient)
    if gradient_norm < stopping_criteria.gradient_norm_tolerance
        @info "CONVERGED: Gradient norm ($gradient_norm) is below the tolerance ($(stopping_criteria.gradient_norm_tolerance))."
        algoInfo.converged = GRADIENT_TOLERANCE_SYMBOL
        return true
    end
    return false
end