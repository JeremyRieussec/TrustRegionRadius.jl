
# Function to perform gradient descent

"""
    gradient_descent(nlp::AbstractNLPModel, stop::AbstractStoppingCriteria; alpha::Float64 = 0.01)

Performs gradient descent optimization on a nonlinear programming (NLP) model.

# Arguments
- `nlp::AbstractNLPModel`: The nonlinear programming model to optimize. Must provide `grad` and `obj` methods.
- `stop::AbstractStoppingCriteria`: Stopping criteria for the optimization process (e.g., maximum iterations, tolerance).
- `alpha::Float64=0.01`: (Optional) Step size (learning rate) for the gradient descent updates.

# Returns
- `state`: The final state of the optimization, including the last iterate, gradient, and objective value.
- `info`: An `AlgorithmInfoGD` object containing the optimization history, convergence information, and final results.

# Description
Initializes the optimization state and iteratively updates the solution by moving in the direction of the negative gradient, scaled by the step size `alpha`. At each iteration, checks for convergence using the provided stopping criteria. Records the optimization history and returns the final state and information object.

# Logging
- Logs the start of the optimization process.
- Logs a message if the maximum number of iterations is reached without convergence.

# Example
```julia
state, info = gradient_descent(nlp, stop; alpha=0.01)
```
"""
function gradient_descent(nlp::AbstractNLPModel, stop::AbstractStoppingCriteria; alpha::Float64 = 0.01)
    @info "Starting Gradient Descent optimization."

    # Initialize variables
    state = initialize_gradient_descent_state(nlp)

    # Initialize algorithm information
    info = AlgorithmInfoGD(
        nlp,
        stop,
        0,
        [alpha],
        [state.objective],
        [norm(state.gradient)],
        [copy(state.x)],
        Vector{Float64}(undef, length(state.x)),
        state.objective,
        norm(state.gradient),
        ""
    )
    
    for k in 1:stop.max_iterations
        if check_convergence(state, stop, info)
            info.final_x = state.x
            info.final_objective = state.objective
            info.final_gradient_norm = norm(state.gradient)
            return state, info
        end
        state.x = state.x - alpha * state.gradient
        state.gradient = grad(nlp, state.x)
        state.objective = obj(nlp, state.x)
        state.iter = k
        # Add iteration information
        add_iteration!(info, alpha, state.objective, norm(state.gradient), state.x)
    end
    @info "Maximum iterations reached WITHOUT convergence: $(stop.max_iterations)."
    info.converged = MAX_ITERATIONS_SYMBOL

    info.final_x = state.x
    info.final_objective = state.objective
    info.final_gradient_norm = norm(state.gradient)

    return state, info
end
