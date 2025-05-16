
# Function to perform gradient descent
"""
    gradient_descent(nlp::AbstractNLPModel, stop::AbstractStoppingCriteria, info::AlgorithmInfoGD; alpha::Float64 = 0.01)
    Perform gradient descent optimization on the given NLP model.

    # Arguments
    - `nlp::AbstractNLPModel`: The NLP model to optimize.
    - `stop::AbstractStoppingCriteria`: Stopping criteria for the optimization.
    - `info::AlgorithmInfoGD`: Information structure to store algorithm details.
    - `alpha::Float64`: Step size for the gradient descent. Default is 0.01.
    
    # Returns
    - `state::GradientDescentState`: The final state of the optimization.
    - `info::AlgorithmInfoGD`: Updated algorithm information.
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
