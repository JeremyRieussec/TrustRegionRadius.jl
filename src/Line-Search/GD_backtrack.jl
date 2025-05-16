# Function to perform gradient descent with backtracking line search
"""
    LS_steepest_backtrack(nlp::AbstractNLPModel, stop::AbstractStoppingCriteria; eta=1e-4, sigma=0.66)
    Perform gradient descent optimization with backtracking line search on the given NLP model.

    # Arguments
    - `nlp::AbstractNLPModel`: The NLP model to optimize.
    - `stop::AbstractStoppingCriteria`: Stopping criteria for the optimization.
    - `eta::Float64`: Parameter for backtracking line search. Default is 1e-4.
    - `sigma::Float64`: Parameter for backtracking line search. Default is 0.66.

    # Returns
    - `state::GradientDescentState`: The final state of the optimization.
    - `info::AlgorithmInfoGD`: Updated algorithm information.
"""
function LS_steepest_backtrack(nlp::AbstractNLPModel, stop::AbstractStoppingCriteria; eta=1e-4, sigma=0.66)
  @info "Starting backtracking line search gradient descent optimization."

  # Initialize variables
  state = initialize_gradient_descent_state(nlp)

  # Initialize algorithm information
    info = AlgorithmInfoGD(
        nlp,
        stop,
        0,
        Float64[],
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
    t = 1.0
    x_trial = state.x - t * state.gradient
    f_trial = obj(nlp, x_trial)
    slope = dot(state.gradient, state.gradient)

    while f_trial > state.objective - eta * t * slope
      t *= sigma
      x_trial = state.x - t * state.gradient
      f_trial = obj(nlp, x_trial)
    end

    state.x = x_trial
    state.objective = f_trial
    state.gradient = grad(nlp, state.x)
    state.iter = k

    # Add iteration information
    add_iteration!(info, t, state.objective, norm(state.gradient), state.x)
  end

  @info "Maximum iterations reached WITHOUT convergence: $(stop.max_iterations)."
  info.converged = MAX_ITERATIONS_SYMBOL

  info.final_x = state.x
  info.final_objective = state.objective
  info.final_gradient_norm = norm(state.gradient)

  return state, info
end