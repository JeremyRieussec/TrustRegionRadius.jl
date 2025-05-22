# Function to perform gradient descent with backtracking line search

"""
  LS_steepest_backtrack(nlp::AbstractNLPModel, stop::AbstractStoppingCriteria; eta=1e-4, sigma=0.66)

Performs gradient descent optimization using a backtracking line search strategy.

# Arguments
- `nlp::AbstractNLPModel`: The nonlinear programming model to be optimized. Must provide `obj` and `grad` methods for objective and gradient evaluation.
- `stop::AbstractStoppingCriteria`: Stopping criteria for the optimization process, such as maximum iterations or tolerance thresholds.
- `eta`: (Optional) Armijo condition parameter for sufficient decrease. Default is `1e-4`.
- `sigma`: (Optional) Step size reduction factor for backtracking. Default is `0.66`.

# Returns
- `state`: The final state of the gradient descent algorithm, containing the optimized variables, objective value, and gradient.
- `info`: An `AlgorithmInfoGD` object containing detailed information about the optimization process, including iteration history and convergence status.

# Description
This function implements the steepest descent (gradient descent) method with a backtracking line search to determine the step size at each iteration. The step size is reduced by a factor of `sigma` until the Armijo condition is satisfied. The optimization stops when the stopping criteria are met or the maximum number of iterations is reached.

# Example
```julia
state, info = LS_steepest_backtrack(nlp, stop; eta=1e-4, sigma=0.66)
```
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