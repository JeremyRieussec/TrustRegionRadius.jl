
"""
    trust_region_with_cg(nlp::AbstractNLPModel, params::AbstractTrustRegionParameters, stoppingTest::AbstractStoppingCriteria)

Performs trust-region optimization using the Conjugate Gradient (CG) Steihaug method to solve the trust-region subproblem.

# Arguments
- `nlp::AbstractNLPModel`: The nonlinear programming model containing the objective function, gradient, and Hessian-vector product.
- `params::AbstractTrustRegionParameters`: Parameters controlling the trust-region algorithm, such as initial radius, update rules, and tolerances.
- `stoppingTest::AbstractStoppingCriteria`: Criteria for stopping the optimization, including maximum iterations and convergence tolerances.

# Returns
- `state`: The final state of the trust-region algorithm, containing the solution, objective value, gradient, and trust-region radius.
- `algoInfo`: An `AlgorithmInfoTR` object containing detailed information about the optimization process, including iterates, objective values, gradient norms, trust-region radii, CG iterations, and convergence status.

# Description
This function initializes the trust-region state and iteratively solves the trust-region subproblem using the truncated CG Steihaug method. At each iteration, it checks for convergence, computes the candidate step, evaluates the predicted and actual reductions, updates the trust-region radius, and records relevant information. The process continues until convergence or the maximum number of iterations is reached.

# Logging
- Logs the start of the optimization and notifies if the maximum number of iterations is reached without convergence.
"""
function trust_region_with_cg(nlp::AbstractNLPModel , params::AbstractTrustRegionParameters, stoppingTest::AbstractStoppingCriteria)

    @info "Starting trust-region optimization with CG:"
    println(params)

    # Initialize the trust-region state
    state = generate_TR_state(params, nlp)

    algoInfo = AlgorithmInfoTR(nlp, stoppingTest, params, 0, [], [], [], [copy(state.x)], [state.objective], [norm(state.gradient)], [state.Δ], Float64[], Float64[], state.x, state.objective, norm(state.gradient), state.Δ, "")

    for k in 1:stoppingTest.max_iterations

        # iteration number
        state.iteration = k

        # Convergence check
        if check_convergence(state, stoppingTest, algoInfo)
            algoInfo.final_x = state.x
            algoInfo.final_objective = state.objective
            algoInfo.final_gradient_norm = norm(state.gradient)
            algoInfo.final_trust_region_radius = state.Δ
            algoInfo.iterations = k - 1
            return state, algoInfo
        end

        # Solve trust-region subproblem using truncated CG
        p, on_boundary, cg_iter = truncated_cg_steihaug(nlp, state.x, state.gradient, state.Δ)
        state.step = p
        state.on_boundary = on_boundary
        state.number_cg_iterations = cg_iter
        push!(algoInfo.number_cg_iterations, cg_iter)
        if on_boundary
            push!(algoInfo.boundary_steps , k)
        end
        

        # Candidate point and objective
        state.x_candidate = state.x + state.step
        state.f_candidate = obj(nlp, state.x_candidate)

        # Predicted reduction
        predicted_reduction = - dot(state.gradient, state.step) - 0.5 * dot(state.step, hprod(nlp, state.x, state.step))
        # Actual reduction
        actual_reduction = state.objective - state.f_candidate
        
        # Compute ρ
        state.ρ = actual_reduction / predicted_reduction
        push!(algoInfo.rhos, state.ρ)
        
        # Update trust-region radius using the appropriate method
        trust_region_update!(state, params, algoInfo, nlp) 

    end
    @info "Maximum iterations reached WITHOUT convergence: $(stoppingTest.max_iterations)."
    algoInfo.converged = MAX_ITERATIONS_SYMBOL

    algoInfo.final_x = state.x
    algoInfo.final_objective = state.objective
    algoInfo.final_gradient_norm = norm(state.gradient)
    algoInfo.final_trust_region_radius = state.Δ
    algoInfo.iterations = stoppingTest.max_iterations
    
    return state, algoInfo
end