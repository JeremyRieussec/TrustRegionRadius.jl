function generate_TR_state(params::AbstractTrustRegionParameters, nlp::AbstractNLPModel)

    x_0 = nlp.meta.x0

    gradient = grad(nlp, x_0)
    objective = obj(nlp, x_0)

    iteration = 0

    step = Vector{Float64}(undef, length(x_0))
    x_candidate = Vector{Float64}(undef, length(x_0))
    f_candidate = Inf

    ρ = -Inf

    accepted = false
    on_boundary = false

    cg_iterations = 0

    return TrustRegionState(x_0, params.Δ * norm(gradient), objective, gradient,  iteration, step, x_candidate, f_candidate, ρ, accepted, on_boundary, cg_iterations)
end

function generate_TR_state(params::Union{YuanFanParameters, HeiGradParametersModified}, nlp::AbstractNLPModel)

    x_0 = nlp.meta.x0

    objective = obj(nlp, x_0)
    gradient = grad(nlp, x_0)
    
    iteration = 0

    step = Vector{Float64}(undef, length(x_0))
    x_candidate = Vector{Float64}(undef, length(x_0))
    f_candidate = Inf

    ρ = -Inf

    accepted = false
    on_boundary = false
    cg_iterations = 0

    return TrustRegionState(x_0, params.μ * norm(gradient) , objective, gradient, iteration, step, x_candidate, f_candidate, ρ, accepted, on_boundary, cg_iterations)
end

function accept_or_reject_step!(state::TrustRegionState, params::AbstractTrustRegionParameters, algoInfo::AlgorithmInfoTR, nlp::AbstractNLPModel)
    if state.ρ > params.η₁
        state.x = copy(state.x_candidate)
        state.objective = state.f_candidate
        state.gradient = grad(nlp, state.x)
        state.accepted = true

        push!(algoInfo.path, copy(state.x))
        push!(algoInfo.objectives, state.objective)
        push!(algoInfo.gradient_norms, norm(state.gradient))
        push!(algoInfo.accepted_steps, state.iteration)
    else
        state.accepted = false
    end
end

function trust_region_update!(state::TrustRegionState, params::SimpleTointGouldTointParameters, algoInfo::AlgorithmInfoTR, nlp::AbstractNLPModel)

    # Update step
    accept_or_reject_step!(state, params, algoInfo, nlp)
    
    # Update trust-region radius based on the acceptance ratio ρ
    if state.ρ > params.η₁
        state.Δ *= params.γ₂ 
    else
        state.Δ *= params.γ₁
    end
    push!(algoInfo.radii, state.Δ)
end

function trust_region_update!(state::TrustRegionState, params::TointGouldTointParameters, algoInfo::AlgorithmInfoTR, nlp::AbstractNLPModel)

    # Update step
    accept_or_reject_step!(state, params, algoInfo, nlp)
    
    # Update trust-region radius based on the acceptance ratio ρ
    if state.ρ > params.η₂
        state.Δ *= params.γ₃ 
    elseif state.ρ < params.η₁
        state.Δ *= params.γ₁
    else
        state.Δ *= params.γ₂
    end
    push!(algoInfo.radii, state.Δ)
end

function trust_region_update!(state::TrustRegionState, params::YuanFanParameters, algoInfo::AlgorithmInfoTR, nlp::AbstractNLPModel)

    # Update step
    accept_or_reject_step!(state, params, algoInfo, nlp)

    if state.ρ > params.η₂
        params.μ *= params.γ₂
    elseif state.ρ < params.η₁
        params.μ *= params.γ₁
    end
    state.Δ = params.μ * norm(state.gradient)
    push!(algoInfo.mus, params.μ)
    push!(algoInfo.radii, state.Δ)
end

function trust_region_update!(state::TrustRegionState, params::HeiParameters, algoInfo::AlgorithmInfoTR, nlp::AbstractNLPModel)

    # Update step
    accept_or_reject_step!(state, params, algoInfo, nlp)

    # Update trust-region radius based on the acceptance ratio ρ and gradient norm
    state.Δ = R_exp(state.ρ, params.η₁, params.β, params.γ₁, params.γ₂, params.M, params.λ1, params.λ2)*norm(state.step)
    
    push!(algoInfo.radii, state.Δ)
end

function trust_region_update!(state::TrustRegionState, params::HeiParametersModified, algoInfo::AlgorithmInfoTR, nlp::AbstractNLPModel)

    # Update step
    accept_or_reject_step!(state, params, algoInfo, nlp)

    # Update trust-region radius based on the acceptance ratio ρ and gradient norm
    if state.ρ > params.η₁
        state.Δ = R_exp(state.ρ, params.η₁, params.β, params.γ₁, params.γ₂, params.M, params.λ1, params.λ2)*norm(state.step)
    else
        state.Δ *= R_exp(state.ρ, params.η₁, params.β, params.γ₁, params.γ₂, params.M, params.λ1, params.λ2)
    end
    
    push!(algoInfo.radii, state.Δ)
end

function trust_region_update!(state::TrustRegionState, params::HeiGradParameters, algoInfo::AlgorithmInfoTR, nlp::AbstractNLPModel)

    # Update step
    accept_or_reject_step!(state, params, algoInfo, nlp)

    # Update trust-region radius based on the acceptance ratio ρ and gradient norm
    state.Δ = R_exp(state.ρ, params.η₁, params.β, params.γ₁, params.γ₂, params.M, params.λ1, params.λ2)*norm(state.gradient)

    push!(algoInfo.radii, state.Δ)
end

function trust_region_update!(state::TrustRegionState, params::HeiGradParametersModified, algoInfo::AlgorithmInfoTR, nlp::AbstractNLPModel)

    # Update step
    accept_or_reject_step!(state, params, algoInfo, nlp)

    params.μ *= R_exp(state.ρ, params.η₁, params.β, params.γ₁, params.γ₂, params.M, params.λ1, params.λ2)

    state.Δ = params.μ * norm(state.gradient)
    
    push!(algoInfo.mus, params.μ)
    push!(algoInfo.radii, state.Δ)
end

function trust_region_update!(state::TrustRegionState, params::ScheinbergParameters, algoInfo::AlgorithmInfoTR, nlp::AbstractNLPModel)

    if state.ρ >= params.η₁ && norm(state.gradient) >= params.ζ * state.Δ
        # Accept step
        state.x = copy(state.x_candidate)
        state.objective = state.f_candidate
        state.gradient = grad(nlp, state.x)
        state.accepted = true

        # Increase trust-region radius
        state.Δ *= params.γ₂
        push!(algoInfo.path, copy(state.x))
        push!(algoInfo.objectives, state.objective)
        push!(algoInfo.gradient_norms, norm(state.gradient))
        push!(algoInfo.accepted_steps, state.iteration)
    else
        # Decrease trust-region radius
        state.Δ *= params.γ₁
    end
    push!(algoInfo.radii, state.Δ)
end

function trust_region_update!(state::TrustRegionState, params::SimpleScheinbergParameters, algoInfo::AlgorithmInfoTR, nlp::AbstractNLPModel)

    # Update step
    accept_or_reject_step!(state, params, algoInfo, nlp)

    # Update trust-region radius based on the acceptance ratio ρ and gradient norm
    if state.ρ >= params.η₁ && norm(state.gradient) >= params.ζ * state.Δ
        state.Δ *= params.γ₂
        push!(algoInfo.path, copy(state.x))
        push!(algoInfo.objectives, state.objective)
        push!(algoInfo.gradient_norms, norm(state.gradient))
        push!(algoInfo.accepted_steps, state.iteration)
    else
        # Decrease trust-region radius
        state.Δ *= params.γ₁
    end
    push!(algoInfo.radii, state.Δ)
end



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