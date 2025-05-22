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