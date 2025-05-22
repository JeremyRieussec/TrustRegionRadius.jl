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

function generate_TR_state(params::Union{YuanFanParameters, HeiFanYuanParameters}, nlp::AbstractNLPModel)

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

