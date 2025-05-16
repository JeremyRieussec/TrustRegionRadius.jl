
# Function to initialize a GradientDescentState
function initialize_gradient_descent_state(nlp::AbstractNLPModel)
    return GradientDescentState(nlp.meta.x0, grad(nlp, nlp.meta.x0), 0, obj(nlp, nlp.meta.x0))
end

include("Gradient_decent.jl")
include("GD_backtrack.jl")