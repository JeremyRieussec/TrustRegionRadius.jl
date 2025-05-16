# Structure to hold the state of the algorithm at each iteration
mutable struct GradientDescentState <: AbstractGradientDescentState
    x::Vector{Float64}      # Current point
    gradient::Vector{Float64}   # Gradient at current point
    iter::Int               # Current iteration
    objective::Float64           # Function value at current point\

    function GradientDescentState(x::Vector{Float64}, grad::Vector{Float64}, iter::Int, fval::Float64)
        new(x, grad, iter, fval)
    end
end

function Base.show(io::IO, state::GradientDescentState)
    println(io, "------------ Gradient Descent State --------------")
    println(io, "  Current point (x): ", state.x)
    println(io, "  Gradient: ", state.gradient)
    println(io, "  Iteration: ", state.iter)
    println(io, "  Function value: ", state.objective)
end