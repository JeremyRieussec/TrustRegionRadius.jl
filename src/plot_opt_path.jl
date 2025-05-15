# Plot the optimization path
function plot_opt_path(path, objectives, f_objective)
    min_x1 = floor(minimum(x[1] for x in path))
    max_x1 = ceil(maximum(x[1] for x in path))
    min_x2 = floor(minimum(x[2] for x in path))
    max_x2 = ceil(maximum(x[2] for x in path))

    x1 = range(minimum([min_x1, -1]), maximum([1.5, max_x1]), length=100)
    x2 = range(minimum([min_x2, -1]), maximum([1.5, max_x2]), length=100)
    
    # Contour plot of the Rosenbrock function
    p1 = contour(x1, x2, f_objective, title="Rosenbrock Function", xlabel="x₁", ylabel="x₂", colorbar=true)
    scatter!(p1, [x[1] for x in path], [x[2] for x in path], label="Path", color=:red, markersize=3)
    scatter!(p1, [path[1][1]], [path[1][2]], label="Start", color=:green, markersize=5)
    scatter!(p1, [path[end][1]], [path[end][2]], label="End", color=:blue, markersize=5)
    
    # Convergence plot
    p2 = plot(objectives, title="Objective Value", xlabel="Iteration", ylabel=L"f(x_k)", legend=false, linewidth=2)
    
    plot(p1, p2, layout=(1,2), size=(1200,400))
end
