# Truncated Conjugate Gradient (CG) method for solving the trust-region subproblem
function truncated_cg_steihaug(nlp::AbstractNLPModel, x_current, g, Δ; χ::Float64=0.1, θ::Float64=0.5, max_iters=100)
    n = length(g)
    normg = norm(g)

    # Zero-gradient guard: return zero step immediately
    if normg == 0
        return zeros(eltype(g), n), false, 0
    end

    p = zeros(n)  # Initial step
    r = -g        # Initial residual
    d = copy(r)   # Initial search direction
    rs_old = dot(r, r)
    rs_new = rs_old
    
    on_boundary = false  # Flag to indicate if the solution is on the boundary

    for k in 1:max_iters
        Hd = hprod(nlp, x_current, d)  # Hessian-vector product

        if dot(d, Hd) <= 0
            τ = find_trust_region_boundary(p, d, Δ) # Solve for τ such that ‖p + τ * d‖ = Δ
            p += τ * d
            on_boundary = true
            return p, on_boundary, k
        end

        α = rs_old / dot(d, Hd)
        
        # Check if the step exceeds the trust-region boundary
        if norm(p + α * d) > Δ
            τ = find_trust_region_boundary(p, d, Δ) # Solve for τ such that ‖p + τ * d‖ = Δ
            p += τ * d
            on_boundary = true
            return p, on_boundary, k
        end

        # Update the step
        p += α * d

        r_new = r - α * Hd
        rs_new = dot(r_new, r_new)

        # Check for convergence
        if sqrt(rs_new) < min(χ, normg^θ)*normg
            return p, on_boundary, k
        end

        # Update the search direction
        β = rs_new / rs_old
        d = r_new + β * d
        r = r_new
        rs_old = rs_new
    end

    return p, on_boundary, max_iters
end

# Helper function to find the step size τ to stay within the trust region
function find_trust_region_boundary(p, d, Δ)
    a = dot(d, d)
    b = 2 * dot(p, d)
    c = dot(p, p) - Δ^2
    τ = (-b + sqrt(b^2 - 4 * a * c)) / (2 * a)
    return τ
end