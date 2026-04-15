
# ============================================================
# trust_region_solver — generic solver for canonical R1–R4
#
# Uses the AbstractRadiusUpdate dispatch interface rather than
# the legacy AbstractTrustRegionParameters interface.
# Returns TROutput for benchmarking.
# ============================================================

"""
    trust_region_solver(nlp, rule, params) -> TROutput

Trust-region method with truncated CG Steihaug subproblem solver.
The radius update is fully determined by `rule::AbstractRadiusUpdate`.

# Arguments
- `nlp::AbstractNLPModel`:      problem (provides obj, grad, hprod)
- `rule::AbstractRadiusUpdate`: one of R1ClassicalUpdate, R2StepSizeUpdate,
                                 R3DFOLikeUpdate, R4RelativeGradUpdate
- `params::TRSolverParams`:     η₁, η₂, Δ₀, max_iterations, tol

# Returns
`TROutput` with status `:solved`, `:max_iter`, or `:failure`,
trajectory vectors, and timing.

# Notes
- Objective evaluations (`f_evals`) count only `obj` calls; gradient
  evaluations are separate (tracked by NLPModels counters).
- The step is accepted whenever ρ ≥ η₁.
- For R4 the initial radius is rule.μ · ‖g₀‖ (ignoring params.Δ₀).
"""
function trust_region_solver(nlp::AbstractNLPModel,
                              rule::AbstractRadiusUpdate,
                              params::TRSolverParams)

    t_start = time()

    # ----------------------------------------------------------
    # Initialisation
    # ----------------------------------------------------------
    x    = copy(nlp.meta.x0)
    g    = grad(nlp, x)
    f    = obj(nlp, x)
    #f_evals = 1    # counts obj() calls only

    g_norm = norm(g)
    Δ = initial_radius(rule, params.Δ₀, g_norm)

    delta_traj    = Float64[Δ]
    grad_traj     = Float64[g_norm]
    obj_traj      = Float64[f]

    # ----------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------
    for k in 1:params.max_iterations

        # Convergence check (before computing anything for this iter)
        if g_norm <= params.tol
            return TROutput(:solved, k - 1, 
                            CUTEst.neval_obj(nlp), 
                            CUTEst.neval_grad(nlp),
                            CUTEst.neval_hess(nlp),
                            CUTEst.neval_hprod(nlp),
                            g_norm, Δ,
                            delta_traj, grad_traj, obj_traj,
                            time() - t_start)
        end

        # Solve trust-region subproblem via truncated CG Steihaug
        p, _, _ = truncated_cg_steihaug(nlp, x, g, Δ)
        s_norm  = norm(p)

        # Candidate evaluation
        x_cand  = x + p
        f_cand  = obj(nlp, x_cand)
            # f_evals += 1

        # Predicted reduction  m(0) - m(p) = -gᵀp - ½pᵀHp
        Hp        = hprod(nlp, x, p)
        predicted = -dot(g, p) - 0.5 * dot(p, Hp)

        # Actual reduction
        actual = f - f_cand

        # Ratio ρ (guard against zero or negative predicted reduction)
        ρ = predicted > 0.0 ? actual / predicted : -Inf

        # Store gradient norm BEFORE acceptance (needed by R3)
        g_norm_old = g_norm

        # Accept / reject
        if ρ >= params.η₁
            x = x_cand
            f = f_cand
            g = grad(nlp, x)
            g_norm = norm(g)
        end
        # g_norm_new reflects the iterate after the accept/reject decision
        g_norm_new = g_norm

        # Update radius
        Δ = update_radius!(rule, Δ, ρ, params.η₁, params.η₂,
                           s_norm, g_norm_old, g_norm_new)

        push!(delta_traj, Δ)
        push!(grad_traj,  g_norm_new)
        push!(obj_traj,   f)
    end

    # ----------------------------------------------------------
    # Maximum iterations reached
    # ----------------------------------------------------------
    @info "trust_region_solver: maximum iterations ($(params.max_iterations)) reached without convergence."
    return TROutput(:max_iter, params.max_iterations, 
                    CUTEst.neval_obj(nlp), 
                    CUTEst.neval_grad(nlp),
                    CUTEst.neval_hess(nlp),
                    CUTEst.neval_hprod(nlp),
                    g_norm, Δ,
                    delta_traj, grad_traj, obj_traj,
                    time() - t_start)
end
