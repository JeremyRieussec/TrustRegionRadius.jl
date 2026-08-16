# =============================================================================
# benchmark/experiments/exp2_trajectories.jl
#
# EXPERIMENT 2 -- per-iteration trajectories on selected problems.
#
# Profiles aggregate; this looks inside individual runs. Plots Δ_k, ‖g_k‖, ρ_k
# and the cumulative Σ Δ_k, which is where the mechanism families separate most
# visibly: Part II predicts Δ_k → 0 for the criticality-anchored rules and
# liminf Δ_k > 0 for the others, so ΣΔ_k² converges for the former and diverges
# for the latter.
#
#   julia --project=benchmark benchmark/experiments/exp2_trajectories.jl
# =============================================================================

"Problems to trace. Keep this short: one figure per problem per quantity."
const TRACE_PROBLEMS = ["ROSENBR", "WOOD", "EXTROSEN3"]

function trajectories()
    arch = ExperimentArchive(tag = "trajectories")
    save_config(arch; rules = RULES, params = SOLVER_PARAMS,
                extra = Dict("experiment" => "exp2_trajectories",
                             "traced_problems" => TRACE_PROBLEMS))

    all_problems = default_problems()
    problems = filter(p -> p[1] in TRACE_PROBLEMS, all_problems)
    isempty(problems) && (problems = analytic_problems()[1:min(3, end)])
    configs = rule_configs()
    labels  = [c[1] for c in configs]

    records = run_experiment(problems, configs;
                             params = SOLVER_PARAMS, trace = true, archive = arch,
                             solver_kwargs = (hessian_norm = true,))

    for (pname, _) in problems
        rs = filter(r -> r.problem == pname && solved(r), records)
        isempty(rs) && continue

        # radius
        plt = plot(; xlabel = "iteration k", ylabel = "Δ_k", yscale = :log10,
                     title = pname, legend = :best, lw = 2)
        for r in rs
            isempty(r.delta_traj) && continue
            plot!(plt, max.(r.delta_traj, 1e-300); label = r.config)
        end
        savefig_archived(arch, "exp2_delta_traj_$(pname).pdf", plt)

        # gradient norm
        plt = plot(; xlabel = "iteration k", ylabel = "‖g_k‖", yscale = :log10,
                     title = pname, legend = :best, lw = 2)
        for r in rs
            isempty(r.grad_traj) && continue
            plot!(plt, max.(r.grad_traj, 1e-300); label = r.config)
        end
        savefig_archived(arch, "exp2_grad_traj_$(pname).pdf", plt)

        # ratio
        plt = plot(; xlabel = "iteration k", ylabel = "ρ_k",
                     title = pname, legend = :best, lw = 1.5)
        for r in rs
            isempty(r.ratio_traj) && continue
            plot!(plt, clamp.(r.ratio_traj, -0.5, 2.0); label = r.config)
        end
        hline!(plt, [SOLVER_PARAMS.η1, SOLVER_PARAMS.η2];
               ls = :dash, c = :black, label = "")
        savefig_archived(arch, "exp2_ratio_traj_$(pname).pdf", plt)
    end

    # cumulative Σ Δ_k -- the family separation
    plt = plot(; xlabel = "iteration k", ylabel = "Σ_{j≤k} Δ_j",
                 yscale = :log10, legend = :bottomright, lw = 2)
    for r in filter(solved, records)
        (isempty(r.delta_traj) || r.problem != problems[1][1]) && continue
        plot!(plt, cumsum(r.delta_traj); label = r.config)
    end
    savefig_archived(arch, "exp2_cumsum_summary.pdf", plt)

    # The three series, per run and never averaged across problems: the mean of a
    # divergent series and two convergent ones is not a summary of anything.
    io = IOBuffer()
    @printf(io, "%-16s %-16s %6s %8s %14s %14s %16s\n", "problem", "rule", "iters",
            "status", "Σ Δ_k", "Σ Δ_k²", "Σ Δ_k²/M_k")
    println(io, "-"^96)
    for r in records
        isempty(r.delta_traj) && continue
        rs = radius_sums(RecordView(r))
        @printf(io, "%-16s %-16s %6d %8s %14.4g %14.4g %16s\n",
                r.problem, r.config, r.iterations, string(r.status),
                rs.sum_delta, rs.sum_delta2,
                isnan(rs.sum_delta2_over_M) ? "--" :
                    @sprintf("%.4g", rs.sum_delta2_over_M))
    end
    println(io)
    println(io, "Σ Δ_k²/M_k with M_k = max_{i≤k}‖H_i‖ is the quantity Part II proves")
    println(io, "finite for the criticality-anchored rules. Σ Δ_k² is equivalent to it")
    println(io, "only while ‖H_k‖ stays bounded, which is the hypothesis the M_k form")
    println(io, "exists to drop, so the two columns separate exactly on the runs where")
    println(io, "the model curvature grows.")
    save_table(arch, "exp2_radius_sums.txt", String(take!(io)))

    finalize_archive(arch; notes = """
        Per-iteration trajectories on $(join([p[1] for p in problems], ", ")).

        exp2_radius_sums.txt reports Σ Δ_k and Σ Δ_k². Part II predicts that the
        criticality-anchored rules (RDFO, RGrad) drive Δ_k → 0 with Σ Δ_k²
        convergent, while RDelta keeps liminf Δ_k > 0 so both series diverge.
        A geometrically growing radius will overflow Σ Δ_k² in double precision;
        that is an overflow, not a proof of divergence, though the series does
        diverge.
        """)
end

if abspath(PROGRAM_FILE) == @__FILE__
    trajectories()
end