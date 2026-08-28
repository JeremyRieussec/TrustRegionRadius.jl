# =============================================================================
# benchmark/experiments/exp3_zeta_sweep.jl
#
# EXPERIMENT 3 -- the criticality parameter ζ of RDFO.
#
# Part II: eventual inactivity of the trust region needs ζ > κ̄, which is
# 4/λ*_min in the :eigenvalue convention and 8/λ*_min in :neighbourhood, the
# default of `kappa_bar` and the convention Part III quotes throughout,
# a constant that depends on the solution and so cannot be chosen a priori.
# Below the threshold the constraint binds at every iteration and convergence
# degrades to linear.
#
# The expected signature is asymmetric, and it is the point of the experiment:
# efficiency (the τ=1 intercept) is nearly flat in ζ, while reliability (the
# right-hand asymptote) collapses for small ζ. A small ζ does not make the
# method slower on the problems it solves; it makes it fail on others.
#
#   julia --project=benchmark benchmark/experiments/exp3_zeta_sweep.jl
# =============================================================================

const ZETAS = [0.001, 0.01, 0.1, 1.0,  10.0, 100.0]

function zeta_sweep()
    arch = ExperimentArchive(tag = "zeta_sweep")
    configs = [(@sprintf("zeta=%g", ζ),
                () -> (rule = RDFO(ζ = ζ),  model = DEFAULT_MODEL(),
                       subsolver = DEFAULT_SUBSOLVER()))
               for ζ in ZETAS]

    save_config(arch; rules = [(c[1], () -> RDFO(ζ = ζ)) for (c, ζ) in zip(configs, ZETAS)],
                params = SOLVER_PARAMS, problem_selection = PROBLEM_SELECTION,
                extra = Dict("experiment" => "exp3_zeta_sweep",
                             "zeta_values" => ZETAS))

    problems = default_problems()
    records  = run_experiment(problems, configs; params = SOLVER_PARAMS, archive = arch)

    labels = [c[1] for c in configs]
    save_table(arch, "exp3_zeta_summary.txt", success_table(records, problems, configs))

    M = metric_matrix(records, problems, configs, :iter)
    τ, prof = performance_profile(M)
    plt = plot(τ, prof; xscale = :log10, label = reshape(labels, 1, :),
               xlabel = "τ (performance ratio, iterations)", ylabel = "π(τ)",
               legend = :bottomright, lw = 2, ylims = (0, 1.02))
    savefig_archived(arch, "exp3_perf_profile_zeta.pdf", plt)
    open(joinpath(arch.tables, "exp3_perf_profile_zeta.tex"), "w") do io
        profile_to_pgfplots(io, τ, prof, labels)
    end

    # solve rate and efficiency against ζ -- the asymmetry, in one figure
    rate = [count(r -> r.config == c[1] && solved(r), records) / length(problems)
            for c in configs]
    eff  = [prof[1, j] for j in 1:length(configs)]
    plt = plot(ZETAS, rate; xscale = :log10, marker = :circle, lw = 2,
               label = "reliability (fraction solved)", xlabel = "ζ",
               ylabel = "fraction", ylims = (0, 1.02), legend = :bottomright)
    plot!(plt, ZETAS, eff; marker = :square, lw = 2, ls = :dash,
          label = "efficiency (π(1), fraction fastest)")
    savefig_archived(arch, "exp3_solverate_vs_zeta.pdf", plt)

    # tail activity: the mechanism behind the reliability loss
    io = IOBuffer()
    @printf(io, "%10s %10s %14s %14s\n", "zeta", "solved", "median iters", "tail active")
    println(io, "-"^52)
    for (j, ζ) in enumerate(ZETAS)
        rs = filter(r -> r.config == configs[j][1] && solved(r), records)
        acts = Float64[]
        for r in rs
            isempty(r.active_traj) && continue
            k0 = max(1, floor(Int, 0.9 * length(r.active_traj)))
            push!(acts, count(r.active_traj[k0:end]) / length(r.active_traj[k0:end]))
        end
        @printf(io, "%10g %10d %14s %14s\n", ζ, length(rs),
                isempty(rs)   ? "--" : @sprintf("%.1f", _median(Float64[r.iterations for r in rs])),
                isempty(acts) ? "--" : @sprintf("%.3f", sum(acts)/length(acts)))
    end
    save_table(arch, "exp3_zeta_activity.txt", String(take!(io)))

    finalize_archive(arch; notes = """
        ζ sweep for RDFO over $(ZETAS).

        exp3_solverate_vs_zeta.pdf plots reliability and efficiency against ζ on
        the same axes. If the survey's account is right the two curves separate:
        efficiency is flat while reliability falls away at small ζ.

        exp3_zeta_activity.txt gives the mechanism. Below the threshold the
        trust-region constraint never stops binding, so the tail-active fraction
        stays near 1 and convergence is linear at best.
        """)
end

if abspath(PROGRAM_FILE) == @__FILE__
    zeta_sweep()
end