# =============================================================================
# benchmark/experiments/exp5_inactivity.jl
#
# EXPERIMENT 5 -- direct measurement of trust-region inactivity.
#
# The claim under test is that the mechanisms split cleanly into those whose
# trust-region constraint eventually stops binding and those where it binds for
# ever, and that this split is invisible to a first-order convergence test:
# both groups have ‖g_k‖ decreasing monotonically to zero.
#
# The observable is the fraction of late iterations with ‖s_k‖ = Δ_k, which the
# solver records in :active_trajectory when trace = true.
#
#   julia --project=benchmark benchmark/experiments/exp5_inactivity.jl
# =============================================================================

using Plots
include(joinpath(@__DIR__, "..", "archive.jl"))
include(joinpath(@__DIR__, "..", "harness.jl"))
include(joinpath(@__DIR__, "..", "config.jl"))

"Fraction of the last `frac` of iterations on which the constraint bound."
function tail_active(r::RunRecord; frac::Float64 = 0.1)
    isempty(r.active_traj) && return NaN
    k0 = max(1, floor(Int, (1 - frac) * length(r.active_traj)))
    tail = r.active_traj[k0:end]
    return count(tail) / length(tail)
end

function main()
    arch = ExperimentArchive(tag = "inactivity")

    # Rules spanning both families, plus a deliberately-too-small parameter.
    rules = [
        ("RDelta",       () -> RDelta()),
        ("RStep",        () -> RStep()),
        ("RDFO(z=1)",    () -> RDFO(ζ = 1.0)),
        ("RDFO(z=0.01)", () -> RDFO(ζ = 0.01)),
        ("RGrad",        () -> RGrad()),
        ("RGradCapped",  () -> RGradCapped(μ = 0.05, μ_max = 0.05)),
    ]
    configs = rule_configs(rules)

    save_config(arch; rules = rules, params = SOLVER_PARAMS,
                problem_selection = PROBLEM_SELECTION,
                extra = Dict("experiment" => "exp5_inactivity"))

    problems = default_problems()
    records  = run_experiment(problems, configs; params = SOLVER_PARAMS,
                              trace = true, archive = arch)
    labels   = [c[1] for c in configs]

    # tail activity per configuration
    io = IOBuffer()
    @printf(io, "%-16s %10s %14s %14s %14s\n",
            "rule", "solved", "tail active", "Σ Δ_k", "Σ Δ_k²")
    println(io, "-"^74)
    means = Float64[]
    for (cname, _) in configs
        rs = filter(r -> r.config == cname && solved(r), records)
        ta = filter(!isnan, [tail_active(r) for r in rs])
        sd  = [sum(r.delta_traj)       for r in rs if !isempty(r.delta_traj)]
        sd2 = [sum(abs2, r.delta_traj) for r in rs if !isempty(r.delta_traj)]
        m = isempty(ta) ? NaN : sum(ta)/length(ta)
        push!(means, m)
        @printf(io, "%-16s %10d %14s %14s %14s\n", cname, length(rs),
                isnan(m)     ? "--" : @sprintf("%.3f", m),
                isempty(sd)  ? "--" : @sprintf("%.3g", sum(sd)/length(sd)),
                isempty(sd2) ? "--" : @sprintf("%.3g", sum(sd2)/length(sd2)))
    end
    save_table(arch, "exp5_inactivity_summary.txt", String(take!(io)))

    plt = bar(labels, means; ylabel = "mean tail-active fraction",
              legend = false, xrotation = 45, ylims = (0, 1.02))
    hline!(plt, [0.5]; ls = :dash, c = :black)
    savefig_archived(arch, "exp5_tail_active.pdf", plt)

    # activity along the run, on one problem
    pname = problems[1][1]
    plt = plot(; xlabel = "iteration k", ylabel = "cumulative active fraction",
                 title = pname, legend = :best, lw = 2, ylims = (0, 1.02))
    for r in filter(r -> r.problem == pname && solved(r), records)
        isempty(r.active_traj) && continue
        plot!(plt, cumsum(r.active_traj) ./ (1:length(r.active_traj)); label = r.config)
    end
    savefig_archived(arch, "exp5_active_trajectory_$(pname).pdf", plt)

    finalize_archive(arch; notes = """
        Direct measurement of trust-region inactivity.

        The tail-active fraction is the share of the last 10% of iterations on
        which ‖s_k‖ = Δ_k. Values near 0 mean the constraint has stopped
        binding, so the step is the unconstrained model minimiser and the method
        inherits the local rate of the underlying (quasi-)Newton iteration.
        Values near 1 mean it never stops binding.

        RDFO(z=0.01) and RGradCapped(mu=0.05) are included precisely because
        their parameters sit below any plausible κ̄ = 4/λ*_min: they should stay
        pinned near 1 while every other rule falls to 0. Note that all of them
        still drive ‖g_k‖ → 0, which is why a first-order convergence test
        cannot tell the two groups apart.
        """)
end

abspath(PROGRAM_FILE) == @__FILE__ && main()
